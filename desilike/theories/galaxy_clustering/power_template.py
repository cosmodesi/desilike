import numpy as np
from scipy import constants

from desilike.base import BaseCalculator
from desilike.theories.primordial_cosmology import get_cosmo, external_cosmo, Cosmoprimo
from .base import APEffect


class BasePowerSpectrumExtractor(BaseCalculator):

    config_fn = 'power_template.yaml'

    def initialize(self, z=1., with_now=False, cosmo=None, fiducial='DESI'):
        self.z = float(z)
        self.fiducial = get_cosmo(fiducial)
        self.with_now = with_now
        self.cosmo_requires = {}
        self.cosmo = cosmo
        if cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'of': [('delta_cb', 'delta_cb')]}}}

    def calculate(self):
        fo = self.cosmo.get_fourier()
        self.sigma8 = fo.sigma8_z(self.z, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.z, of='theta_cb')
        self.pk_dd_interpolator = fo.pk_interpolator(of='delta_cb').to_1d(z=self.z)
        self.f = self.fsigma8 / self.sigma8
        if self.with_now:
            if getattr(self, 'filter', None) is None:
                from cosmoprimo import PowerSpectrumBAOFilter
                self.filter = PowerSpectrumBAOFilter(self.pk_dd_interpolator, engine=self.with_now, cosmo=self.cosmo, cosmo_fid=self.fiducial)
            else:
                self.filter(self.pk_dd_interpolator, cosmo=self.cosmo)
            self.pknow_dd_interpolator = self.filter.smooth_pk_interpolator()


class BasePowerSpectrumTemplate(BasePowerSpectrumExtractor):

    def initialize(self, k=None, z=1., **kwargs):
        super(BasePowerSpectrumTemplate, self).initialize(z=z, **kwargs)
        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        self.cosmo_requires = {}

    def calculate(self):
        super(BasePowerSpectrumTemplate, self).calculate()
        self.pk_dd = self.pk_dd_interpolator(self.k)
        if self.with_now:
            self.pknow_dd = self.pknow_dd_interpolator(self.k)

    @property
    def qpar(self):
        return self.apeffect.qpar

    @property
    def qper(self):
        return self.apeffect.qper

    def ap_k_mu(self, k, mu):
        return self.apeffect.ap_k_mu(k, mu)


class FixedPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def initialize(self, *args, **kwargs):
        super(FixedPowerSpectrumTemplate, self).initialize(*args, cosmo=None, **kwargs)
        self.cosmo = self.fiducial

    @property
    def qpar(self):
        return 1.

    @property
    def qper(self):
        return 1.

    def ap_k_mu(self, k, mu):
        return 1., k[..., None], mu


class FullPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def initialize(self, *args, **kwargs):
        super(FullPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, cosmo=self.cosmo, mode='distances').runtime_info.initialize()
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'k': self.k, 'of': [('delta_cb', 'delta_cb')]}}}
            self.cosmo_requires.update(self.apeffect.cosmo_requires)  # just background
        else:
            self.cosmo.params = self.params.copy()
        self.params.clear()


class ShapeFitPowerSpectrumExtractor(BasePowerSpectrumExtractor):

    def initialize(self, *args, kp=0.03, n_varied=False, with_now='peakaverage', **kwargs):
        super(ShapeFitPowerSpectrumExtractor, self).initialize(*args, with_now=with_now, **kwargs)
        self.kp = float(kp)
        self.n_varied = bool(n_varied)
        if external_cosmo(self.cosmo):
            self.cosmo_requires['primordial'] = {'pk_interpolator': {'k': self.k}}

    def calculate(self):
        super(ShapeFitPowerSpectrumExtractor, self).calculate()
        kp = self.kp * self.fiducial.rs_drag / self.cosmo.rs_drag
        self.Ap = self.pknow_dd_interpolator(kp)
        self.f_sqrt_Ap = self.f * self.Ap**0.5
        self.n = self.cosmo.n_s
        dk = 1e-2
        k = self.kp * np.array([1. - dk, 1. + dk])
        if self.n_varied:
            pk_prim = self.cosmo.get_primordial().pk_interpolator()(k) * k
        else:
            pk_prim = 1.
        self.m = (np.diff(np.log(self.pknow_dd_interpolator(k) / pk_prim)) / np.diff(np.log(k)))[0]


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate, ShapeFitPowerSpectrumExtractor):

    def initialize(self, *args, a=0.6, apmode='qparqper', **kwargs):
        super(ShapeFitPowerSpectrumTemplate, self).initialize(*args, cosmo=None, **kwargs)
        self.a = float(a)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode=apmode)
        for param in list(self.params):
            if param in self.apeffect.params:
                self.apeffect.params.set(param)
                del self.params[param]
        self.cosmo = self.fiducial

    def calculate(self, f=0.8, dm=0., dn=0.):
        self.n_varied = self.runtime_info.base_params['dn'].varied
        super(ShapeFitPowerSpectrumTemplate, self).calculate()
        factor = np.exp(dm / self.a * np.tanh(self.a * np.log(self.k / self.kp)) + dn * np.log(self.k / self.kp))
        self.pk_dd *= factor
        if self.with_now:
            self.pknow_dd *= factor
        self.n += dn
        self.m += dm
        self.f = f
        self.f_sqrt_Ap = f * self.Ap**0.5


class BAOExtractor(BaseCalculator):

    def initialize(self, z=1., cosmo=None, fiducial='DESI'):
        self.z = float(z)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo = cosmo
        if cosmo is None: self.cosmo = self.fiducial
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'thermodynamics': {'rs_drag': None},
                                   'background': {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}}
        if self.fiducial is not None:
            cosmo = self.cosmo
            self.cosmo = self.fiducial
            self.calculate()
            self.cosmo = cosmo
            for name in ['DH_over_rd', 'DM_over_rd', 'DH_over_DM', 'DV_over_rd']:
                setattr(self, name + '_fid', getattr(self, name))

    def calculate(self):
        rd = self.cosmo.rs_drag
        self.DH_over_rd = constants.c / 1e3 / (100. * self.cosmo.efunc(self.z)) / rd
        self.DM_over_rd = self.cosmo.comoving_angular_distance(self.z) / rd
        self.DH_over_DM = self.DH_over_rd / self.DM_over_rd
        self.DV_over_rd = (self.DH_over_rd * self.DM_over_rd**2 * self.z)**(1. / 3.)

    def get(self):
        if self.fiducial is not None:
            self.qpar = self.DH_over_rd / self.DH_over_rd_fid
            self.qper = self.DM_over_rd / self.DM_over_rd_fid
            self.qiso = self.DV_over_rd / self.DV_over_rd_fid
            self.qap = self.DH_over_DM / self.DH_over_DM_fid
        return self


class BAOPowerSpectrumTemplate(BasePowerSpectrumExtractor):

    def initialize(self, *args, apmode='qparqper', with_now='peakaverage', **kwargs):
        super(BAOPowerSpectrumTemplate, self).initialize(*args, cosmo=None, with_now=with_now, **kwargs)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode=apmode)
        for param in list(self.params):
            if param in self.apeffect.params:
                self.apeffect.params.set(param)
                del self.params[param]
        self.cosmo = self.fiducial
        # Set DM_over_rd, etc.
        BAOExtractor.calculate(self)
        for name in ['DH_over_rd', 'DM_over_rd', 'DH_over_DM', 'DV_over_rd']:
            setattr(self, name + '_fid', getattr(self, name))
        # No self.k defined

    def get(self):
        self.DH_over_rd = self.qpar * self.DH_over_rd_fid
        self.DM_over_rd = self.qper * self.DM_over_rd_fid
        self.DV_over_rd = self.apeffect.qiso * self.DV_over_rd_fid
        self.DH_over_DM = self.apeffect.qap * self.DH_over_DM
        return self

    @property
    def qpar(self):
        return self.apeffect.qpar

    @property
    def qper(self):
        return self.apeffect.qper

    def ap_k_mu(self, k, mu):
        return self.apeffect.ap_k_mu(k, mu)
