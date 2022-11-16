import numpy as np
from scipy import constants

from desilike.base import BaseCalculator
from desilike.theories.utils import get_cosmo, external_cosmo
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
            self.cosmo = self.fiducial
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'of': [('theta_cb', 'theta_cb')]}}}

    def calculate(self):
        fo = self.cosmo.get_fourier()
        self.sigma8 = fo.sigma8_z(self.z, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.z, of='theta_cb')
        self.pk_tt_interpolator = fo.pk_interpolator(of='theta_cb').to_1d(z=self.z)
        self.f = self.fsigma8 / self.sigma8
        if self.with_now:
            if getattr(self, 'filter', None) is None:
                from cosmoprimo import PowerSpectrumBAOFilter
                self.filter = PowerSpectrumBAOFilter(self.pk_tt_interpolator, engine=self.with_now, cosmo=self.cosmo, cosmo_fid=self.fiducial)
            else:
                self.filter(self.pk_tt_interpolator, cosmo=self.cosmo)
            self.pknow_tt_interpolator = self.filter.smooth_pk_interpolator()
        return self


class BasePowerSpectrumTemplate(BasePowerSpectrumExtractor):

    def initialize(self, k=None, z=1., **kwargs):
        super(BasePowerSpectrumTemplate, self).initialize(z=z, **kwargs)
        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        self.cosmo_requires = {}

    def calculate(self):
        super(BasePowerSpectrumTemplate, self).calculate()
        self.pk_tt = self.pk_tt_interpolator(self.k)
        if self.with_now:
            self.pknow_tt = self.pknow_tt_interpolator(self.k)
        return self


class FullPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def initialize(self, *args, **kwargs):
        super(FullPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, cosmo=self.cosmo, mode='distances').runtime_info.initialize()
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'k': self.k, 'of': [('theta_cb', 'theta_cb')]}}}
            self.cosmo_requires.update(self.apeffect.cosmo_requires)  # just background
            self.params.clear()

    def calculate(self, **params):
        if params:
            self.cosmo = self.cosmo.clone(**params)
        self.apeffect.cosmo = self.cosmo
        self.apeffect.runtime_info.calculate()
        return super(FullPowerSpectrumTemplate, self).calculate()

    def ap_k_mu(self, k, mu):
        return self.apeffect.ap_k_mu(k, mu)


class ShapeFitPowerSpectrumExtractor(BasePowerSpectrumExtractor):

    def initialize(self, *args, kp=0.03, n_varied=False, with_now='peakaverage', **kwargs):
        super(ShapeFitPowerSpectrumExtractor, self).initialize(*args, with_now=with_now, **kwargs)
        self.kp = float(kp)
        self.n_varied = bool(n_varied)
        if external_cosmo(self.cosmo):
            self.cosmo_requires['primordial'] = {'pk_interpolator': {'k': self.k}}

    def calculate(self):
        super(ShapeFitPowerSpectrumExtractor, self).calculate()
        self.f_sqrt_Ap = self.pknow_tt_interpolator(self.kp)**0.5
        self.Ap = self.f_sqrt_Ap**2 / self.f**2
        self.n = self.cosmo.n_s
        dk = 1e-2
        k = self.kp * np.array([1. - dk, 1. + dk])
        if self.n_varied:
            pk_prim = self.cosmo.get_primordial().pk_interpolator()(k) * k
        else:
            pk_prim = 1.
        self.m = (np.diff(np.log(self.pknow_tt_interpolator(k) / pk_prim)) / np.diff(np.log(k)))[0]
        self.kp_rs = self.kp * self.cosmo.rs_drag
        return self


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate, ShapeFitPowerSpectrumExtractor):

    def initialize(self, *args, a=0.6, apmode='qparqper', **kwargs):
        super(ShapeFitPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.a = float(a)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode=apmode)

    def calculate(self, f=0.8, dm=0., dn=0.):
        self.n_varied = self.runtime_info.base_params['dn'].varied
        super(ShapeFitPowerSpectrumTemplate, self).calculate()
        factor = np.exp(dm / self.a * np.tanh(self.a * np.log(self.k / self.kp)) + dn * np.log(self.k / self.kp))
        self.pk_tt *= factor
        if self.with_now:
            self.pknow_tt *= factor
        self.n += dn
        self.m += dm
        self.qpar, self.qper = self.apeffect.qpar, self.apeffect.qper
        self.f = f
        self.f_sqrt_Ap = f * self.Ap**0.5
        return self

    def ap_k_mu(self, k, mu):
        return self.apeffect.ap_k_mu(k, mu)


class BAOExtractor(BaseCalculator):

    def initialize(self, z=1., cosmo=None, fiducial='DESI'):
        self.z = float(z)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo = cosmo
        if cosmo is None: self.cosmo = self.fiducial
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'thermodynamics': {'rs_drag': None},
                                   'background': {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}}

    def calculate(self):
        rd = self.cosmo.rs_drag
        self.DH_over_rd = constants.c / 1e3 / (100. * self.cosmo.efunc(self.z)) / rd
        self.DM_over_rd = self.cosmo.comoving_angular_distance(self.z) / rd
        self.DH_over_DM = self.DH_over_rd / self.DM_over_rd
        self.DV_over_rd = (self.DH_over_rd * self.DM_over_rd**2 * self.z)**(1. / 3.)
        if self.fiducial is not None:
            self.DH_over_rd_fid = constants.c / 1e3 / (100. * self.fiducial.efunc(self.z)) / rd
            self.DM_over_rd_fid = self.fiducial.comoving_angular_distance(self.z) / rd
            self.DH_over_DM_fid = self.DH_over_rd_fid / self.DM_over_rd_fid
            self.DV_over_rd_fid = (self.DH_over_rd_fid * self.DM_over_rd_fid**2 * self.z)**(1. / 3.)
            self.qpar = self.DH_over_rd / self.DH_over_rd_fid
            self.qper = self.DM_over_rd / self.DM_over_rd_fid
