import numpy as np
from scipy import constants

from desilike.base import BaseCalculator
from desilike.parameter import ParameterCollection
from desilike.theories.primordial_cosmology import get_cosmo, external_cosmo, Cosmoprimo
from .base import APEffect


class BasePowerSpectrumExtractor(BaseCalculator):

    """Base class to extract shape parameters from linear power spectrum."""
    config_fn = 'power_template.yaml'

    def initialize(self, z=1., with_now=False, cosmo=None, fiducial='DESI'):
        self.z = float(z)
        self.fiducial = get_cosmo(fiducial)
        self.with_now = with_now
        self.cosmo_requires = {}
        self.cosmo = cosmo
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'of': [('delta_cb', 'delta_cb')]}}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            self.cosmo.params = self.params.copy()
        self.params.clear()

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

    """Base class for linear power spectrum template."""
    config_fn = 'power_template.yaml'

    def initialize(self, k=None, z=1., with_now=False, apmode='qparqper', fiducial='DESI'):
        self.z = float(z)
        self.cosmo = self.fiducial = get_cosmo(fiducial)
        self.with_now = with_now
        if k is None: k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        self.cosmo_requires = {}
        super(BasePowerSpectrumTemplate, self).calculate()
        for name in ['f', 'sigma8', 'pk_dd_interpolator'] + (['pknow_dd_interpolator'] if self.with_now else []):
            setattr(self, name + '_fid', getattr(self, name))
        self.pk_dd_fid = self.pk_dd_interpolator(self.k)
        if self.with_now:
            self.pknow_dd_fid = self.pknow_dd_interpolator(self.k)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode=apmode)
        ap_params = ParameterCollection()
        for param in list(self.params):
            if param in self.apeffect.params:
                ap_params.set(param)
                del self.params[param]
        self.apeffect.params = ap_params

    def calculate(self):
        self.pk_dd = self.pk_dd_fid
        if self.with_now:
            self.pknow_dd = self.pknow_dd_fid

    @property
    def qpar(self):
        return self.apeffect.qpar

    @property
    def qper(self):
        return self.apeffect.qper

    def ap_k_mu(self, k, mu):
        return self.apeffect.ap_k_mu(k, mu)


class FixedPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    Fixed power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    with_now : str, default=False
        If provided, also compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, *args, **kwargs):
        super(FixedPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.runtime_info.requires = []  # remove APEffect dependence


class DirectPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def initialize(self, *args, k=None, **kwargs):
        if k is None: k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        BasePowerSpectrumExtractor.initialize(self, *args, **kwargs)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, cosmo=self.cosmo, mode='distances').runtime_info.initialize()
        if external_cosmo(self.cosmo):
            self.cosmo_requires['fourier']['pk_interpolator']['k'] = self.k
            self.cosmo_requires.update(self.apeffect.cosmo_requires)  # just background

    def calculate(self):
        BasePowerSpectrumExtractor.calculate(self)
        self.pk_dd = self.pk_dd_interpolator(self.k)
        if self.with_now:
            self.pknow_dd = self.pknow_dd_interpolator(self.k)


class ShapeFitPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    """
    Extract ShapeFit parameters from linear power spectrum.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    kp : float, default=0.03
        Pivot point in ShapeFit parameterization.

    n_varied : bool, default=False
        Use second order ShapeFit parameter ``n``.
        This choice changes the definition of parameter ``m``.

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    def initialize(self, *args, kp=0.03, n_varied=False, with_now='peakaverage', **kwargs):
        super(ShapeFitPowerSpectrumExtractor, self).initialize(*args, with_now=with_now, **kwargs)
        self.kp = float(kp)
        self.n_varied = bool(n_varied)
        if external_cosmo(self.cosmo):
            self.cosmo_requires['primordial'] = {'pk_interpolator': {'k': self.k}}
        if self.fiducial is not None:
            cosmo = self.cosmo
            self.cosmo = self.fiducial
            self.calculate()
            self.cosmo = cosmo
            for name in ['f', 'n', 'm', 'Ap']:
                setattr(self, name + '_fid', getattr(self, name))

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

    def get(self):
        self.dn = self.n - self.n_fid
        self.dm = self.m - self.m_fid
        self.df = self.f_sqrt_Ap / (self.f_fid * self.Ap_fid**0.5)
        return self


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate, ShapeFitPowerSpectrumExtractor):
    r"""
    ShapeFit power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    kp : float, default=0.03
        Pivot point in ShapeFit parameterization.

    a : float, default=0.6
        :math:`a` parameter in ShapeFit parameterization.

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, default='DESI'
        Fiducial cosmology, used to compute the power spectrum.


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    def initialize(self, *args, kp=0.03, a=0.6, with_now='peakaverage', **kwargs):
        self.kp = float(kp)
        self.n_varied = self.params['dn'].varied
        super(ShapeFitPowerSpectrumTemplate, self).initialize(*args, with_now=with_now, **kwargs)
        self.a = float(a)
        for name in ['n', 'm', 'Ap']:
            setattr(self, name + '_fid', getattr(self, name))

    def calculate(self, df=1., dm=0., dn=0.):
        factor = np.exp(dm / self.a * np.tanh(self.a * np.log(self.k / self.kp)) + dn * np.log(self.k / self.kp))
        self.pk_dd = self.pk_dd_fid * factor
        if self.with_now:
            self.pknow_dd = self.pknow_dd_fid * factor
        self.n = self.n_fid + dn
        self.m = self.m_fid + dm
        self.f = self.f_fid * df
        self.f_sqrt_Ap = self.f * self.Ap_fid**0.5

    def get(self):
        return self


class StandardPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    r"""
    Extract standard RSD parameters :math:`(q_{\parallel}, q_{\perp}, df)`.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    r : float, default=8.
        Sphere radius to estimate the normalization of the linear power spectrum.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    """
    def initialize(self, *args, r=8., **kwargs):
        self.r = float(r)
        super(StandardPowerSpectrumExtractor, self).initialize(*args, **kwargs)
        self.apeffect = APEffect(z=self.z, cosmo=self.cosmo, fiducial=self.fiducial, mode='distances')
        if self.fiducial is not None:
            cosmo = self.cosmo
            self.cosmo = self.fiducial
            self.apeffect.qiso = 1.
            self.calculate()
            self.cosmo = cosmo
            for name in ['f', 'sigmar', 'fsigmar']:
                setattr(self, name + '_fid', getattr(self, name))

    def calculate(self):
        r = self.r * self.apeffect.qiso
        fo = self.cosmo.get_fourier()
        self.sigmar = fo.sigma_rz(r, self.z, of='delta_cb')
        self.fsigmar = fo.sigma_rz(r, self.z, of='theta_cb')
        self.f = self.fsigmar / self.sigmar

    def get(self):
        self.df = self.fsigmar / self.fsigmar_fid
        return self


class StandardPowerSpectrumTemplate(BasePowerSpectrumTemplate, StandardPowerSpectrumExtractor):
    r"""
    Standard power spectrum template, in terms of :math:`f` and Alcock-Paczynski parameters.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    r : float, default=8.
        Sphere radius to estimate the normalization of the linear power spectrum.

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, default='DESI'
        Fiducial cosmology, used to compute the power spectrum.
    """
    def initialize(self, *args, r=8., **kwargs):
        self.r = float(r)
        super(StandardPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        for name in ['f', 'sigmar', 'fsigmar']:
            setattr(self, name + '_fid', getattr(self, name))

    def calculate(self, df=1.):
        self.f = self.f_fid * df

    def get(self):
        self.fsigmar = self.f * self.sigmar_fid
        return self


class BAOExtractor(BaseCalculator):
    """
    Extract BAO parameters from base cosmological parameters.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    """
    config_fn = 'power_template.yaml'

    def initialize(self, z=1., cosmo=None, fiducial='DESI'):
        self.z = float(z)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo = cosmo
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'thermodynamics': {'rs_drag': None},
                                   'background': {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            self.cosmo.params = self.params.copy()
        self.params.clear()
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


class BAOPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    BAO power spectrum template.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, *args, with_now='peakaverage', **kwargs):
        super(BAOPowerSpectrumTemplate, self).initialize(*args, with_now=with_now, **kwargs)
        # Set DM_over_rd, etc.
        BAOExtractor.calculate(self)
        for name in ['DH_over_rd', 'DM_over_rd', 'DH_over_DM', 'DV_over_rd']:
            setattr(self, name + '_fid', getattr(self, name))
        # No self.k defined

    def calculate(self, df=1.):
        super(BAOPowerSpectrumTemplate, self).calculate()
        self.f = self.f_fid * df

    def get(self):
        self.DH_over_rd = self.qpar * self.DH_over_rd_fid
        self.DM_over_rd = self.qper * self.DM_over_rd_fid
        self.DV_over_rd = self.apeffect.qiso * self.DV_over_rd_fid
        self.DH_over_DM = self.apeffect.qap * self.DH_over_DM_fid
        return self
