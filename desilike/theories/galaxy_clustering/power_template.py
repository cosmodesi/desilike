import numpy as np
from scipy import constants
from cosmoprimo import PowerSpectrumBAOFilter

from desilike.jax import numpy as jnp
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
        self.cosmo_requires = {}
        self.cosmo = cosmo
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'of': [('delta_cb', 'delta_cb')]}}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            self.cosmo.params = self.params.copy()
        self.params.clear()
        cosmo = self.cosmo
        self.cosmo = self.fiducial
        self.with_now = False
        BasePowerSpectrumExtractor.calculate(self)
        self.cosmo = cosmo
        for name in ['sigma8', 'fsigma8', 'f', 'pk_dd_interpolator']:
            setattr(self, name + '_fid', getattr(self, name))
            delattr(self, name)
        self.with_now = with_now
        if self.with_now:
            self.filter = PowerSpectrumBAOFilter(self.pk_dd_interpolator_fid, engine=self.with_now, cosmo=self.fiducial, cosmo_fid=self.fiducial)
            self.pknow_dd_interpolator_fid = self.filter.smooth_pk_interpolator()

    def calculate(self):
        fo = self.cosmo.get_fourier()
        self.sigma8 = fo.sigma8_z(self.z, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.z, of='theta_cb')
        self.f = self.fsigma8 / self.sigma8
        self.pk_dd_interpolator = fo.pk_interpolator(of='delta_cb').to_1d(z=self.z)
        if self.with_now:
            self.filter(self.pk_dd_interpolator, cosmo=self.cosmo)
            self.pknow_dd_interpolator = self.filter.smooth_pk_interpolator()


class BasePowerSpectrumTemplate(BasePowerSpectrumExtractor):

    """Base class for linear power spectrum template."""
    config_fn = 'power_template.yaml'

    def initialize(self, k=None, z=1., with_now=False, apmode='qparqper', fiducial='DESI', only_now=False):
        self.z = float(z)
        self.cosmo = self.fiducial = get_cosmo(fiducial)
        if k is None: k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        self.cosmo_requires = {}
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode=apmode)
        ap_params = ParameterCollection()
        for param in list(self.params):
            if param in self.apeffect.params:
                ap_params.set(param)
                del self.params[param]
        self.apeffect.params = ap_params
        self.with_now = False
        BasePowerSpectrumExtractor.calculate(self)
        self.with_now = with_now
        if only_now and not self.with_now:
            self.with_now = only_now
        self.only_now = bool(only_now)
        for name in ['sigma8', 'fsigma8', 'f', 'pk_dd_interpolator']:
            setattr(self, name + '_fid', getattr(self, name))
            delattr(self, name)
        self.pk_dd_fid = self.pk_dd_interpolator_fid(self.k)
        if self.with_now:
            self.filter = PowerSpectrumBAOFilter(self.pk_dd_interpolator_fid, engine=self.with_now, cosmo=self.fiducial, cosmo_fid=self.fiducial)
            self.pknow_dd_interpolator_fid = self.filter.smooth_pk_interpolator()
            self.pknow_dd_fid = self.pknow_dd_interpolator_fid(self.k)

    def calculate(self):
        for name in ['sigma8', 'fsigma8', 'f', 'pk_dd_interpolator', 'pk_dd']:
            setattr(self, name, getattr(self, name + '_fid'))
        if self.with_now:
            for name in ['pknow_dd_interpolator', 'pknow_dd']:
                setattr(self, name, getattr(self, name + '_fid'))
        if self.only_now:
            for name in ['dd_interpolator', 'dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))

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
        super(FixedPowerSpectrumTemplate, self).initialize(*args, apmode='qparqper', **kwargs)
        self.apeffect.params = dict(qpar=dict(value=1.), qper=dict(value=1.))
        self.apeffect()  # qpar, qper = 1. as a default
        self.runtime_info.requires = []  # remove APEffect dependence


class DirectPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def initialize(self, *args, cosmo=None, **kwargs):
        super(DirectPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'k': self.k, 'of': [('delta_cb', 'delta_cb')]}}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            self.cosmo.params = self.params.copy()
        self.params.clear()
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, cosmo=self.cosmo, mode='distances').runtime_info.initialize()
        if external_cosmo(self.cosmo):
            self.cosmo_requires.update(self.apeffect.cosmo_requires)  # just background

    def calculate(self):
        BasePowerSpectrumExtractor.calculate(self)
        self.pk_dd = self.pk_dd_interpolator(self.k)
        if self.with_now:
            self.pknow_dd = self.pknow_dd_interpolator(self.k)
        if self.only_now:
            for name in ['dd_interpolator', 'dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))


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
                delattr(self, name)

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
            delattr(self, name)
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
            for name in ['sigmar', 'fsigmar', 'f']:
                setattr(self, name + '_fid', getattr(self, name))
                delattr(self, name)

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

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    with_now : str, default=None
        If not ``None``, compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').
    """
    def initialize(self, *args, r=8., **kwargs):
        self.r = float(r)
        super(StandardPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.apeffect.qiso = 1.
        StandardPowerSpectrumExtractor.calculate(self)
        for name in ['sigmar', 'fsigmar', 'f']:
            setattr(self, name + '_fid', getattr(self, name))
            delattr(self, name)

    def calculate(self, df=1.):
        super(StandardPowerSpectrumTemplate, self).calculate()
        self.f = self.f_fid * df

    def get(self):
        self.fsigmar = self.f * self.sigmar_fid
        return self


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
        cosmo = self.cosmo
        self.cosmo = self.fiducial
        self.calculate()
        self.cosmo = cosmo
        for name in ['Ap', 'f_sqrt_Ap', 'n', 'm']:
            setattr(self, name + '_fid', getattr(self, name))
            delattr(self, name)

    def calculate(self):
        super(ShapeFitPowerSpectrumExtractor, self).calculate()
        s = self.cosmo.rs_drag / self.fiducial.rs_drag
        kp = self.kp / s
        self.Ap = 1. / s**3 * self.pknow_dd_interpolator(kp)
        self.f_sqrt_Ap = self.f * self.Ap**0.5
        self.n = self.cosmo.n_s
        dk = 1e-2
        k = kp * np.array([1. - dk, 1. + dk])
        # No need to include 1/s^3 factors here, as we care about the slope
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

    with_now : str, default=None
        If not ``None``, compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    def initialize(self, *args, kp=0.03, a=0.6, with_now='peakaverage', **kwargs):
        self.a = float(a)
        self.kp = float(kp)
        self.n_varied = self.params['dn'].varied
        super(ShapeFitPowerSpectrumTemplate, self).initialize(*args, with_now=with_now, **kwargs)
        ShapeFitPowerSpectrumExtractor.calculate(self)
        for name in ['f', 'n', 'm', 'Ap']:
            setattr(self, name + '_fid', getattr(self, name))
            delattr(self, name)

    def calculate(self, df=1., dm=0., dn=0.):
        super(ShapeFitPowerSpectrumTemplate, self).calculate()
        factor = np.exp(dm / self.a * np.tanh(self.a * np.log(self.k / self.kp)) + dn * np.log(self.k / self.kp))
        self.pk_dd = self.pk_dd_fid * factor
        if self.with_now:
            self.pknow_dd = self.pknow_dd_fid * factor
        if self.only_now:
            self.pk_dd = self.pknow_dd
        self.n = self.n_fid + dn
        self.m = self.m_fid + dm
        self.f = self.f_fid * df
        self.f_sqrt_Ap = self.f * self.Ap_fid**0.5

    def get(self):
        return self


class BandVelocityPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    r"""
    Extract band power parameters.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    kp : array
        Pivot :math:`k` where to evaluate the velocity divergence power spectrum :math:`P_{\theta \theta}`.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    """
    def initialize(self, *args, kp=None, **kwargs):
        super(BandVelocityPowerSpectrumExtractor, self).initialize(*args, **kwargs)
        self.kp = kp
        if kp is None:
            raise ValueError('Please provide kp')
        else:
            self.kp = np.asarray(kp)
        if self.fiducial is not None:
            cosmo = self.cosmo
            self.cosmo = self.fiducial
            self.calculate()
            self.cosmo = cosmo
            for name in ['f', 'pk_tt', 'pk_tt_interpolator']:
                setattr(self, name + '_fid', getattr(self, name))
                delattr(self, name)

    def calculate(self):
        fo = self.cosmo.get_fourier()
        self.f = fo.sigma8_z(self.z, of='theta_cb') / fo.sigma8_z(self.z, of='delta_cb')
        self.pk_tt_interpolator = fo.pk_interpolator(of='theta_cb').to_1d(z=self.z)
        self.pk_tt = self.pk_tt_interpolator(self.kp)

    def get(self):
        return self


class BandVelocityPowerSpectrumTemplate(BasePowerSpectrumTemplate, BandVelocityPowerSpectrumExtractor):
    r"""
    Band velocity power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate linear power spectrum.

    z : float, default=1.
        Effective redshift.

    kp : array
        Pivot :math:`k` where to change the value of the velocity divergence power spectrum :math:`P_{\theta \theta}`.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    with_now : str, default=None
        If not ``None``, compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').
    """
    _base_param_name = 'ptt'

    def initialize(self, *args, kp=None, **kwargs):
        super(BandVelocityPowerSpectrumTemplate, self).initialize(*args, apmode='qap', **kwargs)
        import re
        params = self.params.select(basename=re.compile(r'{}(-?\d+)'.format(self._base_param_name)))
        nkp = len(params)
        if kp is None:
            if not nkp:
                raise ValueError('No parameter {}* found'.format(self._base_param_name))
            step = (self.k[-1] - self.k[0]) / nkp
            self.kp = (self.k[0] + step / 2., self.k[-1] - step / 2.)
        else:
            self.kp = np.array(kp)
        if nkp:
            if len(self.kp) == 2:
                self.kp = np.linspace(*self.kp, num=nkp)
            self.kp = np.array(self.kp)
            if self.kp.size != nkp:
                raise ValueError('{:d} (!= {:d} parameters {}*) points have been provided'.format(self.kp.size, nkp, self._base_param_name))
        for ikp, kp in enumerate(self.kp):
            basename = '{}{:d}'.format(self._base_param_name, ikp)
            if basename not in self.params:
                value = 1.
                self.params[basename] = dict(value=value, prior={'limits': [0, 3]}, ref={'dist': 'norm', 'loc': value, 'scale': 0.01}, delta=0.005)
            self.params[basename].update(latex=r'(P / P^{{\mathrm{{fid}}}})_{{\{0}\{0}}}(k={1:.3f})'.format('theta', kp))

        if self.kp[0] < self.k[0]:
            raise ValueError('Theory k starts at {0:.2e} but first point is {1:.2e} < {0:.2e}'.format(self.k[0], self.kp[0]))
        if self.kp[-1] > self.k[-1]:
            raise ValueError('Theory k ends at {0:.2e} but last point is {1:.2e} > {0:.2e}'.format(self.k[-1], self.kp[-1]))
        ekp = np.concatenate([[self.k[0]], self.kp, [self.k[-1]]], axis=0)
        self.templates = []
        for ip, kp in enumerate(self.kp):
            diff = self.k - kp
            mask_neg = diff < 0
            diff[mask_neg] /= (ekp[ip] - kp)
            diff[~mask_neg] /= (ekp[ip + 2] - kp)
            self.templates.append(np.maximum(1. - diff, 0.))
        self.templates = np.array(self.templates)
        fo = self.fiducial.get_fourier()
        self.sigma8_fid = fo.sigma8_z(self.z, of='delta_cb')
        self.fsigma8_fid = fo.sigma8_z(self.z, of='theta_cb')
        self.f_fid = self.fsigma8_fid / self.sigma8_fid
        self.pk_tt_interpolator_fid = fo.pk_interpolator(of='theta_cb').to_1d(z=self.z)
        self.pk_tt_fid = self.pk_tt_interpolator_fid(self.k)
        self.pk_dd_fid = self.pk_tt_fid / self.f_fid**2
        if self.with_now:
            from cosmoprimo import PowerSpectrumBAOFilter
            self.filter = PowerSpectrumBAOFilter(self.pk_tt_interpolator_fid, engine=self.with_now, cosmo=self.cosmo, cosmo_fid=self.fiducial)
            self.pknow_tt_fid = self.filter.smooth_pk_interpolator()(self.k)
            self.pknow_dd_fid = self.pknow_tt_fid / self.f_fid**2

    def calculate(self, df=1., **params):
        self.f = self.f_fid * df
        rptt = jnp.array([params['{}{:d}'.format(self._base_param_name, ii)] - 1. for ii in range(len(self.templates))])
        factor = (1. + jnp.dot(rptt, self.templates)) / self.f**2
        self.pk_dd = self.pk_tt_fid * factor
        if self.with_now:
            self.pknow_dd = self.pknow_tt_fid * factor
        if self.only_now:
            for name in ['dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))

    def get(self):
        return self