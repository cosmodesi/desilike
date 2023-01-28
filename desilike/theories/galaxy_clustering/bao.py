"""Warning: not tested!"""

import re

import numpy as np
from scipy import special, integrate

from desilike.base import BaseCalculator
from desilike.theories.primordial_cosmology import get_cosmo, external_cosmo, Cosmoprimo
from desilike.jax import numpy as jnp
from .power_template import BAOPowerSpectrumTemplate
from .base import (BaseTheoryPowerSpectrumMultipoles, BaseTrapzTheoryPowerSpectrumMultipoles,
                   BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles)


class BaseBAOWigglesPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    """Base class for theory BAO power spectrum multipoles, without broadband terms."""

    def initialize(self, *args, template=None, mode='', wiggle=True, smoothing_radius=15., ells=(0, 2), **kwargs):
        super(BaseBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, ells=ells, **kwargs)
        self.wiggle = bool(wiggle)
        self.mode = str(mode)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('Reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template


class DampedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTrapzTheoryPowerSpectrumMultipoles):
    """
    Theory BAO power spectrum multipoles, without broadband terms,
    used in the BOSS DR12 BAO analysis by Beutler et al. 2017.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """
    def initialize(self, *args, mu=200, **kwargs):
        super(DampedBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def calculate(self, b1=1., sigmas=0., sigmapar=9., sigmaper=6., **kwargs):
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.pknow_dd_interpolator(kap)
        pk = self.template.pk_dd_interpolator(kap) if self.wiggle else pknow
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        damped_wiggles = (pk - pknow) * np.exp(-sigmanl2 / 2.)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        pkmu = jac * fog * (b1 + f * muap**2 * (1 - sk))**2 * (pknow + damped_wiggles)
        self.power = self.to_poles(pkmu)


class SimpleBAOWigglesPowerSpectrumMultipoles(DampedBAOWigglesPowerSpectrumMultipoles):
    r"""
    As :class:`DampedBAOWigglesPowerSpectrumMultipoles`, but moving only BAO wiggles (and not damping or RSD terms)
    with scaling parameters.
    """
    def calculate(self, b1=1., sigmas=0., sigmapar=9., sigmaper=6., **kwargs):
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        wiggles = self.template.pk_dd_interpolator(kap) / self.template.pknow_dd_interpolator(kap) if self.wiggle else 1.
        pknow = self.template.pknow_dd_interpolator(self.k)[:, None]
        sigmanl2 = self.k[:, None]**2 * (sigmapar**2 * self.mu**2 + sigmaper**2 * (1. - self.mu**2))
        damping = np.exp(-sigmanl2 / 2.)
        fog = 1. / (1. + (sigmas * self.k * self.mu[:, None])**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (self.k * self.smoothing_radius)**2)
        pkmu = fog * (b1 + f * self.mu**2 * (1 - sk))**2 * damping * wiggles * pknow
        self.power = self.to_poles(pkmu)


class ResummedPowerSpectrumWiggles(BaseCalculator):
    r"""
    Resummed BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    def initialize(self, z=1., cosmo=None, fiducial='DESI', mode='', with_now='peakaverage', smoothing_radius=15.):
        self.z = float(z)
        self.mode = str(mode)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        self.with_now = str(with_now)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo = cosmo
        if cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'pk_interpolator': {'z': self.z, 'of': [('delta_cb', 'delta_cb')]}}, 'thermodynamics': {'rs_drag': None}}

    def calculate(self):
        fo = self.cosmo.get_fourier()
        self.pk_dd_interpolator = fo.pk_interpolator(of='delta_cb').to_1d(z=self.z)
        if getattr(self, 'filter', None) is None:
            from cosmoprimo import PowerSpectrumBAOFilter
            self.filter = PowerSpectrumBAOFilter(self.pk_dd_interpolator, engine=self.with_now, cosmo=self.cosmo, cosmo_fid=self.fiducial)
        else:
            self.filter(self.pk_dd_interpolator, cosmo=self.cosmo)
        self.pknow_dd_interpolator = self.filter.smooth_pk_interpolator()
        k = self.pknow_dd_interpolator.k
        pklin = self.pknow_dd_interpolator.pk
        q = self.cosmo.rs_drag
        j0 = special.jn(0, q * k)
        sk = 0.
        if self.mode: sk = np.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        self.sigma_dd = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * (1. - sk)**2 * pklin, k)
        if self.mode:
            self.sigma_ss = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * sk**2 * pklin, k)
            if self.mode == 'recsym':
                self.sigma_ds = 1. / (3. * np.pi**2) * integrate.simps((1. / 2. * ((1. - sk)**2 + sk**2) + j0 * sk * (1. - sk)) * pklin, k)
            else:
                self.sigma_ds_dd = 1. / (6. * np.pi**2) * integrate.simps((1. - sk)**2 * pklin, k)
                self.sigma_ds_ds = - 1. / (6. * np.pi**2) * integrate.simps(j0 * sk * (1. - sk) * pklin, k)
                self.sigma_ds_ss = 1. / (6. * np.pi**2) * integrate.simps(sk**2 * pklin, k)

    def wiggles(self, k, mu, b1=1., f=0.):
        wiggles = self.pk_dd_interpolator(k) - self.pknow_dd_interpolator(k)
        b1 = b1 - 1.  # lagrangian b1
        sk = 0.
        if self.mode: sk = np.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        ksq = (1 + f * (f + 2) * mu**2) * k**2
        damping_dd = np.exp(-1. / 2. * ksq * self.sigma_dd)
        resummed_wiggles = damping_dd * ((1 + f * mu**2) * (1 - sk) + b1)**2
        if self.mode == 'recsym':
            damping_ds = np.exp(-1. / 2. * ksq * self.sigma_ds)
            resummed_wiggles -= 2. * damping_ds * ((1 + f * mu**2) * (1 - sk) + b1) * (1 + f * mu**2) * sk
            damping_ss = np.exp(-1. / 2. * ksq * self.sigma_ss)
            resummed_wiggles += damping_ss * (1 + f * mu**2)**2 * sk**2
        if self.mode == 'reciso':
            damping_ds = np.exp(-1. / 2. * (ksq * self.sigma_ds_dd + k**2 * (self.sigma_ds_ss - 2. * (1 + f * mu**2) * self.sigma_ds_dd)))
            resummed_wiggles -= 2. * damping_ds * ((1 + f * mu**2) * (1 - sk) + b1) * sk
            damping_ss = np.exp(-1. / 2. * k**2 * self.sigma_ss)  # f = 0.
            resummed_wiggles += damping_ss * sk**2
        return resummed_wiggles * wiggles


class ResummedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTrapzTheoryPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, without broadband terms,
    with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    def initialize(self, *args, mu=200, **kwargs):
        super(ResummedBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)
        self.template.init.update(with_now=False)
        self.template.runtime_info.initialize()
        if self.wiggle:
            self.wiggles = ResummedPowerSpectrumWiggles(mode=self.mode, z=self.template.z,
                                                        cosmo=self.template.cosmo, fiducial=self.template.fiducial,
                                                        smoothing_radius=self.smoothing_radius)

    def calculate(self, b1=1., sigmas=0., **kwargs):
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.wiggles.pknow_dd_interpolator(kap)
        wiggles = 0. if self.wiggle else self.wiggles.wiggles(kap, muap, b1=b1, **kwargs)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        pkmu = jac * fog * (wiggles + (b1 + f * muap**2 * (1 - sk))**2 * pknow)
        self.power = self.to_poles(pkmu)


class BaseBAOWigglesTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):
    """
    Base class for theory BAO power spectrum multipoles, with broadband terms.
    
    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
        
    ells : tuple, default=(0, 2)
        Multipoles to compute.
        
    mu : int, default=200
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).
        
    mode : str, default=''
        Reconstruction mode:
        
        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)
        
    wiggle : bool, default=True
        If ``False``, switch off BAO wiggles: model is computed with smooth power spectrum.
        
    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.
        
    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.
    """

    config_fn = 'bao.yaml'

    def initialize(self, k=None, ells=(0, 2), **kwargs):
        super(BaseBAOWigglesTracerPowerSpectrumMultipoles, self).initialize(k=k, ells=ells)
        self.pt = globals()[self.__class__.__name__.replace('Tracer', '')]()
        self.pt.init.update(k=self.k, ells=self.ells, **kwargs)
        self.kp = 0.1  # pivot to noramlize broadband terms
        self.set_params()

    def set_params(self):
        self_params = self.params.select(basename='al*_*')
        pt_params = self.params.copy()
        for param in pt_params.names():
            if param in self_params: del pt_params[param]
        self.pt.params = pt_params
        broadband_coeffs = {}
        for ell in self.ells:
            broadband_coeffs[ell] = {}
        for param in self_params.params():
            name = param.basename
            match = re.match('al(.*)_(.*)', name)
            if match:
                ell = int(match.group(1))
                pow = int(match.group(2))
                if ell in self.ells:
                    broadband_coeffs[ell][name] = (self.k / self.kp)**pow
                else:
                    del self_params[param]
            else:
                raise ValueError('Unrecognized parameter {}'.format(param))
        self.broadband_matrix = []
        self.broadband_params = [name for ell in self.ells for name in broadband_coeffs[ell]]
        for ill, ell in enumerate(self.ells):
            row = [np.zeros_like(self.k) for i in range(len(self.broadband_params))]
            for name, k_i in broadband_coeffs[ell].items():
                row[self.broadband_params.index(name)] = k_i
            self.broadband_matrix.append(np.column_stack(row))
        self.broadband_matrix = jnp.array(self.broadband_matrix)
        self.params = self_params

    def calculate(self, **params):
        values = jnp.array([params.get(name, 0.) for name in self.broadband_params])
        self.power = self.pt.power + self.broadband_matrix.dot(values)

    @property
    def wiggle(self):
        return self.pt.wiggle

    @wiggle.setter
    def wiggle(self, wiggle):
        self.pt.wiggle = wiggle

    def get(self):
        return self.power


class DampedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):

    pass


class SimpleBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):

    pass


class ResummedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):

    pass


class BaseBAOWigglesCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):
    """
    Base class that implements theory BAO correlation function multipoles, without broadband terms,
    as Hankel transforms of the theory power spectrum multipoles.
    """
    def initialize(self, s=None, ells=(0, 2), **kwargs):
        power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')](**kwargs)
        super(BaseBAOWigglesCorrelationFunctionMultipoles, self).initialize(s=s, ells=ells, power=power)


class DampedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class ResummedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class BaseBAOWigglesTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    """Base class that implements theory BAO correlation function multipoles, with broadband terms."""
    config_fn = 'bao.yaml'

    def initialize(self, s=None, ells=(0, 2), **kwargs):
        super(BaseBAOWigglesTracerCorrelationFunctionMultipoles, self).initialize(s=s, ells=ells)
        self.pt = globals()[self.__class__.__name__.replace('Tracer', '')]()
        self.pt.init.update(s=self.s, ells=self.ells, **kwargs)
        self.sp = 60.  # pivot to noramlize broadband terms
        self.set_params()

    def set_params(self):
        self.k, self.kp = self.s, self.sp
        BaseBAOWigglesTracerPowerSpectrumMultipoles.set_params(self)
        del self.k, self.kp

    def calculate(self, **params):
        values = jnp.array([params.get(name, 0.) for name in self.broadband_params])
        self.corr = self.pt.corr + self.broadband_matrix.dot(values)

    @property
    def wiggle(self):
        return self.pt.wiggle

    @wiggle.setter
    def wiggle(self, wiggle):
        self.pt.wiggle = wiggle

    def get(self):
        return self.corr


class DampedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):

    pass


class SimpleBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):

    pass


class ResummedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):

    pass
