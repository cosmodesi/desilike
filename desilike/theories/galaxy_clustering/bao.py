"""Warning: not tested!"""

import re

import numpy as np
from scipy import special, integrate

from desilike.base import BaseCalculator
from desilike.theories.primordial_cosmology import get_cosmo, external_cosmo, Cosmoprimo
from desilike import plotting
from desilike.jax import numpy as jnp
from .power_template import BAOPowerSpectrumTemplate
from .base import (BaseTheoryPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges,
                   BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles)


class BaseBAOWigglesPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    """Base class for theory BAO power spectrum multipoles, without broadband terms."""

    def initialize(self, *args, template=None, mode='', smoothing_radius=15., ells=(0, 2), **kwargs):
        super(BaseBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, ells=ells, **kwargs)
        self.mode = str(mode)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('Reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z


class DampedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    """
    Theory BAO power spectrum multipoles, without broadband terms,
    used in the BOSS DR12 BAO analysis by Beutler et al. 2017.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """
    def initialize(self, *args, mu=40, method='leggauss', model='howlett2023', **kwargs):
        super(DampedBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.model = str(model)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)

    def calculate(self, b1=1., sigmas=0., sigmapar=9., sigmaper=6.):
        super(DampedBAOWigglesPowerSpectrumMultipoles, self).calculate()
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.pknow_dd_interpolator(kap)
        pk = self.template.pk_dd_interpolator(kap)
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        pkw = pk - pknow
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        kaiser = (b1 + f * muap**2 * (1 - sk))**2
        if self.model == 'beutler2018':
            pkmu = jac * fog * kaiser * (pknow + np.exp(-sigmanl2 / 2.) * pkw)
        else:
            pkmu = jac * (fog * kaiser * pknow + kaiser * np.exp(-sigmanl2 / 2.) * pkw)
        self.power = self.to_poles(pkmu)


class SimpleBAOWigglesPowerSpectrumMultipoles(DampedBAOWigglesPowerSpectrumMultipoles):
    r"""
    As :class:`DampedBAOWigglesPowerSpectrumMultipoles`, but moving only BAO wiggles (and not damping or RSD terms)
    with scaling parameters.
    """
    def calculate(self, b1=1., sigmas=0., sigmapar=9., sigmaper=6.):
        super(SimpleBAOWigglesPowerSpectrumMultipoles, self).calculate()
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.pknow_dd_interpolator(self.k)[:, None]
        sigmanl2 = self.k[:, None]**2 * (sigmapar**2 * self.mu**2 + sigmaper**2 * (1. - self.mu**2))
        pkw = self.template.pk_dd_interpolator(kap) - self.template.pknow_dd_interpolator(kap)
        fog = 1. / (1. + (sigmas * self.k[:, None] * self.mu)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (self.k * self.smoothing_radius)**2)[:, None]
        kaiser = (b1 + f * self.mu**2 * (1 - sk))**2
        if self.model == 'beutler':
            pkmu = fog * kaiser * (pknow + np.exp(-sigmanl2 / 2.) * pkw)
        else:
            pkmu = (fog * kaiser * pknow + kaiser * np.exp(-sigmanl2 / 2.) * pkw)
        self.power = self.to_poles(pkmu)


class ResummedPowerSpectrumWiggles(BaseCalculator):
    r"""
    Resummed BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    def initialize(self, template=None, mode='', smoothing_radius=15.):
        self.mode = str(mode)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template
        self.template.runtime_info.initialize()
        if external_cosmo(self.template.cosmo):
            self.cosmo_requires = {'thermodynamics': {'rs_drag': None}}
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z
        k = self.template.pknow_dd_interpolator.k
        pklin = self.template.pknow_dd_interpolator.pk
        q = self.template.cosmo.rs_drag
        j0 = special.jn(0, q * k)
        sk = 0.
        if self.mode: sk = np.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        # https://www.overleaf.com/project/633e1b59130591a7bf55a9cd eq. 23 - 24
        skc = 1. - sk
        self.sigma_dd = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * skc**2 * pklin, k)
        #print(k.shape, self.sigma_dd.shape)
        if self.mode:
            self.sigma_ss = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * sk**2 * pklin, k)
            if self.mode == 'recsym':
                self.sigma_ds = 1. / (3. * np.pi**2) * integrate.simps((1. / 2. * (skc**2 + sk**2) + j0 * sk * skc) * pklin, k)
            else:
                self.sigma_ds_dd = 1. / (6. * np.pi**2) * integrate.simps(skc**2 * pklin, k)
                self.sigma_ds_ds = - 1. / (6. * np.pi**2) * integrate.simps(j0 * sk * skc * pklin, k)
                self.sigma_ds_ss = 1. / (6. * np.pi**2) * integrate.simps(sk**2 * pklin, k)

    def wiggles(self, k, mu, b1=1., f=0., d=1., sigmas=0.):
        # b1 Eulerian bias, d scaling the growth factor, sigmas FoG
        wiggles = self.template.pk_dd_interpolator(k) - self.template.pknow_dd_interpolator(k)
        sk = 0.
        if self.mode: sk = np.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        skc = 1. - sk
        ksq = (1 + f * (f + 2) * mu**2) * k**2
        dsq = d**2
        damping_dd = np.exp(-1. / 2. * ksq * dsq * self.sigma_dd)
        resummed_wiggles = damping_dd * (b1 + f * mu**2 * skc - sk)**2
        if self.mode == 'recsym':
            damping_ds = np.exp(-1. / 2. * (ksq * dsq * self.sigma_ds + (k * mu * sigmas)**2))
            resummed_wiggles -= 2. * damping_ds * (b1 + f * mu**2 * skc - sk) * (1 + f * mu**2) * sk
            damping_ss = np.exp(-1. / 2. * ksq * dsq * self.sigma_ss)
            resummed_wiggles += damping_ss * (1 + f * mu**2)**2 * sk**2
        if self.mode == 'reciso':
            damping_ds = np.exp(-1. / 2. * (ksq * dsq * self.sigma_ds_dd + k**2 * dsq * (self.sigma_ds_ss - 2. * (1 + f * mu**2) * self.sigma_ds_dd) + (k * mu * sigmas)**2))
            resummed_wiggles -= 2. * damping_ds * (b1 + f * mu**2 * skc - sk) * sk
            damping_ss = np.exp(-1. / 2. * k**2 * dsq * self.sigma_ss)  # f = 0.
            resummed_wiggles += damping_ss * sk**2
        return resummed_wiggles * wiggles


class ResummedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Theory BAO power spectrum multipoles, without broadband terms, with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    def initialize(self, *args, mu=20, method='leggauss', **kwargs):
        super(ResummedBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)
        self.wiggles = ResummedPowerSpectrumWiggles(mode=self.mode, template=self.template,
                                                    smoothing_radius=self.smoothing_radius)

    def calculate(self, b1=1., sigmas=0., d=1., **kwargs):
        super(ResummedBAOWigglesPowerSpectrumMultipoles, self).calculate()
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.pknow_dd_interpolator(kap)
        damped_wiggles = 0. if self.template.only_now else self.wiggles.wiggles(kap, muap, b1=b1, f=f, d=d, **kwargs)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        kaiser = (b1 + f * muap**2 * (1 - sk))**2
        pkmu = jac * (fog * kaiser * pknow + damped_wiggles)
        self.power = self.to_poles(pkmu)


class BaseBAOWigglesTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):
    r"""
    Base class for theory BAO power spectrum multipoles, with broadband terms.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

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
        self.kp = 0.1  # pivot to normalize broadband terms
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.set_params()

    def set_params(self):

        def get_params_matrix(base):
            coeffs = {ell: {} for ell in self.ells}
            for param in self.params.select(basename=base + '*_*'):
                name = param.basename
                ell = None
                if name == base + '0':
                    ell, pow = 0, 0
                else:
                    match = re.match(base + '(.*)_(.*)', name)
                    if match:
                        ell, pow = int(match.group(1)), int(match.group(2))
                if ell is not None:
                    if ell in self.ells:
                        coeffs[ell][name] = (self.k / self.kp)**pow
                    else:
                        del self.params[param]
            params = [name for ell in self.ells for name in coeffs[ell]]
            matrix = []
            for ell in self.ells:
                row = [np.zeros_like(self.k) for i in range(len(params))]
                for name, k_i in coeffs[ell].items():
                    row[params.index(name)][:] = k_i
                matrix.append(np.column_stack(row))
            matrix = jnp.array(matrix)
            return params, matrix

        self.broadband_params, self.broadband_matrix = get_params_matrix('al')
        pt_params = self.params.copy()
        for param in pt_params.basenames():
            if param in self.broadband_params: del pt_params[param]
        self.pt.params = pt_params
        self.params = self.params.select(basename=self.broadband_params)

    def calculate(self, **params):
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        values = jnp.array([params.get(name, 0.) for name in self.broadband_params])
        self.power = self.pt.power + self.broadband_matrix.dot(values)

    @property
    def template(self):
        return self.pt.template

    def get(self):
        return self.power

    @plotting.plotter
    def plot(self):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return ax


class DampedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms, used in the BOSS DR12 BAO analysis by Beutler et al. 2017.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.


    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """


class SimpleBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    As :class:`DampedBAOWigglesTracerPowerSpectrumMultipoles`, but moving only BAO wiggles (and not damping or RSD terms)
    with scaling parameters; essentially used for Fisher forecasts.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    """


class ResummedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms, with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.


    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """


class BaseBAOWigglesCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):
    """
    Base class that implements theory BAO correlation function multipoles, without broadband terms,
    as Hankel transforms of the theory power spectrum multipoles.
    """
    def initialize(self, s=None, ells=(0, 2), **kwargs):
        power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')](**kwargs)
        super(BaseBAOWigglesCorrelationFunctionMultipoles, self).initialize(s=s, ells=ells, power=power)
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))

    def calculate(self):
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        super(BaseBAOWigglesCorrelationFunctionMultipoles, self).calculate()

    @property
    def template(self):
        return self.power.template

    def get(self):
        return self.corr


class DampedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class SimpleBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

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
        self.sp = 60.  # pivot to normalize broadband terms
        for name in ['z', 's', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.set_params()

    def set_params(self):
        self.k, self.kp = self.s, self.sp
        BaseBAOWigglesTracerPowerSpectrumMultipoles.set_params(self)
        del self.k, self.kp

    def calculate(self, **params):
        for name in ['z', 's', 'ells']:
            setattr(self, name, getattr(self.pt, name))
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

    @plotting.plotter
    def plot(self):
        """
        Plot correlation function multipoles.

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for ill, ell in enumerate(self.ells):
            ax.plot(self.s, self.s**2 * self.corr[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return ax


class DampedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    Theory BAO correlation function multipoles, with broadband terms.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.


    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """


class SimpleBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    As :class:`DampedBAOWigglesTracerCorrelationFunctionMultipoles`, but moving only BAO wiggles (and not damping or RSD terms)
    with scaling parameters; essentially used for Fisher forecasts.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.


    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """


class ResummedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    Theory BAO correlation function multipoles, with broadband terms, with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.


    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """


class FlexibleBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms,
    both multiplying anf adding to the wiggles; no damping parameter (BAO damping or Finger-of-God).
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    broadband_kernel : str, default='tsc'
        Additive and multiplicative broadband kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to the standard :math:`k^{n}` broadband terms.

    kp : float, array, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period; typically :math:`2 \pi / r_{d}` (defaults to :math:`2 \pi / 100`).

    """
    config_fn = 'bao.yaml'
    default_kp = 2. * np.pi / 100.  # BAO scale

    def initialize(self, *args, mu=40, method='leggauss', kp=None, broadband_kernel='tsc', **kwargs):
        super(FlexibleBAOWigglesTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)
        #self.template.runtime_info.initialize()
        self.set_broadband_kernel(broadband_kernel=broadband_kernel, kp=kp)
        self.set_params()

    def set_broadband_kernel(self, broadband_kernel, kp=None):
        if hasattr(broadband_kernel, 'items'):
            self.broadband_kernel = dict(broadband_kernel)
        else:
            self.broadband_kernel = {'add': str(broadband_kernel), 'mult': str(broadband_kernel)}
        if hasattr(kp, 'items'):
            self.kp = dict(kp)
        else:
            self.kp = {'add': kp, 'mult': kp}

    def set_params(self):

        def get_orders(base):
            orders = {ell: {} for ell in self.ells}
            for param in self.params.select(basename=base + '*_*'):
                name = param.basename
                ell = None
                if name == base + '0':
                    ell, pow = 0, 0
                else:
                    match = re.match(base + '(.*)_(.*)', name)
                    if match:
                        ell, pow = int(match.group(1)), int(match.group(2))
                if ell is not None:
                    if ell in self.ells:
                        orders[ell][name] = pow
                    else:
                        del self.params[param]
            return orders

        def kernel_support(kernel):
            return {'ngp': 0.5, 'cic': 1., 'tsc': 1.5, 'pcs': 2.}[kernel]

        def kernel_func(k, kp, dkp, kernel='tsc'):
            toret = np.zeros_like(k)
            for kkp in kp:
                diff = k - kkp
                adiff = np.abs(diff)
                adiff[diff < 0.] /= dkp[0]
                adiff[diff >= 0.] /= dkp[1]
                if kernel == 'ngp':
                    mask = adiff < 0.5
                    np.add.at(toret, mask, 1.)
                elif kernel == 'cic':
                    mask = adiff < 1.
                    np.add.at(toret, mask, 1. - adiff[mask])
                elif kernel == 'tsc':
                    mask = adiff < 0.5
                    np.add.at(toret, mask, 3. / 4. - adiff[mask]**2)
                    mask = (adiff >= 0.5) & (adiff < 1.5)
                    np.add.at(toret, mask, 1. / 2. * (3. / 2. - adiff[mask])**2)
                elif kernel == 'pcs':
                    mask = adiff < 1.
                    np.add.at(toret, mask, 1. / 6. * (4. - 6. * adiff[mask]**2 + 3. * adiff[mask]**3))
                    mask = (adiff >= 1.) & (adiff < 2.)
                    np.add.at(toret, mask, 1. / 6. * (2. - adiff[mask])**3)
            return toret

        self.broadband_orders, self.broadband_matrix = {}, {}
        for base in ['add', 'mult']:
            base_param_name = base[0] + 'l'
            self.broadband_orders[base] = get_orders(base_param_name)
            self.broadband_matrix[base] = {}
            kernel = self.broadband_kernel[base]
            if kernel == 'power':
                for ell in self.ells:
                    row = jnp.array([(self.k / (self.kp[base] if self.kp[base] is not None else 0.1))**pow for pow in self.broadband_orders[base][ell].values()])
                    self.broadband_matrix[base][ell] = row
            elif kernel in ['ngp', 'cic', 'tsc', 'pcs']:
                for ell in self.ells:
                    kp = self.kp[base]
                    ids = self.broadband_orders[base][ell].values()
                    nkp = len(ids)
                    if kp is None:
                        if nkp:
                            kp = (self.k[-1] - self.k[0]) / nkp
                        elif base == 'mult' and self.template.only_now:  # no terms
                            self.broadband_matrix[base][ell] = jnp.zeros((0, len(self.k)), dtype='f8')
                            continue
                        else:
                            kp = self.default_kp
                    if np.ndim(kp) == 0:
                        kp = np.arange(self.k[0], self.k[-1] + kp * (1. - 1e-9), kp)
                    elif isinstance(kp, tuple):
                        kpmin, kpmax, dkp = kp
                        if kpmin is None: kpmin = self.k[0]
                        if kpmax is None: kpmax = self.k[-1]
                        if dkp is None: dkp = self.default_kp
                        kp = np.arange(kpmin, kpmax + dkp * (1. - 1e-9), dkp)
                    kp = np.array(kp, dtype='f8')
                    nkp = len(kp)
                    if not ids:
                        for ikp in range(nkp):
                            basename = '{}{:d}_{:d}'.format(base_param_name, ell, ikp)
                            self.params[basename] = dict(value=0., prior=None, ref={'dist': 'norm', 'loc': 0., 'scale': 0.01}, delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, ikp))
                            self.broadband_orders[base][ell][basename] = ikp
                    if set(self.broadband_orders[base][ell].values()) != set(range(nkp)):
                        raise ValueError('Found parameters {}, but expected all parameters 0 to {:d}'.format(list(self.broadband_orders[base][ell].keys()), nkp))
                    support = kernel_support(kernel)
                    dkp = np.diff(kp)
                    dkp_low, dkp_high = np.insert(dkp, 0, dkp[0]), np.insert(dkp, -1, dkp[-1])
                    kmin, kmax = self.k[0] - dkp[0] * support, self.k[-1] + dkp[-1] * support
                    row = []
                    row.append(kernel_func(self.k, np.arange(kp[0], kmin - dkp_low[0], -dkp_low[0]), (dkp_low[0], dkp_high[0]), kernel=kernel))
                    for kkp, ddkp_low, ddkp_high in list(zip(kp, dkp_low, dkp_high))[1:-1]:
                        row.append(kernel_func(self.k, [kkp], (ddkp_low, ddkp_high), kernel=kernel))
                    row.append(kernel_func(self.k, np.arange(kp[-1], kmax + dkp_high[0], dkp_high[-1]), (dkp_low[-1], dkp_high[-1]), kernel=kernel))
                    row = jnp.array([row[idx] for idx in ids])
                    self.broadband_matrix[base][ell] = row
            else:
                raise ValueError('Unknown kernel: {}'.format(kernel))

    def calculate(self, b1=1.5, **kwargs):
        super(FlexibleBAOWigglesTracerPowerSpectrumMultipoles, self).calculate()
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.pknow_dd_interpolator(self.k)
        wiggles = self.template.pk_dd_interpolator(kap) / self.template.pknow_dd_interpolator(kap) - 1.
        damped_wiggles = 0.
        for ell in self.ells:
            mult = jnp.array([kwargs[name] for name in self.broadband_orders['mult'][ell]]).dot(self.broadband_matrix['mult'][ell])
            if ell == 0: mult += 1.
            add = jnp.array([kwargs[name] for name in self.broadband_orders['add'][ell]]).dot(self.broadband_matrix['add'][ell])
            leg = special.legendre(ell)(self.mu)
            damped_wiggles += wiggles * mult[:, None] * leg + add[:, None] * leg
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (self.k * self.smoothing_radius)**2)[:, None]
        pkmu = (b1 + f * self.mu**2 * (1 - sk))**2 * pknow[:, None] * (1. + damped_wiggles)
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power

    @plotting.plotter
    def plot(self):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return ax


class FlexibleBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    config_fn = 'bao.yaml'

    def initialize(self, *args, kp=None, broadband_kernel='tsc', **kwargs):
        super(FlexibleBAOWigglesTracerCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        FlexibleBAOWigglesTracerPowerSpectrumMultipoles.set_broadband_kernel(self, broadband_kernel=broadband_kernel, kp=kp)
        for base, kernel in self.broadband_kernel.items():
            if kernel != 'power' and (self.kp[base] is None or np.ndim(self.kp[base]) == 0):
                self.kp[base] = (self.kin[0], 0.3, self.kp[base])
        self.power.init.update(broadband_kernel=self.broadband_kernel, kp=self.kp)

    @plotting.plotter
    def plot(self):
        """
        Plot correlation function multipoles.

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for ill, ell in enumerate(self.ells):
            ax.plot(self.s, self.s**2 * self.corr[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return ax