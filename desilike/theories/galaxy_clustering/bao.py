"""
BAO power spectrum and correlation function multipoles with various models.

This module provides theory predictions for BAO power spectrum and correlation
function multipoles in galaxy clustering analyses. It supports multiple
parameterizations of BAO wiggles (damped, resummed, flexible) and broadband
terms (power-law, kernel-based), as well as reconstruction conventions.

Key Classes
-----------
Power Spectrum (without broadband)
    - BaseBAOWigglesPowerSpectrumMultipoles: Base class
    - DampedBAOWigglesPowerSpectrumMultipoles: Chen 2023 damped wiggles model
    - ResummedBAOWigglesPowerSpectrumMultipoles: Resummed BAO model
    - FlexibleBAOWigglesPowerSpectrumMultipoles: Flexible multiplicative wiggles

Power Spectrum (with broadband)
    - BaseBAOWigglesTracerPowerSpectrumMultipoles: Base class
    - DampedBAOWigglesTracerPowerSpectrumMultipoles: Damped with broadband
    - ResummedBAOWigglesTracerPowerSpectrumMultipoles: Resummed with broadband
    - FlexibleBAOWigglesTracerPowerSpectrumMultipoles: Flexible with broadband

Correlation Function (without broadband)
    - BaseBAOWigglesCorrelationFunctionMultipoles: Base class (Hankel transform)
    - DampedBAOWigglesCorrelationFunctionMultipoles: Damped model
    - ResummedBAOWigglesCorrelationFunctionMultipoles: Resummed model

Correlation Function (with broadband)
    - BaseBAOWigglesTracerCorrelationFunctionMultipoles: Base class
    - DampedBAOWigglesTracerCorrelationFunctionMultipoles: Damped with broadband
    - ResummedBAOWigglesTracerCorrelationFunctionMultipoles: Resummed with broadband
    - FlexibleBAOWigglesTracerCorrelationFunctionMultipoles: Flexible with broadband

Reconstruction Modes
    - '': No reconstruction (linear theory)
    - 'recsym': Symmetric reconstruction (both data and randoms displaced)
    - 'reciso': Isotropic reconstruction (data only displaced)


Creating a New BAO Theory Model
===============================

To add a new BAO wiggle model (e.g., a new damping prescription or resummation scheme),
follow these steps:

1. **Create a new wiggle class** inheriting from ``BaseBAOWigglesPowerSpectrumMultipoles``::

    class MyBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles):
        r'''My custom BAO wiggle model.

        Parameters
        ----------
        my_param : float, default=1.0
            Description of my custom parameter.
        '''

        def initialize(self, *args, my_param=1.0, **kwargs):
            super().initialize(*args, **kwargs)
            self.my_param = float(my_param)
            # Initialize any custom objects (e.g., ProjectToMultipoles, kernels)
            self.to_poles = ProjectToMultipoles(mu=10, ells=self.ells)
            self.mu = self.to_poles.mu

        def calculate(self, b1=1., dbeta=1., **custom_params):
            '''Calculate power spectrum multipoles.

            Parameters
            ----------
            b1 : float, default=1.
                Linear bias.
            dbeta : float, default=1.
                Relative growth rate: f = f_fid * dbeta.
            **custom_params : dict
                Any additional model-specific parameters.
            '''
            # Always call parent to update z, rs_drag_fid
            super().calculate()

            # Access template attributes
            f = dbeta * self.template.f
            k = self.k
            mu = self.mu

            # Get power spectrum (with AP distortion if needed)
            jac, kap, muap = self.template.ap_k_mu(k, mu)
            pknow = _interp(self.template, 'pknow_dd', kap)
            pk = _interp(self.template, 'pk_dd', kap)

            # Compute power in (k, mu) space
            pkmu = (b1 + f * mu**2)**2 * pknow  # Simple example

            # Project to multipoles
            self.power = self.to_poles(pkmu)

2. **Add broadband support** by creating a tracer variant::

    class MyBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
        r'''My BAO model with broadband terms.'''
        _pt_cls = MyBAOWigglesPowerSpectrumMultipoles

3. **Add correlation function support** (optional) via Hankel transform::

    class MyBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):
        _pt_cls = MyBAOWigglesPowerSpectrumMultipoles

    class MyBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
        # Inherits from BaseBAOWigglesTracerCorrelationFunctionMultipoles
        pass

Key Implementation Details
--------------------------

**Template access**: The power spectrum template is stored in ``self.template``. Key attributes:

- ``self.template.k`` : wavenumbers used for internal computations
- ``self.template.pk_dd`` : full BAO power spectrum (matter-matter)
- ``self.template.pknow_dd`` : no-wiggle (smooth) power spectrum
- ``self.template.f`` : growth rate f = d ln D / d ln a
- ``self.template.z`` : effective redshift
- ``self.template.ap_k_mu(k, mu)`` : Apply Alcock-Paczynski distortion to (k, mu)

**Multipole projection**: Use ``ProjectToMultipoles`` to convert (k, mu) -> P_ell(k)::

    from .base import ProjectToMultipoles
    self.to_poles = ProjectToMultipoles(mu=10, ells=(0, 2))
    self.power = self.to_poles(pkmu)  # pkmu shape: (n_k, n_mu)

**Reconstruction modes**: For reconstruction, modify the bias factor:

- ``mode=''`` : Standard redshift-space model, no reconstruction
- ``mode='recsym'`` : Symmetric reconstruction (both data and randoms displaced)
- ``mode='reciso'`` : Isotropic reconstruction (data only displaced)

Apply smoothing-radius-dependent suppression::

    if self.mode == 'reciso':
        sk = jnp.exp(-0.5 * (k * self.smoothing_radius)**2)
        b_eff = b1 + f * mu**2 * (1 - sk)  # Modified bias

**Finger-of-God damping**: Apply FOG in redshift space::

    sigmas = 0.  # FOG damping parameter
    fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2
    pkmu *= fog
"""

import re

import numpy as np
from scipy import special, integrate

from desilike.base import BaseCalculator
from desilike.cosmo import is_external_cosmo
from desilike import plotting
from desilike.jax import numpy as jnp
from desilike.jax import jit, interp1d
from .power_template import BAOPowerSpectrumTemplate
from .base import SpectrumToCorrelationMultipoles, ProjectToMultipoles


def _interp(template, name, k):
    """Interpolate power spectrum in log-log space."""
    return interp1d(jnp.log10(k), jnp.log10(template.k), getattr(template, name), method='cubic')
    #return getattr(template, name + '_interpolator')(k)


def _get_orders(base, params, ells):
    """
    Extract multipole orders from parameter collection.

    Parses parameter names matching pattern 'base*_*' and maps them to
    (ell, index) pairs. Removes any parameters for multipoles not in ``ells``.

    Parameters
    ----------
    base : str
        Base name for parameters (e.g., 'al', 'ml', 'bl').
    params : ParameterCollection
        Collection of parameters to parse.
    ells : tuple
        Valid multipoles to retain.

    Returns
    -------
    orders : dict
        Mapping from ell -> {param_name: index}.
    """
    orders = {ell: {} for ell in ells}
    for param in params.select(basename=base + '*_*'):
        name = param.basename
        ell = None
        if name == base + '0':
            ell, ind = 0, 0
        else:
            match = re.match(base + '(.*)_(.*)', name)
            if match:
                ell, ind = int(match.group(1)), int(match.group(2))
        if ell is not None:
            if ell in ells:
                orders[ell][name] = ind
            else:
                del params[param]
    return orders


def _kernel_func(x, kernel='tsc'):
    """
    Evaluate kernel function at distances x.
    Used in approximate broadband parameterizations.

    Parameters
    ----------
    x : ndarray
        Dimensionless distance (units of kernel period).
    kernel : str, default='tsc'
        Kernel type: 'ngp' (nearest-grid-point), 'cic' (cloud-in-cell),
        'tsc' (triangular-shaped cloud), 'pcs' (piecewise cubic spline).

    Returns
    -------
    result : ndarray
        Kernel values at x.

    Notes
    -----
    Kernels are normalized to integrate to 1 over a period.
    """
    toret = np.zeros_like(x)
    if kernel == 'ngp':
        # Nearest grid point: box from 0 to 0.5
        mask = x < 0.5
        np.add.at(toret, mask, 1.)
    elif kernel == 'cic':
        # Cloud in cell: linear from 1 to 0 over [0, 1]
        mask = x < 1.
        np.add.at(toret, mask, 1. - x[mask])
    elif kernel == 'tsc':
        # Triangular shaped cloud: parabolic over [0, 1.5]
        mask = x < 0.5
        np.add.at(toret, mask, 3. / 4. - x[mask]**2)
        mask = (x >= 0.5) & (x < 1.5)
        np.add.at(toret, mask, 1. / 2. * (3. / 2. - x[mask])**2)
    elif kernel == 'pcs':
        # Piecewise cubic spline over [0, 2]
        mask = x < 1.
        np.add.at(toret, mask, 1. / 6. * (4. - 6. * x[mask]**2 + 3. * x[mask]**3))
        mask = (x >= 1.) & (x < 2.)
        np.add.at(toret, mask, 1. / 6. * (2. - x[mask])**3)
    return toret


class BaseBAOWigglesPowerSpectrumMultipoles(BaseCalculator):
    r"""
    Base class for theory BAO power spectrum multipoles without broadband terms.

    Computes power spectrum multipoles :math:`P_\ell(k)` from a BAO template,
    accounting for Alcock-Paczynski scaling, redshift-space distortions,
    and reconstruction effects.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers [h/Mpc] where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipole orders to compute (typically monopole and quadrupole).
    mu : int, default=10
        Number of mu-bins in [0, 1] for angular projection.
    mode : str, default=''
        Reconstruction mode:
        - '': No reconstruction
        - 'recsym': Symmetric reconstruction
        - 'reciso': Isotropic reconstruction
    smoothing_radius : float, default=15.
        Smoothing radius [Mpc/h] used in reconstruction Wiener filter.
    template : BasePowerSpectrumTemplate, optional
        Power spectrum template. Defaults to BAOPowerSpectrumTemplate.

    Attributes
    ----------
    power : ndarray, shape (n_ells, n_k)
        Power spectrum multipoles.
    z : float
        Effective redshift.
    k : ndarray
        Wavenumbers where multipoles are evaluated.
    ells : tuple
        Computed multipole orders.
    rs_drag_fid : float
        Fiducial sound horizon at drag epoch.
    """
    _klim = (1e-4, 1., 2000)  # (k_min, k_max, n_k) for internal template evaluation

    def initialize(self, k=None, template=None, mode='', smoothing_radius=15., ells=(0, 2), **kwargs):
        """
        Initialize BAO power spectrum multipoles.

        Parameters
        ----------
        k : array, optional
            Theory wavenumbers. If None, defaults to [0.01, 0.2] with 101 points.
        ells : tuple, default=(0, 2)
            Multipole orders to compute.
        mode : str, default=''
            Reconstruction mode ('', 'recsym', 'reciso').
        smoothing_radius : float, default=15.
            Smoothing radius for reconstruction.
        template : BasePowerSpectrumTemplate, optional
            Power spectrum template instance.
        **kwargs : dict
            Additional arguments passed to template initialization.
        """
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        self.mode = str(mode)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('Reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(self._klim[0], self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(self._klim[1], self.k[-1] * 2, self.template.init.get('k', [0.])[0]), self._klim[2])  # margin for AP effect
        self.template.init.update(k=kin)
        self.template.init.setdefault('with_now', 'peakaverage', if_none=True)
        self.z = self.template.z
        self.rs_drag_fid = self.template.fiducial.rs_drag
        if tuple(self.ells) == (0,):  # one should be able to initialize pt without parameters  --- just to k and ells
            for param in self.init.params.select(basename=['dbeta']):
                param.update(fixed=True)

    def calculate(self):
        """
        Calculate power spectrum multipoles.

        Sets self.power with shape (n_ells, n_k). Must be overridden
        by subclasses to implement specific models.
        """
        self.z = self.template.z
        self.rs_drag_fid = self.template.fiducial.rs_drag
        # Add computation of BAO power spectrum multipoles here,
        # using self.template for the power spectrum template and self.k for the wavenumbers.

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'power', 'rs_drag_fid']:
            state[name] = getattr(self, name)
        return state

    def get(self):
        """Return power spectrum multipoles."""
        return self.power

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill], color=f'C{ill:d}', linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


class DampedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles):
    r"""
    Damped BAO power spectrum multipoles (Chen 2024 model).

    Implements damping of BAO wiggles via Gaussian envelope :math:`exp(-k^2*\Sigma_\mathrm{nl}^2/2)`,
    combined with bias and growth rate evolution.

    Supports pre-, reciso, recsym, and redshift-space reconstruction.

    Parameters
    ----------
    k : array, optional
        Theory wavenumbers [h/Mpc].
    ells : tuple, default=(0, 2)
        Multipole orders.
    mu : int, default=10
        Number of mu-bins.
    mode : str, default=''
        Reconstruction mode.
    smoothing_radius : float, default=15.
        Smoothing radius [Mpc/h].
    template : BasePowerSpectrumTemplate, optional
        Power spectrum template.
    model : str, default='standard'
        Damping model variant:
        - 'standard': Apply damping to wiggles only (Chen 2023)
        - 'fix-damping': Fix damping parameters across AP coordinates
        - 'move-all': Apply AP distortion to both wiggles and smooth component
        - 'fog-damping': Apply Finger-of-God to wiggles (Beutler et al. 2016)

    References
    ----------
    - Chen et al. 2024: https://arxiv.org/abs/2402.14070
    - Beutler et al. 2017: https://arxiv.org/abs/1607.03149
    """
    def initialize(self, *args, mu=10, model='standard', **kwargs):
        super().initialize(*args, **kwargs)
        self.model = str(model)
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        if self.template.only_now:
            for param in self.init.params.select(basename=['sigmapar', 'sigmaper']):
                param.update(fixed=True)

    def calculate(self, b1=1., dbeta=1., sigmas=0., sigmapar=9., sigmaper=6.):
        """
        Calculate damped BAO wiggle power spectrum.

        Parameters
        ----------
        b1 : float, default=1.
            Linear bias.
        dbeta : float, default=1.
            Relative growth rate: f = f_fid * dbeta.
        sigmas : float, default=0.
            Finger-of-God (FoG) damping parameter [Mpc/h].
        sigmapar : float, default=9.
            Parallel BAO damping [Mpc/h].
        sigmaper : float, default=6.
            Perpendicular BAO damping [Mpc/h].
        """
        super().calculate()

        # Growth rate
        f = dbeta * self.template.f

        # Apply AP distortion to (k, mu) coordinates
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknowap = _interp(self.template, 'pknow_dd', kap)
        pkap = _interp(self.template, 'pk_dd', kap)

        if self.model == 'standard':
            # Chen 2023: damping applied in distorted space
            k, mu = self.k[:, None], self.mu
            pkwap = pkap - pknowap  # Wiggles in distorted space
            sigma_nl2ap = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
            sk = 0.
            if self.mode == 'reciso':
                sk = jnp.exp(-0.5 * (k * self.smoothing_radius)**2)
            # Bias + growth in original space; damped wiggles in AP space
            Cap = (b1 + f * muap**2 * (1 - sk))**2 * jnp.exp(-sigma_nl2ap / 2.)
            fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2.
            B = (b1 + f * mu**2 * (1 - sk))**2 * fog
            pknow = _interp(self.template, 'pknow_dd', k)
            pkmu = B * pknow + Cap * pkwap
        else:
            # Alternative models: apply damping in different coordinate systems
            if 'fix-damping' in self.model:
                k, mu = self.k[:, None], self.mu
            else:
                k, mu = kap, muap
            sigma_nl2 = k**2 * (sigmapar**2 * mu**2 + sigmaper**2 * (1. - mu**2))
            damped_wiggles = (pkap - pknowap) / pknowap * jnp.exp(-sigma_nl2 / 2.)
            if 'move-all' in self.model:
                k, mu = kap, muap
            else:
                k, mu = self.k[:, None], self.mu
            pknow = _interp(self.template, 'pknow_dd', k)
            fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2.
            sk = 0.
            if self.mode == 'reciso':
                sk = jnp.exp(-0.5 * (k * self.smoothing_radius)**2)
            pksmooth = (b1 + f * mu**2 * (1 - sk))**2 * pknow
            if 'fog-damping' in self.model:  # Beutler et al. 2016
                pkmu = pksmooth * fog * (1. + damped_wiggles)
            else:  # Howlett 2023
                pkmu = pksmooth * (fog + damped_wiggles)

        # Project to multipoles
        self.power = self.to_poles(pkmu)


class ResummedPowerSpectrumWiggles(BaseCalculator):
    r"""
    Resummed BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    template : BasePowerSpectrumTemplate, optional
        Power spectrum template.
    mode : str, default=''
        Reconstruction mode ('', 'recsym', 'reciso').
    smoothing_radius : float, default=15.
        Smoothing radius [Mpc/h].
    shotnoise : float, default=0.
        Shot noise contribution [Mpc^3/h^3].

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    def initialize(self, template=None, mode='', smoothing_radius=15., shotnoise=0.):
        self.mode = str(mode)
        self.shotnoise = float(shotnoise)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template
        self.template.runtime_info.initialize()
        if is_external_cosmo(self.template.cosmo):
            self.cosmo_requires = {'thermodynamics': {'rs_drag': None}}
        self.z = self.template.z

    def calculate(self):
        """BAO damping scale from linear power spectrum."""
        self.z = self.template.z
        k = self.template.k
        pklin = self.template.pknow_dd
        q = self.template.cosmo.rs_drag
        j0 = special.jn(0, q * k)
        sk = 0.
        if self.mode: sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        # https://www.overleaf.com/project/633e1b59130591a7bf55a9cd eq. 17
        skc = 1. - sk
        self.sigma_sn2 = 1. / self.smoothing_radius / 6 / np.pi**(3. / 2.)
        self.sigma_nl2 = 1. / (3. * np.pi**2) * integrate.simpson((1. - j0) * pklin, k)
        self.sigma_dd2 = 1. / (3. * np.pi**2) * integrate.simpson((1. - j0) * skc**2 * pklin, k)
        if self.mode == 'reciso':
            self.sigma_x2 = 1. / (3. * np.pi**2) * integrate.simpson((1. - j0) * skc * pklin, k)

    def wiggles(self, k, mu, b1=1., f=0., d=1.):
        """
        Evaluate resummed BAO wiggles with reconstruction.

        Parameters
        ----------
        k : ndarray
            Wavenumber [h/Mpc].
        mu : ndarray
            Cosine of angle to line of sight.
        b1 : float, default=1.
            Linear bias.
        f : float, default=0.
            Growth rate.
        d : float, default=1.
            Growth factor relative to fiducial.

        Returns
        -------
        wiggles : ndarray
            Resummed wiggle component.
        """
        # b1 Eulerian bias, d scaling the growth factor, sigmas FoG
        wiggles = _interp(self.template, 'pk_dd', k) - _interp(self.template, 'pknow_dd', k)
        ksq = (1 + f * (f + 2) * mu**2) * k**2
        d2 = d**2
        sigma_dd2 = self.sigma_dd2 + self.shotnoise * self.sigma_sn2 / b1**2
        sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        skc = 1. - sk
        if self.mode == 'recsym':
            # Symmetric reconstruction: shift both data and randoms
            resummed_wiggles = (b1 + f * mu**2)**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_dd2)
        elif self.mode == 'reciso':
            # Isotropic reconstruction: shift data only
            resummed_wiggles = (b1 + f * mu**2 * skc - sk)**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_dd2)
            sigma_ds2 = (1. + f * mu**2) * sigma_dd2 + f * (1. + f) * mu**2 * self.sigma_x2
            resummed_wiggles += 2. * (b1 + f * mu**2 * skc - sk) * (1 + f * mu**2) * sk * jnp.exp(-1. / 2. * ksq * d2 * sigma_ds2)
            sigma_ss2 = sigma_dd2 + f**2 * mu**2 * self.sigma_nl2 + 2 * f * mu**2 * self.sigma_x2
            resummed_wiggles += (1 + f * mu**2)**2 * sk**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_ss2)
        else:
            # Redshift-space, no reconstruction
            resummed_wiggles = (b1 + f * mu**2)**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_dd2)
        return resummed_wiggles * wiggles


class ResummedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles):
    r"""
    Resummed BAO power spectrum multipoles.

    Implements EFT resummation of BAO wiggles combined with bias and growth
    rate evolution. Supports reconstruction.

    Parameters
    ----------
    k : array, optional
        Theory wavenumbers [h/Mpc].
    ells : tuple, default=(0, 2)
        Multipole orders.
    mu : int, default=10
        Number of mu-bins.
    mode : str, default=''
        Reconstruction mode.
    smoothing_radius : float, default=15.
        Smoothing radius [Mpc/h].
    template : BasePowerSpectrumTemplate, optional
        Power spectrum template.
    model : str, default='standard'
        Model variant ('standard', 'fog-damping', 'move-all').

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    _default_options = dict(shotnoise=0.)  # to be given shot noise by window matrix

    def initialize(self, *args, mu=10, model='standard', **kwargs):
        shotnoise = kwargs.pop('shotnoise', self._default_options['shotnoise'])
        super().initialize(*args, **kwargs)
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        self.model = str(model)
        self.wiggles = ResummedPowerSpectrumWiggles(mode=self.mode, template=self.template,
                                                    smoothing_radius=self.smoothing_radius,
                                                    shotnoise=shotnoise)
        if self.template.only_now:
            for param in self.init.params.select(basename=['q']):
                param.update(fixed=True)

    def calculate(self, b1=1., dbeta=1., sigmas=0., d=1., **kwargs):
        """
        Calculate resummed BAO wiggle power spectrum.

        Parameters
        ----------
        b1 : float, default=1.
            Linear bias.
        dbeta : float, default=1.
            Relative growth rate: f = f_fid * dbeta.
        sigmas : float, default=0.
            Finger-of-God damping [Mpc/h].
        d : float, default=1.
            Growth factor relative to fiducial.
        **kwargs : dict
            Additional model parameters.
        """
        # Copy z, rs_drag_fid from template
        super().calculate()
        f = dbeta * self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = _interp(self.template, 'pknow_dd', kap)
        damped_wiggles = 0. if self.template.only_now else self.wiggles.wiggles(kap, muap, b1=b1, f=f, d=d, **kwargs) / pknow
        if 'move-all' in self.model: k, mu = kap, muap
        else: k, mu = self.k[:, None], self.mu
        pknow = _interp(self.template, 'pknow_dd', k)
        fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        pksmooth = (b1 + f * mu**2 * (1 - sk))**2 * pknow
        if 'fog-damping' in self.model:  # Beutler2016
            pkmu = pksmooth * fog * (1. + damped_wiggles)
        else:  # Howlett 2023
            pkmu = pksmooth * (fog + damped_wiggles)
        self.power = self.to_poles(pkmu)


class FlexibleBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles with terms multiplying the wiggles; no damping parameter (BAO damping or Finger-of-God).
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    mu : int, default=10
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
    wiggles : str, default='pcs'
        Multiplicative wiggles kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to :math:`k^{n}` wiggles terms.
    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    """
    @staticmethod
    def _params(params, wiggles='pcs'):
        ells = [0, 2, 4]
        if wiggles == 'power':
            # Power-law wiggles: a_ell,pow * k^pow
            for ell in ells:
                for pow in range(-3, 2):
                    params[f'ml{ell:d}_{pow:d}'] = dict(value=0., ref=dict(limits=[-1e2, 1e2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, pow))
        else:
            # Kernel wiggles: sum of kernels at discrete k-points
            for ell in ells:
                for ik in range(-2, 10):  # should be more than enough
                    # We are adding a very loose prior just to regularize the fit --- parameters at the high-k end can be e.g. poorly constrained
                    # because these modes are given zero weight by the window matrix
                    params[f'ml{ell:d}_{ik:d}'] = dict(value=0., prior=dict(dist='norm', loc=0., scale=1e4), ref=dict(limits=[-1e-2, 1e-2]), delta=0.005, latex=f'a_{{{ell:d}, {ik:d}}}')
        return params

    def initialize(self, *args, mu=10, model='standard', wiggles='pcs', kp=None, **kwargs):
        super().initialize(*args, **kwargs)
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        self.model = str(model)
        self.wiggles = str(wiggles)
        if kp is None: self.kp = 2. * np.pi / self.rs_drag_fid
        else: self.kp = float(kp)
        self._set_params()
        # Fix wiggle parameters for no-wiggle-only
        if self.template.only_now:
            for param in self.init.params.select(basename='ml*_*'):
                param.update(fixed=True)

    def _set_params(self):
        """
        Build wiggle kernel matrix from parameters.

        Constructs array of kernel functions evaluated at k-bins,
        indexed by multipole order.
        """
        self.wiggles_orders = _get_orders('ml', self.init.params, self.ells)
        self.wiggles_matrix = {}
        if self.wiggles == 'power':
            # Power-law kernels
            for ell in self.ells:
                self.wiggles_matrix[ell] = jnp.array([(self.k / self.kp)**pow for pow in self.wiggles_orders[ell].values()])
        elif self.wiggles in ['ngp', 'cic', 'tsc', 'pcs']:
            # Spatial kernels: evaluate at each k-bin
            for ell in self.ells:
                tmp, bb_orders = [], {}
                for name, ik in self.wiggles_orders[ell].items():
                    kernel = _kernel_func(np.abs(self.k / self.kp - ik), kernel=self.wiggles)
                    if not np.allclose(kernel, 0., rtol=0., atol=1e-8):
                        tmp.append(kernel)
                        bb_orders[name] = ik
                self.wiggles_orders[ell] = bb_orders
                self.wiggles_matrix[ell] = jnp.array(tmp)
        else:
            raise ValueError('Unknown kernel: {}'.format(self.wiggles))
        # Keep only active parameters
        bb_params = ['b1', 'dbeta']
        for params in self.wiggles_orders.values(): bb_params += list(params)
        self.init.params = self.init.params.select(basename=bb_params)

    @jit(static_argnums=[0])
    def get_wiggles(self, wiggles, **kwargs):
        """
        Apply multipole-dependent wiggle modulation.

        Parameters
        ----------
        wiggles : ndarray, shape (n_k, n_mu)
            BAO wiggles in (k, mu) space.
        **kwargs : dict
            Wiggle amplitude parameters ml_ell_ik.

        Returns
        -------
        damped_wiggles : ndarray, shape (n_ells, n_k, n_mu)
            Multipole-projected wiggles with applied modulation.
        """
        damped_wiggles = 0.
        for ell in self.ells:
            mult = jnp.array([kwargs[name] for name in self.wiggles_orders[ell]]).dot(self.wiggles_matrix[ell])
            if ell == 0: mult += 1.
            leg = special.legendre(ell)(self.mu)
            damped_wiggles += wiggles * mult[:, None] * leg
        return damped_wiggles

    def calculate(self, b1=1., dbeta=1., **kwargs):
        """
        Calculate flexible BAO wiggle power spectrum.

        Parameters
        ----------
        b1 : float, default=1.
            Linear bias.
        dbeta : float, default=1.
            Relative growth rate: f = f_fid * dbeta.
        **kwargs : dict
            Wiggle amplitudes ml_ell_ik for each multipole ell and kernel index ik.
        """
        # Copy z, rs_drag_fid from template
        super().calculate()
        f = dbeta * self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = _interp(self.template, 'pknow_dd', kap)
        pk = _interp(self.template, 'pk_dd', kap)
        damped_wiggles = self.get_wiggles(pk - pknow, **kwargs) / pknow
        if 'move-all' in self.model: k, mu = kap, muap
        else: k, mu = self.k[:, None], self.mu
        pknow = _interp(self.template, 'pknow_dd', k)
        sk = 0.
        if self.mode == 'reciso': sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        pksmooth = (b1 + f * mu**2 * (1 - sk))**2 * pknow
        pkmu = pksmooth * (1. + damped_wiggles)
        self.power = self.to_poles(pkmu)


# --- Base classes with broadband terms ---

class BaseBAOWigglesTracerPowerSpectrumMultipoles(BaseCalculator):
    r"""
    Base class for theory BAO power spectrum multipoles, with broadband terms.

    Extends BaseBAOWigglesPowerSpectrumMultipoles with additional broadband
    parameters to capture deviation from linear theory (quasi-linear scales,
    small-scale power corrections).

    Parameters
    ----------
    k : array, optional
        Theory wavenumbers [h/Mpc].
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    mu : int, default=10
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
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.
    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.
    """
    config_fn = 'bao.yaml'
    _initialize_with_namespace = True  # to properly forward parameters to pt
    _pt_cls = BaseBAOWigglesPowerSpectrumMultipoles

    @staticmethod
    def _params(params, broadband='power'):
        """Add broadband parameters."""
        broadband = str(broadband)
        ells = [0, 2, 4]
        if 'power' in broadband:
            for ell in ells:
                for pow in range(-3, 2):
                    param = dict(value=0., ref=dict(limits=[-1e2, 1e2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, pow))
                    if broadband == 'power3' and (pow not in [-2, -1, 0]): param.update(fixed=True)
                    params[f'al{ell:d}_{pow:d}'] = param
        else:
            for ell in ells:
                for ik in range(-2, 10):  # should be more than enough for k < 0.4 h/Mpc
                    # We are adding a very loose prior just to regularize the fit --- parameters at the high-k end can be e.g. poorly constrained
                    # because these modes are given zero weight by the window matrix
                    params[f'al{ell:d}_{ik:d}'] = dict(value=0., prior=dict(dist='norm', loc=0., scale=1e4), ref=dict(limits=[-1e-2, 1e-2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, ik))
        return params

    def initialize(self, k=None, ells=(0, 2), broadband='power', kp=None, pt=None, **kwargs):
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = self._pt_cls()
        self.pt = pt
        self.pt.init.update(k=self.k, ells=self.ells, **kwargs)
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.broadband = str(broadband)
        if kp is None: self.kp = 2. * np.pi / self.pt.rs_drag_fid
        else: self.kp = float(kp)
        self._set_params()

    def _set_params(self):
        """
        Build broadband kernel matrix.

        Extracts broadband parameters from init.params and constructs
        matrix of basis functions (powers or kernels) at each k-bin.
        """
        self.broadband_orders = _get_orders('al', self.init.params, self.ells)
        self.broadband_matrix = {}
        pt_params = self.init.params.copy()
        bb_params = []
        for params in self.broadband_orders.values(): bb_params += list(params)
        self.init.params = self.init.params.select(basename=bb_params)
        for param in list(pt_params):
            if param.basename in bb_params or (param.derived is True):
                del pt_params[param]
        self.pt.init.params.update(pt_params, basename=True)
        if 'power' in self.broadband:  # even-power for the correlation function
            for ell in self.ells:
                self.broadband_matrix[ell] = jnp.array([(self.k / self.kp)**pow for pow in self.broadband_orders[ell].values()])
        elif self.broadband in ['ngp', 'cic', 'tsc', 'pcs']:
            pk_now = lambda k: _interp(self.template, 'pknow_dd_fid', k)
            for ell in self.ells:
                tmp, bb_orders = [], {}
                for name, ik in self.broadband_orders[ell].items():  # iterate over nodes
                    kernel = _kernel_func(np.abs(self.k / self.kp - ik), kernel=self.broadband)  # the kernel function
                    if not np.allclose(kernel, 0., rtol=0., atol=1e-8):
                        tmp.append(kernel * pk_now(np.clip(ik * self.kp, self.k[0], self.k[-1])))  # scale kernel by typical amplitude of Pnowiggle
                        bb_orders[name] = ik
                self.broadband_orders[ell] = bb_orders
                self.broadband_matrix[ell] = jnp.array(tmp)
        else:
            raise ValueError('Unknown kernel: {}'.format(self.broadband))
        # Only keep those terms that change the power spectrum
        bb_params = []
        for params in self.broadband_orders.values(): bb_params += list(params)
        self.init.params = self.init.params.select(basename=bb_params)

    @jit(static_argnums=[0])
    def get_broadband(self, **params):
        """
        Evaluate broadband contribution.

        Parameters
        ----------
        **params : dict
            Broadband amplitudes al_ell_ik.

        Returns
        -------
        broadband : ndarray, shape (n_ells, n_k)
            Broadband correction for each multipole and wavenumber.
        """
        return jnp.array([jnp.array([params.get(name, 0.) for name in self.broadband_orders[ell]]).dot(self.broadband_matrix[ell]) for ell in self.ells])

    def calculate(self, **params):
        """Calculate power spectrum with broadband correction."""
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.power = self.pt.power.copy() + self.get_broadband(**params)

    @property
    def template(self):
        """Get power spectrum template from internal pt."""
        return self.pt.template

    def get(self):
        """Return power spectrum multipoles."""
        return self.power

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill], color=f'C{ill:d}', linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


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
    mu : int, default=10
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
    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
        'fog-damping_move-all' to use the model of https://arxiv.org/abs/1607.03149.
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.
    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """
    _pt_cls = DampedBAOWigglesPowerSpectrumMultipoles



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
    mu : int, default=10
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
    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.
    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    _default_options = dict(shotnoise=0.)  # to be given shot noise by window matrix
    _pt_cls = ResummedBAOWigglesPowerSpectrumMultipoles


class FlexibleBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms, with flexible BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    mu : int, default=10
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
    model : str, default='standard'
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
    wiggles : str, default='pcs'
        Multiplicative wiggles kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to :math:`k^{n}` wiggles terms.
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.
    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.
    """
    _pt_cls = FlexibleBAOWigglesPowerSpectrumMultipoles


# --- Correlation function classes (via Hankel transform) ---

class BaseBAOWigglesCorrelationFunctionMultipoles(BaseCalculator):
    """
    Base class for BAO correlation function multipoles via Hankel transform.

    Converts power spectrum multipoles to real-space correlation function
    using FFTLog Hankel transform.

    Parameters
    ----------
    s : array, optional
        Theory separations [Mpc/h]. Defaults to [20, 200] with 181 points.
    ells : tuple, default=(0, 2)
        Multipole orders.
    **kwargs : dict
        Passed to power spectrum template.
    """
    _pt_cls = BaseBAOWigglesPowerSpectrumMultipoles
    # To transfer namespace to power spectrum calculator
    _initialize_with_namespace = True

    def initialize(self, s=None, ells=(0, 2), **kwargs):
        self.s = np.asarray(s) if s is not None else np.linspace(20., 200, 181)
        self.ells = tuple(ells)
        self.power = self._pt_cls(ells=self.ells, **kwargs)
        self.to_correlation = SpectrumToCorrelationMultipoles(s=self.s, spectrum=self.power)
        self.power.init.params = self.init.params.copy()
        self.init.params.clear()
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))

    def calculate(self):
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        self.corr = self.to_correlation(self.power.power)

    @property
    def template(self):
        return self.power.template

    def get(self):
        return self.corr


class DampedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    _pt_cls = DampedBAOWigglesPowerSpectrumMultipoles


class ResummedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    _pt_cls = ResummedBAOWigglesPowerSpectrumMultipoles


class BaseBAOWigglesTracerCorrelationFunctionMultipoles(BaseCalculator):
    r"""
    Base class that implements theory BAO correlation function multipoles, with broadband terms.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    mu : int, default=10
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
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'even-power' for powers of :math:`s^{2}` (motivated theoretically by Stephen Chen),
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.
    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.
    """
    config_fn = 'bao.yaml'
    # To transfer namespace to power spectrum calculator
    _initialize_with_namespace = True

    @staticmethod
    def _params(params, broadband='power'):
        broadband = str(broadband)
        ells = [0, 2, 4]
        if 'power' in broadband:
            # Configuration-space power-law broadband
            for ell in ells:
                for pow in range(-2, 3):
                    param = dict(value=0., ref=dict(limits=[-1e-3, 1e-3]), delta=0.005, latex=f'a_{{{ell:d}, {pow:d}}}')
                    if broadband == 'power3' and (pow not in [-2, -1, 0]): param.update(fixed=True)
                    if broadband == 'even-power' and (pow not in [0, 2]): param.update(fixed=True)
                    params[f'al{ell:d}_{pow:d}'] = param
        else:
            # Kernel basis (in Fourier, Hankel-transformed)
            for ell in ells:
                for ik in range(-2, 3):  # should be more than enough
                    # Infinite prior
                    param = dict(value=0., prior=None, ref=dict(limits=[-1e2, 1e2]), delta=0.005, latex=f'a_{{{ell:d}, {ik:d}}}')
                    if broadband == 'pcs2' and (ell == 0 or ik not in [0, 1]): param.update(fixed=True)
                    params[f'al{ell:d}_{ik:d}'] = param
                for ik in [0, 2]:
                    params[f'bl{ell:d}_{ik:d}'] = dict(value=0., ref=dict(limits=[-1e-3, 1e-3]), delta=0.005, latex=f'b_{{{ell:d}, {ik:d}}}')
        return params

    def initialize(self, s=None, ells=(0, 2), sp=None, broadband='power', pt=None, **kwargs):
        self.broadband = str(broadband)
        if sp is None: self.sp = 2. * np.pi / 0.02
        else: self.sp = float(sp)
        if 'power' in self.broadband:
            # Power-law broadband: use power spectrum directly
            if pt is None:
                pt = globals()[self.__class__.__name__.replace('TracerCorrelationFunction', 'PowerSpectrum')](**kwargs)
            power = pt
            self.broadband = 'power'
        else:
            # Kernel basis: transform from Fourier space
            self.broadband = self.broadband[:3]  # remove e.g. -2 from pcs2
            power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')](broadband=self.broadband, pt=pt, **kwargs)
        self.power = power
        self.s = np.asarray(s) if s is not None else np.linspace(20., 200, 101)
        self.ells = tuple(ells)
        self.power.init.update(ells=self.ells, **kwargs)
        self.to_correlation = SpectrumToCorrelationMultipoles(s=self.s, spectrum=self.power)
        self._set_params()
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))

    def _set_params(self):
        """
        Build broadband matrix for correlation function.

        For power-law: interprets k as s, kp as sp.
        For kernels: handles Fourier-space basis transformed to real space.
        """
        if 'power' in self.broadband:
            self.k, self.kp = self.s, self.sp
            # other model parameters, e.g. bias
            BaseBAOWigglesTracerPowerSpectrumMultipoles._set_params(self)
            del self.k, self.kp
        else:
            self.broadband_orders = _get_orders('bl', self.init.params, self.ells)
            self.broadband_matrix = {}

            for ell in self.ells:
                self.broadband_matrix[ell] = jnp.array([(self.s / self.sp)**pow for pow in self.broadband_orders[ell].values()])
            power_params = self.init.params.copy()
            bb_params = []
            for params in self.broadband_orders.values(): bb_params += list(params)
            self.init.params = self.init.params.select(basename=bb_params)
            for param in list(power_params):
                if param.basename in bb_params: del power_params[param]
            self.power.init.params = power_params

    def calculate(self, **params):
        """Calculate correlation function with broadband."""
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        self.corr = self.to_correlation(self.power.power)
        self.corr += jnp.array([jnp.array([params.get(name, 0.) for name in self.broadband_orders[ell]]).dot(self.broadband_matrix[ell]) for ell in self.ells])

    @property
    def pt(self):
        """Get internal power spectrum object."""
        return getattr(self.power, 'pt', self.power)

    @property
    def template(self):
        return self.power.template

    @property
    def wiggle(self):
        return self.power.wiggle

    @wiggle.setter
    def wiggle(self, wiggle):
        self.power.wiggle = wiggle

    def get(self):
        """Return correlation function multipoles."""
        return self.corr

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot correlation function multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self.s, self.s**2 * self.corr[ill], color=f'C{ill:d}', linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig


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
    mu : int, default=10
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
    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
        'fog-damping_move-all' to use the model of https://arxiv.org/abs/1607.03149.
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.
    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.

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
    mu : int, default=10
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
    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.
    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    _default_options = dict(shotnoise=0.)  # to be given shot noise by window matrix


class FlexibleBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    Theory BAO correlation function multipoles, with broadband terms, with flexible BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    mu : int, default=10
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
    model : str, default='standard'
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
    wiggles : str, default='pcs'
        Multiplicative wiggles kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to :math:`k^{n}` wiggles terms.
    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.
    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.
    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.
    """