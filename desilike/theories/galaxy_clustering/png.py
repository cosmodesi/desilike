import numpy as np
from scipy import constants

from desilike import plotting, utils
from desilike.jax import interp1d
from desilike.jax import numpy as jnp
from .power_template import FixedPowerSpectrumTemplate
from .base import ProjectToMultipoles
from .full_shape import BasePTPowerSpectrumMultipoles, BaseTracerPowerSpectrumMultipoles, MultitracerBiasParameters


class PNGTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    Kaiser tracer power spectrum multipoles, with scale dependent bias sourced by local primordial non-Gaussianities.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).
    method : str, default='prim'
        Method to compute :math:`\alpha`, which relates primordial potential to current density contrast.
        - "prim": :math:`\alpha` is the square root of the primordial power spectrum to the current density power spectrum
        - else: :math:`\alpha` is the transfer function, rescaled by the factor in the Poisson equation, and the growth rate,
          normalized to :math:`1 / (1 + z)` at :math:`z = 10` (in the matter dominated era).
    mode : str, default='b-p'
        fnl_loc is degenerate with PNG bias bphi.

        - "b-p": ``bphi = 2 * 1.686 * (b1 - p)``, p as a parameter
        - "bphi": ``bphi`` as a parameter
        - "bfnl_loc": ``bfnl_loc = bphi * fnl_loc`` as a parameter
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`FixedPowerSpectrumTemplate`.
    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).

    Reference
    ---------
    https://arxiv.org/pdf/1904.08859.pdf
    """
    config_fn = 'png.yaml'

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers, deterministic=['b1', 'sigmas', 'bphi', 'p', 'bfnl_loc'], stochastic=['sn0'], ntracers=2)

    @classmethod
    def _params(cls, params, tracers=None, mode='b-p'):
        keep_params = ['b1', 'sigmas', 'sn0']
        # tnl_loc (local tauNL) is fixed to 0 by default (see png.yaml), so
        # fNL-only analyses are unaffected; free it for tauNL / joint fits.
        # Only the 'b-p' / 'bphi' modes expose bphi, which the tauNL term
        # needs; 'bfnl' mode is left as-is (fNL-only).
        if mode == 'bphi':
            keep_params += ['fnl_loc', 'tnl_loc', 'bphi']
        elif mode == 'b-p':
            keep_params += ['fnl_loc', 'tnl_loc', 'p']
        elif mode == 'bfnl':
            keep_params += ['bfnl_loc']
        else:
            raise ValueError('Unknown mode {}; it must be one of ["bphi", "b-p", "bfnl"]'.format(mode))
        params = params.select(basename=keep_params)
        params = cls._get_multitracer(tracers=tracers)._params(params)
        return params

    def initialize(self, k=None, ells=(0, 2), mu=10, tracers=None, z=None, method='prim', mode='b-p', template=None):
        self._set_options(k=k, ells=ells, tracers=tracers)
        if template is None:
            template = FixedPowerSpectrumTemplate()
        BasePTPowerSpectrumMultipoles._set_template(self, template=template, z=z)
        kin = np.geomspace(min(1e-3, self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(1., self.k[-1] * 2, self.template.init.get('k', [0.])[0]), 1000)
        kin = np.insert(kin, 0, 1e-4)
        self.template.init.update(k=kin)
        self.method = str(method)
        self.mode = str(mode)
        self.z = self.template.z
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        self.decode_params = self._get_multitracer(tracers=tracers)

    def calculate(self, **params):
        params = self.decode_params(params, defaults=dict(b1=1., sigmas=0., sn0=0., bphi=1., p=1., bfnl_loc=0., tnl_loc=0.))
        (b1X, b1Y), (sigmasX, sigmasY), sn0 = [params[name] for name in ['b1', 'sigmas', 'sn0']]
        self.z = self.template.z
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pk_dd = self.template.pk_dd
        kin = self.template.k
        cosmo = self.template.cosmo
        f = self.template.f
        pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(kin)  # power_prim is ~ k^(n_s - 1)
        if self.method == 'prim':
            pphi_prim = 9 / 25 * 2 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
            alpha = 1. / (pk_dd / pphi_prim)**0.5
        else:
            # Normalization in the matter dominated era
            # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
            tk = (pk_dd / pk_prim / kin / (pk_dd[0] / pk_prim[0] / kin[0]))**0.5
            znorm = 10.
            normalized_growth_factor = cosmo.growth_factor(self.template.z) / cosmo.growth_factor(znorm) / (1 + znorm)
            alpha = 3. * cosmo.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * kin**2 * tk * normalized_growth_factor)
        # Remove first k, used to normalize tk
        kin, pk_dd, alpha = kin[1:], pk_dd[1:], alpha[1:]
        alpha = interp1d(jnp.log10(kap), np.log10(kin), alpha)
        bphiX = bphiY = None  # PNG bias bphi = 2 delta_c (b1 - p); None in 'bfnl' mode (not separable)
        if self.mode == 'bphi':
            fnl_loc = params['fnl_loc']
            bphiX, bphiY = params['bphi']
            bfnl_locX, bfnl_locY = bphiX * fnl_loc, bphiY * fnl_loc
        elif self.mode == 'b-p':
            fnl_loc = params['fnl_loc']
            pX, pY = params['p']
            bphiX, bphiY = [2. * 1.686 * (b1 - p) for b1, p in [(b1X, pX), (b1Y, pY)]]
            bfnl_locX, bfnl_locY = bphiX * fnl_loc, bphiY * fnl_loc
        else:

            bfnl_locX, bfnl_locY = params['bfnl_loc']
        # bfnl_loc is typically 2 * delta_c * (b1 - p) * fnl_loc
        bX, bY = b1X + bfnl_locX * alpha, b1Y + bfnl_locY * alpha
        fog = 1. / ((1. + sigmasX**2 * kap**2 * muap**2 / 2.) * (1. + sigmasY**2 * kap**2 * muap**2 / 2.))
        pk_obs = interp1d(jnp.log10(kap), np.log10(kin), pk_dd)
        # Deterministic (Kaiser) term: fNL is a scale-dependent bias shift
        # bX = b1 + bfnl_loc*alpha, sitting INSIDE (b + f mu^2) -> it IS
        # boosted by f mu^2 and contributes to all multipoles.
        pkmu = jac * fog * (bX + f * muap**2) * (bY + f * muap**2) * pk_obs + sn0 / self.nbar
        # tauNL (local trispectrum): stochastic scale-dependent bias
        # contribution (Ferraro & Smith 2014, eq. 8). Added as standalone
        # power -- NOT inside the Kaiser (b + f mu^2) factor and NOT boosted
        # by f mu^2 (isotropic -> monopole only), with a steeper k^-4 shape
        # (alpha^2 vs alpha for fNL). Rank-one in tracer space (bphiX bphiY).
        # tnl_loc is fixed to 0 by default (png.yaml) -> fNL-only analyses
        # are bit-identical. Only available in 'b-p'/'bphi' modes, where
        # bphi is defined.
        # TODO(future): to constrain tauNL (or tnl*bphi^2) WITHOUT assuming
        # bphi, add a 'btnl'-style mode fitting the bphi-folded combination
        # directly (auto-spectrum only; cannot factorize the cross-spectra).
        # CONVENTION (important for JOINT fnl+tnl fits): the Kaiser product
        # (bX + f mu^2)(bY + f mu^2) above already contains the deterministic
        # Delta_b^2 = fnl^2 bphiX bphiY alpha^2 term. The correct stochastic
        # power is proportional to the Suyama-Yamaguchi EXCESS, tnl - (6/5 fnl)^2
        # (no stochasticity when SY is saturated). Since we add (25/36) tnl_loc
        # bphi^2 alpha^2 on TOP of the Kaiser Delta_b^2, this parameter is the
        # SY-excess:   tnl_loc = tauNL - (6/5 fnl_loc)^2 = tauNL - (36/25) fnl^2.
        #   * fnl-only (tnl_loc=0): unchanged, includes the standard Delta_b^2.
        #   * tnl-only (fnl_loc=0): tnl_loc == tauNL exactly (our validation case).
        #   * joint: report full tauNL = tnl_loc + (36/25) fnl_loc^2, OR make
        #     tnl_loc the FULL tauNL by using coefficient ((25/36)*tnl_loc -
        #     fnl_loc**2) below (this then drops Delta_b^2, i.e. changes the
        #     fnl-only model -> breaks bit-compat, so left off by default).
        # NB: the tnl_loc >= 0 positivity prior is then exactly the SY bound.
        tnl_loc = params['tnl_loc']
        if bphiX is not None:
            pkmu = pkmu + jac * fog * (25. / 36.) * tnl_loc * bphiX * bphiY * alpha**2 * pk_obs
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power

    @plotting.plotter
    def plot(self, fig=None, scaling='loglog'):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        scaling : str, default='loglog'
            Either 'kpk' or 'loglog'.
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
        k_exp = 1 if scaling == 'kpk' else 0
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k**k_exp * self.power[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        if scaling == 'kpk':
            ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if scaling == 'loglog':
            ax.set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig

class PNGTracerVelocityPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    Kaiser tracer-velocity power spectrum multipoles, with scale dependent bias sourced by local primordial non-Gaussianities.

    **Warning:** We model infact -iP(k) in order to avoid any trouble if complex. Need to take the abs value of the power spectrum estimator.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(1, 3)
        Multipoles to compute.

    mu : int, default=200
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    method : str, default='prim'
        Method to compute :math:`\alpha`, which relates primordial potential to current density contrast.

        - "prim": :math:`\alpha` is the square root of the primordial power spectrum to the current density power spectrum
        - else: :math:`\alpha` is the transfer function, rescaled by the factor in the Poisson equation, and the growth rate,
          normalized to :math:`1 / (1 + z)` at :math:`z = 10` (in the matter dominated era).

    mode : str, default='b-p'
        fnl_loc is degenerate with PNG bias bphi.

        - "b-p": ``bphi = 2 * 1.686 * (b1 - p)``, p as a parameter
        - "bphi": ``bphi`` as a parameter
        - "bfnl_loc": ``bfnl_loc = bphi * fnl_loc`` as a parameter

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`FixedPowerSpectrumTemplate`.

    Reference
    ---------
    To be added... 
    """
    config_fn = 'png.yaml'

    def initialize(self, *args, ells=(1, 3), method='prim', mode='b-p', template=None, **kwargs):
        super(PNGTracerVelocityPowerSpectrumMultipoles, self).initialize(*args, ells=ells,  **kwargs)
        self.to_poles = ProjectToMultipoles(mu=np.linspace(-1, 1, 81), method='trapz', ells=self.ells)
        self.mu = self.to_poles.mu
        if template is None:
            template = FixedPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(1e-3, self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(1., self.k[-1] * 2, self.template.init.get('k', [0.])[0]), 1000)
        kin = np.insert(kin, 0, 1e-4)
        self.template.init.update(k=kin)
        self.method = str(method)
        self.mode = str(mode)
        keep_params = ['b1', 'bv', 'sigmas', 'sigmau']
        if self.mode == 'bphi':
            keep_params += ['fnl_loc', 'bphi']
        elif self.mode == 'b-p':
            keep_params += ['fnl_loc', 'p']
        elif self.mode == 'bfnl':
            keep_params += ['bfnl_loc']
        else:
            raise ValueError('Unknown mode {}; it must be one of ["bphi", "b-p", "bfnl"]'.format(self.mode))
        self.z = self.template.z
        self.params = self.params.select(basename=keep_params)

    def calculate(self, b1=2., bv=1., sigmas=0., sigmau=0., **params):
        self.z = self.template.z
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pk_dd = self.template.pk_dd
        kin = self.template.k
        cosmo = self.template.cosmo
        f = self.template.f
        pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(kin)  # power_prim is ~ k^(n_s - 1)
        if self.method == 'prim':
            pphi_prim = 9 / 25 * 2 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
            alpha = 1. / (pk_dd / pphi_prim)**0.5
        else:
            # Normalization in the matter dominated era
            # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
            tk = (pk_dd / pk_prim / kin / (pk_dd[0] / pk_prim[0] / kin[0]))**0.5
            znorm = 10.
            normalized_growth_factor = cosmo.growth_factor(self.template.z) / cosmo.growth_factor(znorm) / (1 + znorm)
            alpha = 3. * cosmo.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * kin**2 * tk * normalized_growth_factor)
        # Remove first k, used to normalize tk
        kin, pk_dd, alpha = kin[1:], pk_dd[1:], alpha[1:]
        if self.mode == 'bphi':
            fnl_loc = params['fnl_loc']
            bphi = params['bphi']
            bfnl_loc = bphi * fnl_loc
        elif self.mode == 'b-p':
            fnl_loc = params['fnl_loc']
            p = params.get('p', 1.)
            bfnl_loc = 2. * 1.686 * (b1 - p) * fnl_loc
        else:
            bfnl_loc = params['bfnl_loc']
        # bfnl_loc is typically 2 * delta_c * (b1 - p)
        bias = b1 + bfnl_loc * interp1d(jnp.log10(kap), np.log10(kin), alpha)
        # Velocity term: We do not include the 1j --> will remove it in the data as well.
        vel_bias = bv * f * muap * 100 / (1 + self.z) / kap
        # Finger of God terms:
        fog = 1. / (1. + sigmas**2 * kap**2 * muap**2 / 2.) * np.sinc(sigmau * kap)
        # Power spectrum:
        pkmu = jac * fog * (bias + f * muap**2) * vel_bias * interp1d(jnp.log10(kap), np.log10(kin), pk_dd) 
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power

    @plotting.plotter
    def plot(self, fig=None, scaling='loglog', figsize=None):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        scaling : str, default='loglog'
            Either 'kpk' or 'loglog'.
        
        figsize : (width, height), default=None
            If not figure is passed, fix the size of the created figure.

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
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.axes[0]
        k_exp = 1 if scaling == 'kpk' else 1
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k**k_exp * self.power[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        if scaling == 'kpk':
            ax.set_ylabel(r'$-ik P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}\mathrm{km}/s$]')
        if scaling == 'loglog':
            ax.set_ylabel(r'$-ik P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}\mathrm{km}/s$]')
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig

