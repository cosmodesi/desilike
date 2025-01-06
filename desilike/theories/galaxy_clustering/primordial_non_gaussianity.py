import numpy as np
from scipy import constants

from desilike import plotting
from desilike.jax import interp1d
from desilike.jax import numpy as jnp
from .base import BaseTheoryPowerSpectrumMultipolesFromWedges
from .power_template import FixedPowerSpectrumTemplate


class PNGTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Kaiser tracer power spectrum multipoles, with scale dependent bias sourced by local primordial non-Gaussianities.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
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

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).

    Reference
    ---------
    https://arxiv.org/pdf/1904.08859.pdf
    """
    config_fn = 'primordial_non_gaussianity.yaml'

    def initialize(self, *args, ells=(0, 2), method='prim', mode='b-p', template=None, shotnoise=1e4, **kwargs):
        super(PNGTracerPowerSpectrumMultipoles, self).initialize(*args, ells=ells, **kwargs)
        self.nd = 1. / shotnoise
        if template is None:
            template = FixedPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(1e-3, self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(1., self.k[-1] * 2, self.template.init.get('k', [0.])[0]), 1000)
        kin = np.insert(kin, 0, 1e-4)
        self.template.init.update(k=kin)
        self.method = str(method)
        self.mode = str(mode)
        keep_params = ['b1', 'sigmas', 'sn0']
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

    def calculate(self, b1=2., sigmas=0., sn0=0., **params):
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
        fog = 1. / (1. + sigmas**2 * kap**2 * muap**2 / 2.)**2.
        pkmu = jac * fog * (bias + f * muap**2)**2 * interp1d(jnp.log10(kap), np.log10(kin), pk_dd) + sn0 / self.nd
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
