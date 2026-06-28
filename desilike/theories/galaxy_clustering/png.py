"""
Primordial Non-Gaussianity (PNG) power spectrum multipoles.

Classes
-------
PNGTracerSpectrum2Poles
    Kaiser tracer density power spectrum multipoles with local PNG scale-dependent bias.
PNGTracerVelocitySpectrum2Poles
    Kaiser tracer-velocity cross power spectrum multipoles with local PNG scale-dependent bias.

The scale-dependent function :math:`\\alpha(k)` is computed from the template's cosmo at each
call (JAX-friendly), supporting automatic differentiation through all cosmological parameters.
"""

import numpy as np
import jax.numpy as jnp

from ...base import Calculator
from ...parameter import Parameter, VariableCollection
from ..primordial_cosmology import _interp_loglog
from .bao import ProjectToPoles
from .template import FixedSpectrum2Template
from ._multitracer import propose_params_multitracer, assign_params


_delta_c = 1.686  # linear collapse threshold
_C_KMS = 299792.458  # speed of light in km/s


def _alpha_png(k, pk_dd, pk_prim, h, method, Omega0_m=None, growth_factor_z=None, growth_factor_znorm=None):
    r"""Compute the PNG scale-dependent function :math:`\alpha(k)` from JAX arrays.

    Parameters
    ----------
    k : array, shape (n_k,)
        Wavenumbers in h/Mpc.  For the ``'transfer'`` method, ``k[0]`` should be small
        (~1e-4 h/Mpc) so that T(k[0]) ≈ 1 is a valid normalization point.
    pk_dd : array, shape (n_k,)
        Matter power spectrum in (Mpc/h)^3, same k grid.
    pk_prim : array, shape (n_k,)
        Primordial scalar power spectrum from cosmoprimo (same k grid).
    h : scalar
        Reduced Hubble constant H0 / (100 km/s/Mpc).
    method : str
        ``'prim'`` or ``'transfer'``.
    Omega0_m : scalar, optional
        Total matter density parameter at z = 0 (required for ``'transfer'``).
    growth_factor_z : scalar, optional
        Linear growth factor D(z) (required for ``'transfer'``).
    growth_factor_znorm : scalar, optional
        Linear growth factor D(z_norm=10) (required for ``'transfer'``).

    Returns
    -------
    alpha : array, shape (n_k,)

    References
    ----------
    Dalal et al. 2008  https://arxiv.org/abs/0710.4560
    Barreira 2020  https://arxiv.org/pdf/1904.08859.pdf
    """
    if method == 'prim':
        pphi_prim = 9. / 25. * 2. * jnp.pi**2 / k**3 * pk_prim / h**3
        return 1. / jnp.sqrt(pk_dd / pphi_prim)
    else:  # 'transfer'
        znorm = 10.
        growth_ratio = growth_factor_z / (growth_factor_znorm * (1. + znorm))
        tk = jnp.sqrt(pk_dd / pk_prim / k / (pk_dd[0] / pk_prim[0] / k[0]))
        return 3. * Omega0_m * 100.**2 / (2. * _C_KMS**2 * k**2 * tk * growth_ratio)


class PNGTracerSpectrum2Poles(Calculator):
    r"""
    Kaiser tracer power spectrum multipoles with local PNG scale-dependent bias.

    The scale-dependent bias is :math:`b_\mathrm{eff}(k) = b_1 + b_{f_\mathrm{NL}} \alpha(k)`,
    where :math:`b_{f_\mathrm{NL}} = b_\phi f_\mathrm{NL}` and :math:`b_\phi = 2 \delta_c (b_1 - p)`
    in the universal mass function approximation.

    :math:`\alpha(k)` is computed at each call from the template's cosmology, making the model
    fully JAX-differentiable through all cosmological parameters.  AP distortions are applied
    via ``template.ap_k_mu``.

    For cross-spectra between two tracers :math:`X` and :math:`Y`, the power spectrum is
    :math:`\mathrm{FoG}_X \mathrm{FoG}_Y (b^\mathrm{eff}_X + f\mu^2)(b^\mathrm{eff}_Y + f\mu^2) P_{dd}`.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to ``np.linspace(0.01, 0.2, 101)``.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    z : float, default=1.
        Effective redshift.
    method : str, default='prim'
        How to compute :math:`\alpha(k)`:

        - ``'prim'``: :math:`\alpha = \sqrt{P_\phi^\mathrm{prim}(k) / P_\delta(k)}`.
        - ``'transfer'``: from the transfer function normalized in the matter-dominated
          era at :math:`z_\mathrm{norm}=10`; see eq. 2.3 of arXiv:1904.08859.
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    mode : str, default='b-p'
        Parameterization of the PNG bias:

        - ``'b-p'``: :math:`b_{f_\mathrm{NL}} = 2\delta_c(b_1 - p) f_\mathrm{NL}`;
          free params ``b1``, ``p``, ``fnl_loc``.
        - ``'bphi'``: :math:`b_{f_\mathrm{NL}} = b_\phi f_\mathrm{NL}`;
          free params ``b1``, ``bphi``, ``fnl_loc``.
        - ``'bfnl'``: :math:`b_{f_\mathrm{NL}}` directly; free params ``b1``, ``bfnl_loc``.
    tracers : str, (str, str), or None, default=None
        Tracer namespacing of the bias parameters (auto, namespaced auto, or cross).
        ``fnl_loc`` stays unnamespaced (shared); ``sn0`` is stochastic.
    nbar : float, default=1e-4
        Number density [(Mpc/h)\ :sup:`-3`]. The ``sn0`` parameter is in units of ``1/nbar``.
    template : Spectrum2Template
        Power spectrum template providing :math:`P_{dd}`, :math:`f`, AP distortions,
        and the underlying cosmology for :math:`\alpha(k)`.
    """

    @classmethod
    def propose_params(cls, tracers=None, mode='b-p'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        mode : str, default='b-p'
            One of ``'b-p'``, ``'bphi'``, ``'bfnl'``.

        Returns
        -------
        VariableCollection
        """
        if mode not in ('b-p', 'bphi', 'bfnl'):
            raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        auto_params = [
            Parameter('b1', value=2., prior=dict(limits=[0.1, 10.]),
                      ref=dict(limits=[1.5, 2.5]), fd_eps=0.1, latex='b_1'),
            Parameter('sigmas', value=0., prior=dict(limits=[0., 10.]),
                      ref=dict(limits=[1., 4.]), fd_eps=0.2, latex=r'\Sigma_{s}'),
            Parameter('sn0', value=0., prior=dict(dist='norm', loc=0., scale=1000.),
                      ref=dict(dist='norm', loc=0., scale=0.1), fd_eps=0.05, latex='s_{n,0}'),
        ]
        if mode == 'b-p':
            auto_params += [
                Parameter('fnl_loc', value=0., prior=dict(limits=[-300., 300.]),
                          ref=dict(limits=[-10., 10.]), fd_eps=1., latex=r'f_{\mathrm{NL}}^{\mathrm{loc}}'),
                Parameter('p', value=1., prior=dict(limits=[0., 3.]), ref=dict(limits=[0.5, 1.5]), fd_eps=0.1, latex='p'),
            ]
        elif mode == 'bphi':
            auto_params += [
                Parameter('fnl_loc', value=0., prior=dict(limits=[-300., 300.]),
                          ref=dict(limits=[-10., 10.]), fd_eps=1., latex=r'f_{\mathrm{NL}}^{\mathrm{loc}}'),
                Parameter('bphi', value=1., prior=dict(limits=[-10., 10.]), ref=dict(limits=[3., 4.]), fd_eps=0.1, latex=r'b_{\phi}'),
            ]
        else:
            auto_params += [
                Parameter('bfnl_loc', value=0., prior=dict(limits=[-1e3, 1e3]),
                          ref=dict(limits=[-50., 50.]), fd_eps=1., latex=r'b_{\phi}f_{\mathrm{NL}}^{\mathrm{loc}}'),
            ]
        return propose_params_multitracer(auto_params, tracers, stochastic=('sn0',), shared=('fnl_loc',), cross=True)

    def __init__(self, k=None, ells=(0, 2), method='prim', mu=10, mode='b-p',
                 tracers=None, nbar=1e-4, params=None, template=None):
        if mode not in ('b-p', 'bphi', 'bfnl'):
            raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        vc = type(self).propose_params(tracers=tracers, mode=mode)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
        if template is None:
            template = FixedSpectrum2Template()
        self.template = template
        k_arr = np.linspace(0.01, 0.2, 101) if k is None else np.asarray(k, dtype='f8')
        # Extend to 1e-4 at the low end so the 'transfer' normalization point is in-grid.
        kin_fine = np.geomspace(min(1e-4, k_arr[0] / 2.), max(1., k_arr[-1] * 2.), 1000)
        self.template.update(k=kin_fine)

    def __post_init__(self, k=None, ells=(0, 2), method='prim', mu=10, mode='b-p',
                      tracers=None, nbar=1e-4, params=None, template=None):
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self._mode = str(mode)
        self._method = str(method)
        self._z = float(self.template.z)
        self._nbar = float(nbar)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        reqs = {'primordial.pk': [{'k': self.template.k}]}
        if self._method == 'transfer':
            reqs.update({'background.growth_factor': [{'z': self._z}, {'z': 10.}],
                         'params.Omega_m': None})
        self.template.cosmo.add_requirements(reqs)
        self.template.cosmo()

    def __call__(self):
        k = self.k[:, None]       # (n_k, 1)
        mu = self._to_poles.mu    # (n_mu,)

        jac, kap, muap = self.template.ap_k_mu(k, mu)
        pk_dd = jac * _interp_loglog(kap, self.template.k, self.template.pk_dd)

        h = self.template.cosmo['h']
        pk_prim_fine = self.template.cosmo.get('primordial.pk', k=self.template.k)
        if self._method == 'transfer':
            Omega_m = self.template.cosmo.get('params.Omega_m')
            growth_factor_z = self.template.cosmo.get('background.growth_factor', z=self._z)
            growth_factor_znorm = self.template.cosmo.get('background.growth_factor', z=10.)
        else:
            Omega_m = growth_factor_z = growth_factor_znorm = None
        alpha_fine = _alpha_png(self.template.k, self.template.pk_dd, pk_prim_fine, h, self._method,
                                Omega0_m=Omega_m, growth_factor_z=growth_factor_z,
                                growth_factor_znorm=growth_factor_znorm)
        alpha = _interp_loglog(kap, self.template.k, alpha_fine)
        f = self.template.f

        if isinstance(self.b1, tuple):  # cross-spectrum
            b1_X, b1_Y = self.b1
            sigmas_X, sigmas_Y = self.sigmas
            if self._mode == 'b-p':
                p_X, p_Y = self.p
                bfnl_loc_X = 2. * _delta_c * (b1_X - p_X) * self.fnl_loc
                bfnl_loc_Y = 2. * _delta_c * (b1_Y - p_Y) * self.fnl_loc
            elif self._mode == 'bphi':
                bphi_X, bphi_Y = self.bphi
                bfnl_loc_X = bphi_X * self.fnl_loc
                bfnl_loc_Y = bphi_Y * self.fnl_loc
            else:  # 'bfnl'
                bfnl_loc_X, bfnl_loc_Y = self.bfnl_loc
            b_eff_X = b1_X + bfnl_loc_X * alpha
            b_eff_Y = b1_Y + bfnl_loc_Y * alpha
            fog_X = 1. / (1. + sigmas_X**2 * kap**2 * muap**2 / 2.)
            fog_Y = 1. / (1. + sigmas_Y**2 * kap**2 * muap**2 / 2.)
            pkmu = fog_X * fog_Y * (b_eff_X + f * muap**2) * (b_eff_Y + f * muap**2) * pk_dd
        else:
            if self._mode == 'b-p':
                bfnl_loc = 2. * _delta_c * (self.b1 - self.p) * self.fnl_loc
            elif self._mode == 'bphi':
                bfnl_loc = self.bphi * self.fnl_loc
            else:  # 'bfnl'
                bfnl_loc = self.bfnl_loc
            b_eff = self.b1 + bfnl_loc * alpha
            fog = 1. / (1. + self.sigmas**2 * kap**2 * muap**2 / 2.)**2
            pkmu = fog * (b_eff + f * muap**2)**2 * pk_dd

        sn = jnp.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * self.sn0 / self._nbar
        self.poles = self._to_poles(pkmu) + sn
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class PNGTracerVelocitySpectrum2Poles(Calculator):
    r"""
    Kaiser tracer-velocity cross power spectrum multipoles with local PNG scale-dependent bias.

    Models :math:`-i P_{gv}(k, \mu)` (the imaginary prefactor is dropped so all outputs are real;
    the data estimator must be adjusted accordingly).  Computes odd multipoles :math:`\ell = 1, 3`.

    The velocity bias reads :math:`v(k, \mu) = b_v f \mu H_0 / [(1+z) k]`.  :math:`\alpha(k)` is
    computed at each call from the template's cosmology (JAX-friendly).

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to ``np.linspace(0.01, 0.2, 101)``.
    ells : tuple of int, default=(1, 3)
        Multipole orders (should be odd).
    method : str, default='prim'
        How to compute :math:`\alpha(k)`; ``'prim'`` or ``'transfer'``.
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    mode : str, default='b-p'
        PNG bias parameterization; same options as :class:`PNGTracerSpectrum2Poles`.
    template : Spectrum2Template
        Power spectrum template providing :math:`P_{dd}`, :math:`f`, AP distortions,
        and the underlying cosmology for :math:`\alpha(k)`.
    """

    def __init__(self, k=None, ells=(1, 3), method='prim', mu=10, mode='b-p', template=None):
        self.b1 = Parameter('b1', value=2., prior=dict(limits=[0.1, 10.]),
                            ref=dict(limits=[1.5, 2.5]), fd_eps=0.1, latex='b_1')
        self.bv = Parameter('bv', value=1., prior=dict(limits=[0.1, 10.]),
                            ref=dict(limits=[0.5, 1.5]), fd_eps=0.1, latex='b_v')
        self.sigmas = Parameter('sigmas', value=0., prior=dict(limits=[0., 10.]),
                                ref=dict(limits=[1., 4.]), fd_eps=0.2, latex=r'\Sigma_{s}')
        self.sigmau = Parameter('sigmau', value=0., prior=dict(limits=[0., 50.]),
                                ref=dict(limits=[0., 20.]), fd_eps=0.2, latex=r'\Sigma_{u}')
        if mode == 'b-p':
            self.fnl_loc = Parameter('fnl_loc', value=0., prior=dict(limits=[-300., 300.]),
                                     ref=dict(limits=[-10., 10.]), fd_eps=1., latex=r'f_{\mathrm{NL}}^{\mathrm{loc}}')
            self.p = Parameter('p', value=1., prior=dict(limits=[0., 3.]),
                               ref=dict(limits=[0.5, 1.5]), fd_eps=0.1, latex='p')
        elif mode == 'bphi':
            self.fnl_loc = Parameter('fnl_loc', value=0., prior=dict(limits=[-300., 300.]),
                                     ref=dict(limits=[-10., 10.]), fd_eps=1., latex=r'f_{\mathrm{NL}}^{\mathrm{loc}}')
            self.bphi = Parameter('bphi', value=1., prior=dict(limits=[-10., 10.]),
                                  ref=dict(limits=[3., 4.]), fd_eps=0.1, latex=r'b_{\phi}')
        elif mode == 'bfnl':
            self.bfnl_loc = Parameter('bfnl_loc', value=0., prior=dict(limits=[-1e3, 1e3]),
                                      ref=dict(limits=[-50., 50.]), fd_eps=1., latex=r'b_{\phi}f_{\mathrm{NL}}^{\mathrm{loc}}')
        else:
            raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        if template is None:
            template = FixedSpectrum2Template()
        self.template = template
        k_arr = np.linspace(0.01, 0.2, 101) if k is None else np.asarray(k, dtype='f8')
        kin_fine = np.geomspace(min(1e-4, k_arr[0] / 2.), max(1., k_arr[-1] * 2.), 1000)
        self.template.update(k=kin_fine)

    def __post_init__(self, k=None, ells=(1, 3), method='prim', mu=10, mode='b-p', template=None):
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self._mode = str(mode)
        self._method = str(method)
        self._z = float(self.template.z)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        reqs = {'primordial.pk': [{'k': self.template.k}]}
        if self._method == 'transfer':
            reqs.update({'background.growth_factor': [{'z': self._z}, {'z': 10.}],
                         'params.Omega_m': None})
        self.template.cosmo.add_requirements(reqs)
        self.template.cosmo()

    def __call__(self):
        k = self.k[:, None]       # (n_k, 1)
        mu = self._to_poles.mu    # (n_mu,)

        jac, kap, muap = self.template.ap_k_mu(k, mu)
        pk_dd = jac * _interp_loglog(kap, self.template.k, self.template.pk_dd)

        pk_prim_fine = self.template.cosmo.get('primordial.pk', k=self.template.k)
        h = self.template.cosmo['h']
        if self._method == 'transfer':
            Omega_m = self.template.cosmo.get('params.Omega_m')
            growth_factor_z = self.template.cosmo.get('background.growth_factor', z=self._z)
            growth_factor_znorm = self.template.cosmo.get('background.growth_factor', z=10.)
        else:
            Omega_m = growth_factor_z = growth_factor_znorm = None
        alpha_fine = _alpha_png(self.template.k, self.template.pk_dd, pk_prim_fine, h, self._method,
                                Omega0_m=Omega_m, growth_factor_z=growth_factor_z,
                                growth_factor_znorm=growth_factor_znorm)
        alpha = _interp_loglog(kap, self.template.k, alpha_fine)
        f = self.template.f

        if self._mode == 'b-p':
            bfnl_loc = 2. * _delta_c * (self.b1 - self.p) * self.fnl_loc
        elif self._mode == 'bphi':
            bfnl_loc = self.bphi * self.fnl_loc
        else:  # 'bfnl'
            bfnl_loc = self.bfnl_loc

        b_eff = self.b1 + bfnl_loc * alpha
        fog = 1. / (1. + self.sigmas**2 * kap**2 * muap**2 / 2.) * jnp.sinc(self.sigmau * kap)
        vel_bias = self.bv * f * muap * 100. / (1. + self._z) / kap
        pkmu = fog * (b_eff + f * muap**2) * vel_bias * pk_dd
        self.poles = self._to_poles(pkmu)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj
