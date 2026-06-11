"""
Primordial Non-Gaussianity (PNG) power spectrum multipoles.

Classes
-------
PNGTracerSpectrum2Poles
    Kaiser tracer density power spectrum multipoles with local PNG scale-dependent bias.
PNGTracerVelocitySpectrum2Poles
    Kaiser tracer-velocity cross power spectrum multipoles with local PNG scale-dependent bias.

The scale-dependent function :math:`\\alpha(k)` and the matter power spectrum are computed
once at compile time from a fixed fiducial cosmology (mirroring desilike_bak's
``FixedPowerSpectrumTemplate``); only the bias / :math:`f_\\mathrm{NL}` parameters vary.
"""

import numpy as np
import jax.numpy as jnp

from ...base import Calculator
from ...parameter import Parameter, VariableCollection
from .bao import ProjectToPoles
from .template import _get_fiducial, _kw_pk
from ._multitracer import propose_params_multitracer, assign_params


_delta_c = 1.686  # linear collapse threshold


def _png_cosmo(fiducial, k, z, method, engine):
    r"""Compute the PNG ingredients at wavenumbers *k* for a fixed fiducial cosmology.

    Parameters
    ----------
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology
        Fiducial cosmology.
    k : array
        Output wavenumbers [h/Mpc].
    z : float
        Effective redshift.
    method : str
        How to compute :math:`\alpha(k)`:

        - ``'prim'``: :math:`\alpha = \sqrt{P_\phi^\mathrm{prim}(k) / P_\delta(k)}`.
        - ``'transfer'``: from the transfer function normalized in the matter-dominated
          era at :math:`z_\mathrm{norm}=10`; see eq. 2.3 of arXiv:1904.08859.
    engine : str
        cosmoprimo Boltzmann engine.

    Returns
    -------
    pk_dd : ndarray, shape (n_k,)
        Matter power spectrum.
    alpha : ndarray, shape (n_k,)
        Scale-dependent function linking primordial potential to density contrast.
    f : float
        Growth rate :math:`f = d\ln D / d\ln a`.

    References
    ----------
    Dalal et al. 2008  https://arxiv.org/abs/0710.4560
    Slosar et al. 2008  https://arxiv.org/abs/0805.3580
    Barreira 2020  https://arxiv.org/pdf/1904.08859.pdf
    """
    from cosmoprimo import constants
    k = np.asarray(k, dtype='f8')
    cosmo = _get_fiducial(fiducial).clone(engine=engine)
    fo = cosmo.get_fourier()

    # Prepend k=1e-4 for transfer-function normalization in the 'transfer' method.
    kin = np.concatenate([[1e-4], k])
    pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
    pk_dd_full = pk_interp(kin)
    # Primordial power spectrum P_prim(k) ~ k^(n_s - 1).
    pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(kin)

    if method == 'prim':
        # alpha = sqrt(P_phi_prim / P_delta);
        # P_phi_prim = (9/25) (2 pi^2 / k^3) P_prim / h^3   [Mpc^3 -> (h/Mpc)^3].
        pphi_prim = 9. / 25. * 2. * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
        alpha_full = 1. / np.sqrt(pk_dd_full / pphi_prim)
    else:
        # Transfer-function method, normalized at z=10 (matter-dominated). arXiv:1904.08859 eq. 2.3.
        tk = np.sqrt(pk_dd_full / pk_prim / kin / (pk_dd_full[0] / pk_prim[0] / kin[0]))
        znorm = 10.
        growth_ratio = float(cosmo.growth_factor(z) / cosmo.growth_factor(znorm) / (1. + znorm))
        c_kms = float(constants.c / 1e3)
        alpha_full = 3. * float(cosmo.Omega0_m) * 100.**2 / (2. * c_kms**2 * kin**2 * tk * growth_ratio)

    # Strip the normalization point so arrays align with k.
    pk_dd = pk_dd_full[1:]
    alpha = alpha_full[1:]
    sigma8 = float(fo.sigma8_z(z, of='delta_cb'))
    fsigma8 = float(fo.sigma8_z(z, of='theta_cb'))
    f = fsigma8 / sigma8
    return pk_dd, alpha, f


class PNGTracerSpectrum2Poles(Calculator):
    r"""
    Kaiser tracer power spectrum multipoles with local PNG scale-dependent bias.

    The scale-dependent bias is :math:`b_\mathrm{eff}(k) = b_1 + b_{f_\mathrm{NL}} \alpha(k)`,
    where :math:`b_{f_\mathrm{NL}} = b_\phi f_\mathrm{NL}` and :math:`b_\phi = 2 \delta_c (b_1 - p)`
    in the universal mass function approximation.

    The fiducial cosmology is fixed: :math:`\alpha(k)`, the matter power spectrum and the
    growth rate are computed once at compile time, and only the bias / :math:`f_\mathrm{NL}`
    parameters vary.

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
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fixed fiducial cosmology used to compute :math:`\alpha(k)` and :math:`P_{dd}`.
    engine : str, default='eisenstein_hu'
        cosmoprimo Boltzmann engine.
    method : str, default='prim'
        How to compute :math:`\alpha(k)` (``'prim'`` or ``'transfer'``); see :func:`_png_cosmo`.
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
    shotnoise : float, default=1e4
        Shot-noise scale [(h/Mpc)\ :sup:`3`]. The ``sn0`` parameter is in units of this.
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

    def __init__(self, k=None, ells=(0, 2), z=1., fiducial='DESI', engine='eisenstein_hu',
                 method='prim', mu=10, mode='b-p', tracers=None, shotnoise=1e4, params=None):
        # Nodes (Parameters) live in __init__.
        if mode not in ('b-p', 'bphi', 'bfnl'):
            raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        vc = type(self).propose_params(tracers=tracers, mode=mode)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)

    def __post_init__(self, k=None, ells=(0, 2), z=1., fiducial='DESI', engine='eisenstein_hu',
                      method='prim', mu=10, mode='b-p', tracers=None, shotnoise=1e4, params=None):
        # Non-node setup: precompute fixed-fiducial cosmo ingredients (numpy, once at compile).
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self._mode = str(mode)
        self._nbar = 1. / float(shotnoise)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        self._pk_dd, self._alpha, self._f = _png_cosmo(fiducial, self.k, float(z), str(method), str(engine))

    def __call__(self):
        k = self.k[:, None]            # (n_k, 1)
        mu = self._to_poles.mu          # (n_mu,)
        pk_dd = jnp.asarray(self._pk_dd)[:, None]   # (n_k, 1)
        alpha = jnp.asarray(self._alpha)[:, None]   # (n_k, 1)
        f = self._f

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
            fog_X = 1. / (1. + sigmas_X**2 * k**2 * mu**2 / 2.)
            fog_Y = 1. / (1. + sigmas_Y**2 * k**2 * mu**2 / 2.)
            pkmu = fog_X * fog_Y * (b_eff_X + f * mu**2) * (b_eff_Y + f * mu**2) * pk_dd
        else:
            if self._mode == 'b-p':
                bfnl_loc = 2. * _delta_c * (self.b1 - self.p) * self.fnl_loc
            elif self._mode == 'bphi':
                bfnl_loc = self.bphi * self.fnl_loc
            else:  # 'bfnl'
                bfnl_loc = self.bfnl_loc
            b_eff = self.b1 + bfnl_loc * alpha
            fog = 1. / (1. + self.sigmas**2 * k**2 * mu**2 / 2.)**2
            pkmu = fog * (b_eff + f * mu**2)**2 * pk_dd

        sn = np.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * self.sn0 / self._nbar
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

    The velocity bias reads :math:`v(k, \mu) = b_v f \mu H_0 / [(1+z) k]`.  The fiducial cosmology
    is fixed (see :class:`PNGTracerSpectrum2Poles`).

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to ``np.linspace(0.01, 0.2, 101)``.
    ells : tuple of int, default=(1, 3)
        Multipole orders (should be odd).
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fixed fiducial cosmology.
    engine : str, default='eisenstein_hu'
        cosmoprimo Boltzmann engine.
    method : str, default='prim'
        How to compute :math:`\alpha(k)`; see :func:`_png_cosmo`.
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    mode : str, default='b-p'
        PNG bias parameterization; same options as :class:`PNGTracerSpectrum2Poles`.
    """

    def __init__(self, k=None, ells=(1, 3), z=1., fiducial='DESI', engine='eisenstein_hu',
                 method='prim', mu=10, mode='b-p'):
        # Nodes (Parameters) live in __init__.
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

    def __post_init__(self, k=None, ells=(1, 3), z=1., fiducial='DESI', engine='eisenstein_hu',
                      method='prim', mu=10, mode='b-p'):
        # Non-node setup: precompute fixed-fiducial cosmo ingredients.
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self._mode = str(mode)
        self._z = float(z)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        self._pk_dd, self._alpha, self._f = _png_cosmo(fiducial, self.k, self._z, str(method), str(engine))

    def __call__(self):
        k = self.k[:, None]            # (n_k, 1)
        mu = self._to_poles.mu          # (n_mu,)
        pk_dd = jnp.asarray(self._pk_dd)[:, None]
        alpha = jnp.asarray(self._alpha)[:, None]
        f = self._f
        z = self._z

        if self._mode == 'b-p':
            bfnl_loc = 2. * _delta_c * (self.b1 - self.p) * self.fnl_loc
        elif self._mode == 'bphi':
            bfnl_loc = self.bphi * self.fnl_loc
        else:  # 'bfnl'
            bfnl_loc = self.bfnl_loc

        b_eff = self.b1 + bfnl_loc * alpha
        # FoG: density side (Lorentzian) x velocity side (sinc damping).
        fog = 1. / (1. + self.sigmas**2 * k**2 * mu**2 / 2.) * jnp.sinc(self.sigmau * k)
        # Velocity bias: -i bv f mu H0 / [(1+z) k]; the -i convention is dropped.
        vel_bias = self.bv * f * mu * 100. / (1. + z) / k
        pkmu = fog * (b_eff + f * mu**2) * vel_bias * pk_dd
        self.poles = self._to_poles(pkmu)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj
