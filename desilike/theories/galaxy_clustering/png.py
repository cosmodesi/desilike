"""
Primordial Non-Gaussianity (PNG) power spectrum multipoles.

Classes
-------
PNGSpectrum2Template
    Power spectrum template extended with the scale-dependent alpha(k) function for PNG.
PNGTracerSpectrum2Poles
    Kaiser tracer density power spectrum multipoles with local PNG scale-dependent bias.
PNGTracerVelocitySpectrum2Poles
    Kaiser tracer-velocity cross power spectrum multipoles with local PNG scale-dependent bias.
"""

import numpy as np
import jax.numpy as jnp

from ...base import Calculator, ExternalCalculator
from ...parameter import Parameter
from .bao import ProjectToMultipoles
from .full_shape import _interp_loglog
from .template import _get_fiducial, _ap_k_mu, _kw_pk
from ._multitracer import apply_tracers


_delta_c = 1.686  # linear collapse threshold


class PNGSpectrum2Template(ExternalCalculator):
    r"""
    Power spectrum template with scale-dependent :math:`\alpha(k)` for local PNG.

    Extends :class:`DirectSpectrum2Template` by computing the transfer-function-derived
    :math:`\alpha(k)`, which relates the primordial Bardeen potential to the late-time
    matter density contrast.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to ``np.logspace(-3, 1, 400)``.
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology for AP distance ratios.
    engine : str, default='camb'
        Boltzmann solver engine (passed to cosmoprimo).
    method : str, default='prim'
        How to compute :math:`\alpha(k)`.

        - ``'prim'``: :math:`\alpha = \sqrt{P_\phi^\mathrm{prim}(k) / P_\delta(k)}`.
        - ``'transfer'``: from the transfer function normalized in the matter-dominated era
          at :math:`z_\mathrm{norm}=10`; see eq. 2.3 of `arXiv:1904.08859 <https://arxiv.org/pdf/1904.08859.pdf>`_.

    with_now : str or False, default=False
        Engine for the no-wiggle power spectrum ('peakaverage', 'wallish2018').
        ``False`` sets ``pknow_dd = pk_dd``.

    Attributes exposed via ``tree_flatten``
    ----------------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth matter power spectra.
    f, f0, fk : float or ndarray
        Growth rate :math:`f = d\ln D / d\ln a`; ``f0`` is the :math:`k \to 0` limit,
        ``fk`` is the k-dependent version.
    qpar, qper : float
        AP distortion ratios (LOS and transverse).
    sigma8, fsigma8 : float
        Amplitude parameters.
    alpha : ndarray, shape (n_k,)
        Scale-dependent function linking primordial potential to density contrast.

    References
    ----------
    Dalal et al. 2008  https://arxiv.org/abs/0710.4560
    Slosar et al. 2008  https://arxiv.org/abs/0805.3580
    Barreira 2020  https://arxiv.org/pdf/1904.08859.pdf
    """

    def __init__(self, k=None, z=1., fiducial='DESI', engine='camb', method='prim', with_now=False):
        self.h = Parameter('h', value=0.6727, prior=dict(limits=[0.3, 1.0]),
                           ref=dict(dist='norm', loc=0.6727, scale=0.05), latex='h')
        self.omega_cdm = Parameter('omega_cdm', value=0.1200, prior=dict(limits=[0.05, 0.3]),
                                   ref=dict(dist='norm', loc=0.1200, scale=0.005),
                                   latex=r'\omega_\mathrm{cdm}')
        self.omega_b = Parameter('omega_b', value=0.02237, prior=dict(limits=[0.01, 0.04]),
                                  ref=dict(dist='norm', loc=0.02237, scale=0.001),
                                  latex=r'\omega_b')
        self.logA = Parameter('logA', value=3.044, prior=dict(limits=[2., 4.]),
                               ref=dict(dist='norm', loc=3.044, scale=0.1),
                               latex=r'\ln(10^{10}A_s)')
        self.n_s = Parameter('n_s', value=0.9649, prior=dict(limits=[0.7, 1.3]),
                              ref=dict(dist='norm', loc=0.9649, scale=0.01),
                              latex='n_s')

    def __post_init__(self, k=None, z=1., fiducial='DESI', engine='camb', method='prim', with_now=False):
        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)
        self._engine = str(engine)
        self._method = str(method)
        self._with_now = with_now

        from cosmoprimo import constants
        fid = _get_fiducial(fiducial)
        self._fiducial = fid
        self._DH_fid = float(constants.c / 1e3 / (100. * fid.efunc(self.z)))
        self._DM_fid = float(fid.comoving_angular_distance(self.z))

    def __call__(self):
        import cosmoprimo
        from cosmoprimo import PowerSpectrumBAOFilter, constants

        A_s = float(np.exp(float(self.logA.value)) * 1e-10)
        cosmo = cosmoprimo.Cosmology(h=float(self.h.value), omega_cdm=float(self.omega_cdm.value),
                                     omega_b=float(self.omega_b.value), A_s=A_s,
                                     n_s=float(self.n_s.value), engine=self._engine)
        fo = cosmo.get_fourier()

        # Prepend k=1e-4 for transfer function normalization in 'transfer' method.
        kin = np.concatenate([[1e-4], self.k])

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=self.z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=self.z)
        pk_dd_full = pk_interp(kin)
        # Primordial power spectrum P_prim(k) ~ k^(n_s - 1) from cosmoprimo
        pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(kin)

        if self._method == 'prim':
            # alpha = sqrt( P_phi_prim / P_delta )
            # P_phi_prim = (9/25) * (2*pi^2/k^3) * P_prim / h^3   [converting Mpc^3 → (h/Mpc)^3]
            pphi_prim = 9. / 25. * 2. * np.pi**2 / kin**3 * pk_prim / float(self.h.value)**3
            alpha_full = 1. / np.sqrt(pk_dd_full / pphi_prim)
        else:
            # Transfer-function method: alpha from Poisson equation, normalized at z=10
            # (matter-dominated era).  Ref: arXiv:1904.08859, eq. 2.3.
            tk = np.sqrt(pk_dd_full / pk_prim / kin / (pk_dd_full[0] / pk_prim[0] / kin[0]))
            znorm = 10.
            growth_ratio = float(cosmo.growth_factor(self.z) / cosmo.growth_factor(znorm) / (1. + znorm))
            c_kms = float(constants.c / 1e3)
            alpha_full = 3. * float(cosmo.Omega0_m) * 100.**2 / (2. * c_kms**2 * kin**2 * tk * growth_ratio)

        # Strip the normalization point so alpha and pk_dd align with self.k.
        self.alpha = alpha_full[1:]
        self.pk_dd = pk_dd_full[1:]

        if self._with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=self._with_now,
                                                cosmo=cosmo, cosmo_fid=self._fiducial)
            self.pknow_dd = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self.pknow_dd = self.pk_dd.copy()

        sigma8 = float(fo.sigma8_z(self.z, of='delta_cb'))
        fsigma8 = float(fo.sigma8_z(self.z, of='theta_cb'))
        self.sigma8 = sigma8
        self.fsigma8 = fsigma8
        self.f = fsigma8 / sigma8

        k0 = 1e-3
        self.f0 = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self.fk = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))

        DH = float(constants.c / 1e3 / (100. * cosmo.efunc(self.z)))
        DM = float(cosmo.comoving_angular_distance(self.z))
        self.qpar = DH / self._DH_fid
        self.qper = DM / self._DM_fid

    def ap_k_mu(self, k, mu):
        """AP distortion of (k, mu); returns (jac, kap, muap)."""
        return _ap_k_mu(k, mu, self.qpar, self.qper)

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk,
                 self.qpar, self.qper, self.sigma8, self.fsigma8, self.alpha],
                {'k': self.k, 'z': self.z})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk,
         obj.qpar, obj.qper, obj.sigma8, obj.fsigma8, obj.alpha) = children
        obj.k = aux['k']
        obj.z = aux['z']
        return obj


class PNGTracerSpectrum2Poles(Calculator):
    r"""
    Kaiser tracer power spectrum multipoles with local PNG scale-dependent bias.

    The scale-dependent bias is :math:`b_\mathrm{eff}(k) = b_1 + b_{f_\mathrm{NL}} \alpha(k)`,
    where :math:`b_{f_\mathrm{NL}} = b_\phi f_\mathrm{NL}` and :math:`b_\phi = 2 \delta_c (b_1 - p)`
    in the universal mass function approximation.

    For cross-spectra between two tracers :math:`X` and :math:`Y`, the power spectrum is
    :math:`\mathrm{FoG}_X \mathrm{FoG}_Y (b^\mathrm{eff}_X + f\mu^2)(b^\mathrm{eff}_Y + f\mu^2) P_{dd}`.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to ``np.linspace(0.01, 0.2, 101)``.
    pt : PNGSpectrum2Template, default=None
        PNG template module. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    tracers : str, (str, str), or None, default=None
        Tracer namespacing of the bias parameters:

        - ``None``: single auto-spectrum, unnamespaced parameters.
        - ``'LRG'``: auto-spectrum with parameters namespaced ``LRG.b1`` etc.
        - ``('LRG', 'QSO')``: cross-spectrum; bias parameters become per-tracer tuples
          (``LRG.b1``, ``QSO.b1``), shot noise is namespaced ``LRGxQSO.sn0``, and
          ``fnl_loc`` stays unnamespaced (shared cosmological parameter).

        For a multitracer analysis build one instance per spectrum, e.g.
        ``[PNGTracerSpectrum2Poles(tracers='LRG'), PNGTracerSpectrum2Poles(tracers=('LRG', 'QSO'))]``,
        and unify shared parameters with :func:`~desilike.base.share_params`.
    mode : str, default='b-p'
        Parameterization of the PNG bias:

        - ``'b-p'``: :math:`b_{f_\mathrm{NL}} = 2\delta_c(b_1 - p) f_\mathrm{NL}`;
          free params ``b1``, ``p``, ``fnl_loc``.
        - ``'bphi'``: :math:`b_{f_\mathrm{NL}} = b_\phi f_\mathrm{NL}`;
          free params ``b1``, ``bphi``, ``fnl_loc``.
        - ``'bfnl'``: :math:`b_{f_\mathrm{NL}}` directly;
          free params ``b1``, ``bfnl_loc``.
    shotnoise : float, default=1e4
        Shot-noise scale [(h/Mpc)\ :sup:`3`]. The ``sn0`` parameter is in units of this.
    """

    def __init__(self, k=None, pt=None, ells=(0, 2), tracers=None, mode='b-p', **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        if mode not in ('b-p', 'bphi', 'bfnl'):
            raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        self.b1 = Parameter('b1', value=2., prior=dict(limits=[0., 5.]),
                            ref=dict(dist='norm', loc=2., scale=0.2), latex='b_1')
        self.sigmas = Parameter('sigmas', value=0., prior=dict(limits=[0., 20.]),
                                ref=dict(dist='norm', loc=0., scale=1.), latex=r'\sigma_s')
        self.sn0 = Parameter('sn0', value=0., prior=None,
                             ref=dict(dist='norm', loc=0., scale=1.), latex='s_{n,0}')
        if mode == 'b-p':
            self.fnl_loc = Parameter('fnl_loc', value=0., prior=dict(dist='norm', loc=0., scale=50.),
                                     ref=dict(dist='norm', loc=0., scale=5.), latex=r'f_\mathrm{NL}')
            self.p = Parameter('p', value=1., prior=None, ref=dict(dist='norm', loc=1., scale=0.1), latex='p')
        elif mode == 'bphi':
            self.fnl_loc = Parameter('fnl_loc', value=0., prior=dict(dist='norm', loc=0., scale=50.),
                                     ref=dict(dist='norm', loc=0., scale=5.), latex=r'f_\mathrm{NL}')
            self.bphi = Parameter('bphi', value=1., prior=None, ref=dict(dist='norm', loc=1., scale=0.5), latex=r'b_\phi')
        else:
            self.bfnl_loc = Parameter('bfnl_loc', value=0., prior=dict(dist='norm', loc=0., scale=100.),
                                      ref=dict(dist='norm', loc=0., scale=10.), latex=r'b_{f_\mathrm{NL}}')
        # fnl_loc is a shared cosmological parameter (never namespaced); sn0 is stochastic.
        apply_tracers(self, tracers, stochastic=('sn0',), shared=('fnl_loc',), cross=True)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        if pt is None:
            pt = PNGSpectrum2Template()
        self.pt = pt
        self.pt.update(k=np.geomspace(min(1e-4, self.k[0] / 2.), max(1., self.k[-1] * 2.), 500))

    def __post_init__(self, k=None, pt=None, ells=(0, 2), mu=10, mode='b-p', shotnoise=1e4, **kwargs):
        # Non-node setup only (``tracers`` consumed by __init__).
        self.ells = tuple(ells)
        self._mode = str(mode)
        self._nbar = 1. / float(shotnoise)
        self._to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)

    def __call__(self):
        jac, kap, muap = self.pt.ap_k_mu(self.k[:, None], self._to_poles.mu)
        pk_dd_ap = _interp_loglog(kap, self.pt.k, self.pt.pk_dd)
        alpha_ap = _interp_loglog(kap, self.pt.k, self.pt.alpha)
        f = self.pt.f

        cross = isinstance(self.b1, tuple)

        if cross:
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
            b_eff_X = b1_X + bfnl_loc_X * alpha_ap
            b_eff_Y = b1_Y + bfnl_loc_Y * alpha_ap
            fog_X = 1. / (1. + sigmas_X**2 * kap**2 * muap**2 / 2.)
            fog_Y = 1. / (1. + sigmas_Y**2 * kap**2 * muap**2 / 2.)
            pkmu = jac * fog_X * fog_Y * (b_eff_X + f * muap**2) * (b_eff_Y + f * muap**2) * pk_dd_ap
        else:
            if self._mode == 'b-p':
                bfnl_loc = 2. * _delta_c * (self.b1 - self.p) * self.fnl_loc
            elif self._mode == 'bphi':
                bfnl_loc = self.bphi * self.fnl_loc
            else:  # 'bfnl'
                bfnl_loc = self.bfnl_loc
            b_eff = self.b1 + bfnl_loc * alpha_ap
            fog = 1. / (1. + self.sigmas**2 * kap**2 * muap**2 / 2.)**2
            pkmu = jac * fog * (b_eff + f * muap**2)**2 * pk_dd_ap

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

    The velocity bias reads :math:`v(k, \mu) = b_v f \mu H_0 / [(1+z) k]`.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to ``np.linspace(0.01, 0.2, 101)``.
    pt : PNGSpectrum2Template, default=None
        PNG template module. A default instance is created if None.
    ells : tuple of int, default=(1, 3)
        Multipole orders (should be odd).
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    mode : str, default='b-p'
        PNG bias parameterization; same options as :class:`PNGTracerSpectrum2Poles`.
    """

    def __init__(self, k=None, pt=None, ells=(1, 3), mode='b-p', **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        self.b1 = Parameter('b1', value=2., prior=dict(limits=[0., 5.]),
                            ref=dict(dist='norm', loc=2., scale=0.2), latex='b_1')
        self.bv = Parameter('bv', value=1., prior=None,
                            ref=dict(dist='norm', loc=1., scale=0.1), latex='b_v')
        self.sigmas = Parameter('sigmas', value=0., prior=dict(limits=[0., 20.]),
                                ref=dict(dist='norm', loc=0., scale=1.), latex=r'\sigma_s')
        self.sigmau = Parameter('sigmau', value=0., prior=dict(limits=[0., 20.]),
                                ref=dict(dist='norm', loc=0., scale=1.), latex=r'\sigma_u')
        if mode == 'b-p':
            self.fnl_loc = Parameter('fnl_loc', value=0., prior=dict(dist='norm', loc=0., scale=50.),
                                     ref=dict(dist='norm', loc=0., scale=5.), latex=r'f_\mathrm{NL}')
            self.p = Parameter('p', value=1., prior=None,
                               ref=dict(dist='norm', loc=1., scale=0.1), latex='p')
        elif mode == 'bphi':
            self.fnl_loc = Parameter('fnl_loc', value=0., prior=dict(dist='norm', loc=0., scale=50.),
                                     ref=dict(dist='norm', loc=0., scale=5.), latex=r'f_\mathrm{NL}')
            self.bphi = Parameter('bphi', value=1., prior=None,
                                  ref=dict(dist='norm', loc=1., scale=0.5), latex=r'b_\phi')
        elif mode == 'bfnl':
            self.bfnl_loc = Parameter('bfnl_loc', value=0., prior=dict(dist='norm', loc=0., scale=100.),
                                      ref=dict(dist='norm', loc=0., scale=10.), latex=r'b_{f_\mathrm{NL}}')
        else:
            raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        if pt is None:
            pt = PNGSpectrum2Template()
        self.pt = pt
        self.pt.update(k=np.geomspace(min(1e-4, self.k[0] / 2.), max(1., self.k[-1] * 2.), 500))

    def __post_init__(self, k=None, pt=None, ells=(1, 3), mu=10, mode='b-p'):
        # Non-node setup only.
        self.ells = tuple(ells)
        self._mode = str(mode)
        self._to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)

    def __call__(self):
        jac, kap, muap = self.pt.ap_k_mu(self.k[:, None], self._to_poles.mu)
        pk_dd_ap = _interp_loglog(kap, self.pt.k, self.pt.pk_dd)
        alpha_ap = _interp_loglog(kap, self.pt.k, self.pt.alpha)
        f = self.pt.f
        z = self.pt.z

        if self._mode == 'b-p':
            bfnl_loc = 2. * _delta_c * (self.b1 - self.p) * self.fnl_loc
        elif self._mode == 'bphi':
            bfnl_loc = self.bphi * self.fnl_loc
        else:  # 'bfnl'
            bfnl_loc = self.bfnl_loc

        b_eff = self.b1 + bfnl_loc * alpha_ap
        # FoG: density side (Lorentzian) × velocity side (sinc damping)
        fog = 1. / (1. + self.sigmas**2 * kap**2 * muap**2 / 2.) * jnp.sinc(self.sigmau * kap)
        # Velocity bias: -i * bv * f * mu * H0 / [(1+z) * k]; we drop the -i convention.
        vel_bias = self.bv * f * muap * 100. / (1. + z) / kap
        pkmu = jac * fog * (b_eff + f * muap**2) * vel_bias * pk_dd_ap
        self.poles = self._to_poles(pkmu)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj
