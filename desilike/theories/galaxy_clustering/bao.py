"""
BAO multipole models for galaxy clustering.

Classes
-------
DampedBAOWigglesPTSpectrum2Poles
    Power spectrum multipoles with Gaussian-damped BAO wiggles (Chen 2024 / Beutler 2017).
ResummedBAOWigglesPTSpectrum2Poles
    Power spectrum multipoles with EFT-resummed BAO wiggles (Senatore & Zaldarriaga 2015).
DampedBAOWigglesTracerSpectrum2Poles
    DampedBAOWigglesPTSpectrum2Poles with additive polynomial broadband.
ResummedBAOWigglesTracerSpectrum2Poles
    ResummedBAOWigglesPTSpectrum2Poles with additive polynomial broadband.
SpectrumToCorrelation
    FFTLog Hankel transform helper: pk multipoles -> xi multipoles.
DampedBAOWigglesPTCorrelation2Poles
    Correlation function multipoles from DampedBAOWigglesPTSpectrum2Poles via FFTLog.
ResummedBAOWigglesPTCorrelation2Poles
    Correlation function multipoles from ResummedBAOWigglesPTSpectrum2Poles via FFTLog.
DampedBAOWigglesTracerCorrelation2Poles
    DampedBAOWigglesPTCorrelation2Poles with additive s-space polynomial broadband.
ResummedBAOWigglesTracerCorrelation2Poles
    ResummedBAOWigglesPTCorrelation2Poles with additive s-space polynomial broadband.
"""

import numpy as np
import jax.numpy as jnp
import interpax
from scipy import special, integrate

from ...base import Calculator, ExternalCalculator
from ...parameter import Parameter
from .template import BAOSpectrum2Template
from ._multitracer import apply_tracers


# ── interpolation ─────────────────────────────────────────────────────────────

def _interp_loglog(k_query, k_knots, pk_knots):
    """Cubic spline interpolation of pk in log10(k)-pk space.

    Accepts any shape for k_query; k_knots and pk_knots must be 1-D.
    Uses constant extrapolation beyond the knot range.
    """
    shape = jnp.shape(k_query)
    flat = jnp.ravel(k_query)
    result = interpax.interp1d(jnp.log10(flat), jnp.log10(k_knots), pk_knots,
                                method='cubic', extrap=True)
    return jnp.reshape(result, shape)


# ── multipole projection ──────────────────────────────────────────────────────

class ProjectToMultipoles:
    """Project P(k, mu) -> P_ell(k) via Gauss-Legendre quadrature on mu in [0, 1].

    Uses the symmetry P(k, mu) = P(k, -mu) to integrate only over [0, 1].
    """

    def __init__(self, mu=20, ells=(0, 2, 4)):
        # Symmetric GL quadrature: 2*mu nodes on [-1,1], keep upper half [0,1].
        x_full, w_full = np.polynomial.legendre.leggauss(2 * mu)
        x = x_full[mu:]
        w = (w_full[mu:] + w_full[mu - 1::-1]) / 2.
        self.mu = x  # shape (mu,)
        # wmu[i_ell, i_mu] = (2*ell+1) * L_ell(mu_j) * w_j
        self.wmu = np.array([(2 * ell + 1) * special.legendre(ell)(x) * w for ell in ells])

    def __call__(self, pkmu):
        """Sum over mu: pkmu (n_k, n_mu) -> spectrum (n_ells, n_k)."""
        return jnp.sum(pkmu * self.wmu[:, None, :], axis=-1)


# ── damped BAO multipoles ─────────────────────────────────────────────────────

class DampedBAOWigglesPTSpectrum2Poles(Calculator):
    r"""
    BAO power spectrum multipoles with Gaussian-damped wiggles.

    Implements the model of Chen 2024 (arXiv:2402.14070). The BAO wiggles
    are damped by a Gaussian envelope parameterized by ``sigmapar`` and ``sigmaper``.
    Finger-of-God damping is applied via the ``sigmas`` parameter.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to np.linspace(0.01, 0.2, 101).
    template : BAOSpectrum2Template, default=None
        Power spectrum template. A default ``BAOSpectrum2Template()`` is created
        if None is passed; the template ``k`` range is extended automatically to provide
        margin for AP distortion.
    ells : tuple of int, default=(0, 2)
        Multipole orders to compute.
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    mode : str, default=''
        Reconstruction mode: '' (pre-recon), 'recsym', or 'reciso'.
    smoothing_radius : float, default=15.
        Reconstruction smoothing radius [Mpc/h].
    model : str, default='standard'
        Damping model variant:

        - 'standard'    : Chen 2024 — damping applied in AP-distorted space.
        - 'fix-damping' : damping computed in undistorted (k, mu) space.
        - 'move-all'    : AP distortion applied to both smooth and wiggle components.
        - 'fog-damping' : FoG applied to the full power (Beutler 2017 convention).

    Attributes set by ``__call__``
    --------------------------------
    spectrum : ndarray, shape (n_ells, n_k)
        Power spectrum multipoles.

    References
    ----------
    Chen et al. 2024  https://arxiv.org/abs/2402.14070
    Beutler et al. 2017  https://arxiv.org/abs/1607.03149
    """

    def __init__(self, k=None, template=None, tracers=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        # ``tracers`` (a single tracer name) namespaces each parameter (e.g. 'LRG.b1').
        self.b1 = Parameter('b1', value=1., prior=dict(limits=[0., 4.]),
                            ref=dict(dist='norm', loc=1., scale=0.1), latex='b_1')
        self.dbeta = Parameter('dbeta', value=1., prior=dict(limits=[0., 3.]),
                               ref=dict(dist='norm', loc=1., scale=0.05), latex=r'\delta\beta')
        self.sigmas = Parameter('sigmas', value=0., prior=dict(limits=[-1., 10.]),
                                ref=dict(dist='norm', loc=0., scale=1.), latex=r'\Sigma_s')
        self.sigmapar = Parameter('sigmapar', value=9., prior=dict(limits=[0., 25.]),
                                  ref=dict(dist='norm', loc=9., scale=1.), latex=r'\Sigma_\parallel')
        self.sigmaper = Parameter('sigmaper', value=6., prior=dict(limits=[0., 20.]),
                                  ref=dict(dist='norm', loc=6., scale=1.), latex=r'\Sigma_\perp')
        apply_tracers(self, tracers)  # namespacing only (no cross)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        k = np.asarray(k, dtype='f8')
        if template is None:
            template = BAOSpectrum2Template()
        self.template = template  # Calculator dep
        # Extend template k range to cover AP-distorted queries with margin.
        k_min = min(1e-4, float(np.min(k)) / 2.)
        k_max = max(1., float(np.max(k)) * 2.)
        _, tmpl_kw = self.template._init
        update_kw = {'k': np.geomspace(k_min, k_max, 2000)}
        if not tmpl_kw.get('with_now'):
            update_kw['with_now'] = 'peakaverage'
        self.template.update(**update_kw)
        # Fix damping params when wiggles are suppressed.
        if self.template._init[1].get('only_now', False):
            self.sigmapar.update(fixed=True)
            self.sigmaper.update(fixed=True)

    def __post_init__(self, k=None, template=None, ells=(0, 2), mu=10,
                      mode='', smoothing_radius=15., model='standard', **kwargs):
        # Non-node setup only.
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self.mode = str(mode)
        if self.mode not in ('', 'recsym', 'reciso'):
            raise ValueError(f"mode must be '', 'recsym', or 'reciso'; got {mode!r}")
        self.smoothing_radius = float(smoothing_radius)
        self.model = str(model)
        self._to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self._mu = self._to_poles.mu  # shape (n_mu,), GL nodes in [0, 1]

    def __call__(self):
        template = self.template
        k = self.k[:, None]   # (n_k, 1)  — broadcast with mu
        mu = self._mu          # (n_mu,)

        f = self.dbeta * template.f

        # AP-distorted coordinates; shapes (n_k, n_mu).
        jac, kap, muap = template.ap_k_mu(k, mu)
        pknowap = _interp_loglog(kap, template.k, template.pknow_dd)
        pkap = _interp_loglog(kap, template.k, template.pk_dd)

        if self.model == 'standard':
            # Chen 2024: damping in AP-distorted space; smooth bias in undistorted space.
            pkwap = pkap - pknowap
            sigma_nl2ap = kap ** 2 * (self.sigmapar ** 2 * muap ** 2 + self.sigmaper ** 2 * (1. - muap ** 2))
            sk = 0.
            if self.mode == 'reciso':
                sk = jnp.exp(-0.5 * (k * self.smoothing_radius) ** 2)
            Cap = (self.b1 + f * muap ** 2 * (1. - sk)) ** 2 * jnp.exp(-0.5 * sigma_nl2ap)
            fog = 1. / (1. + 0.5 * (self.sigmas * k * mu) ** 2) ** 2
            B = (self.b1 + f * mu ** 2 * (1. - sk)) ** 2 * fog
            pknow = _interp_loglog(k, template.k, template.pknow_dd)
            pkmu = B * pknow + Cap * pkwap

        else:
            # Alternative damping conventions.
            if 'fix-damping' in self.model:
                kd, mud = k, mu
            else:
                kd, mud = kap, muap
            sigma_nl2 = kd ** 2 * (self.sigmapar ** 2 * mud ** 2 + self.sigmaper ** 2 * (1. - mud ** 2))
            damped_wiggles = (pkap - pknowap) / pknowap * jnp.exp(-0.5 * sigma_nl2)
            if 'move-all' in self.model:
                ks, mus = kap, muap
            else:
                ks, mus = k, mu
            pknow = _interp_loglog(ks, template.k, template.pknow_dd)
            fog = 1. / (1. + 0.5 * (self.sigmas * ks * mus) ** 2) ** 2
            sk = 0.
            if self.mode == 'reciso':
                sk = jnp.exp(-0.5 * (ks * self.smoothing_radius) ** 2)
            pksmooth = (self.b1 + f * mus ** 2 * (1. - sk)) ** 2 * pknow
            if 'fog-damping' in self.model:  # Beutler 2017
                pkmu = pksmooth * fog * (1. + damped_wiggles)
            else:
                pkmu = pksmooth * (fog + damped_wiggles)

        self.poles = self._to_poles(pkmu)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


# ── resummed BAO multipoles ───────────────────────────────────────────────────

class ResummedBAOWigglesPTSpectrum2Poles(Calculator):
    r"""
    BAO power spectrum multipoles with EFT-resummed wiggles.

    Implements the resummation scheme of Senatore & Zaldarriaga 2015 (arXiv:1404.5616).
    The BAO damping scale is derived self-consistently from the no-wiggle power spectrum
    rather than treated as a free parameter.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to np.linspace(0.01, 0.2, 101).
    template : BAOSpectrum2Template, default=None
        Power spectrum template. A default ``BAOSpectrum2Template()`` is created
        if None is passed.
    ells : tuple of int, default=(0, 2)
        Multipole orders to compute.
    mu : int, default=10
        Number of Gauss-Legendre mu-bins in [0, 1].
    mode : str, default=''
        Reconstruction mode: '' (pre-recon), 'recsym', or 'reciso'.
    smoothing_radius : float, default=15.
        Reconstruction smoothing radius [Mpc/h].
    model : str, default='standard'
        Model variant:

        - 'standard'    : smooth in undistorted space, wiggles at AP-distorted coordinates.
        - 'move-all'    : AP distortion applied to both smooth and wiggle components.
        - 'fog-damping' : FoG applied to the full power (Beutler 2017 convention).
    shotnoise : float, default=0.
        Shot-noise contribution to the effective damping scale.

    Attributes set by ``__call__``
    --------------------------------
    spectrum : ndarray, shape (n_ells, n_k)
        Power spectrum multipoles.

    References
    ----------
    Senatore & Zaldarriaga 2015  https://arxiv.org/abs/1404.5616
    Beutler et al. 2017  https://arxiv.org/abs/1607.03149
    """

    def __init__(self, k=None, template=None, tracers=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        # ``tracers`` (a single tracer name) namespaces each parameter via apply_tracers.
        self.b1 = Parameter('b1', value=1., prior=dict(limits=[0., 4.]),
                            ref=dict(dist='norm', loc=1., scale=0.1), latex='b_1')
        self.dbeta = Parameter('dbeta', value=1., prior=dict(limits=[0., 3.]),
                               ref=dict(dist='norm', loc=1., scale=0.05), latex=r'\delta\beta')
        self.sigmas = Parameter('sigmas', value=0., prior=dict(limits=[-1., 10.]),
                                ref=dict(dist='norm', loc=0., scale=1.), latex=r'\Sigma_s')
        # d scales the growth factor relative to fiducial (d=1: fiducial).
        self.d = Parameter('d', value=1., prior=dict(limits=[0., 3.]),
                           ref=dict(dist='norm', loc=1., scale=0.05), latex='d')
        apply_tracers(self, tracers)  # namespacing only (no cross)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        k = np.asarray(k, dtype='f8')
        if template is None:
            template = BAOSpectrum2Template()
        self.template = template
        k_min = min(1e-4, float(np.min(k)) / 2.)
        k_max = max(1., float(np.max(k)) * 2.)
        _, tmpl_kw = self.template._init
        update_kw = {'k': np.geomspace(k_min, k_max, 2000)}
        if not tmpl_kw.get('with_now'):
            update_kw['with_now'] = 'peakaverage'
        self.template.update(**update_kw)
        # Fix d when only_now: no wiggles means no BAO scale to constrain d.
        if self.template._init[1].get('only_now', False):
            self.d.update(fixed=True)

    def __post_init__(self, k=None, template=None, ells=(0, 2), mu=10,
                      mode='', smoothing_radius=15., model='standard', shotnoise=0., **kwargs):
        # Non-node setup only.
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self.mode = str(mode)
        if self.mode not in ('', 'recsym', 'reciso'):
            raise ValueError(f"mode must be '', 'recsym', or 'reciso'; got {mode!r}")
        self.smoothing_radius = float(smoothing_radius)
        self.model = str(model)
        self._shotnoise = float(shotnoise)
        self._to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self._mu = self._to_poles.mu

    def _sigma_base(self):
        """Compute damping-scale integrals from the fiducial no-wiggle PK.

        Uses scipy (numpy ops); runs at Python execution time (once at JIT trace).
        Returns numpy scalars (sigma_nl2, sigma_dd2, sigma_x2_or_None, sigma_sn2).
        """
        k = self.template.k            # numpy, set in template.__post_init__
        pknow = self.template._pknow_dd_fid  # numpy, set in template.__post_init__
        q = self.template._fiducial.rs_drag  # float
        j0 = special.jn(0, q * k)
        if self.mode:
            sk = np.exp(-0.5 * (k * self.smoothing_radius) ** 2)
        else:
            sk = np.zeros_like(k)
        skc = 1. - sk
        norm = 1. / (3. * np.pi ** 2)
        sigma_nl2 = norm * float(integrate.simpson((1. - j0) * pknow, x=k))
        sigma_dd2 = norm * float(integrate.simpson((1. - j0) * skc ** 2 * pknow, x=k))
        sigma_x2 = norm * float(integrate.simpson((1. - j0) * skc * pknow, x=k)) if self.mode == 'reciso' else None
        sigma_sn2 = 1. / (self.smoothing_radius * 6. * np.pi ** 1.5)
        return sigma_nl2, sigma_dd2, sigma_x2, sigma_sn2

    def __call__(self):
        template = self.template
        k = self.k[:, None]
        mu = self._mu

        f = self.dbeta * template.f

        jac, kap, muap = template.ap_k_mu(k, mu)
        pknow_ap = _interp_loglog(kap, template.k, template.pknow_dd)
        pk_ap = _interp_loglog(kap, template.k, template.pk_dd)

        # Damping scales: numpy at trace time, become JIT constants.
        sigma_nl2, sigma_dd2_base, sigma_x2, sigma_sn2 = self._sigma_base()
        sigma_dd2 = sigma_dd2_base + self._shotnoise * sigma_sn2 / self.b1 ** 2

        # Resummed BAO wiggles evaluated at AP-distorted (kap, muap).
        d2 = self.d ** 2
        ksq_ap = (1. + f * (f + 2.) * muap ** 2) * kap ** 2
        sk_ap = jnp.exp(-0.5 * (kap * self.smoothing_radius) ** 2) if self.mode else 0.
        skc_ap = 1. - sk_ap

        if self.mode == 'reciso':
            sigma_ds2 = (1. + f * muap ** 2) * sigma_dd2 + f * (1. + f) * muap ** 2 * sigma_x2
            sigma_ss2 = sigma_dd2 + f ** 2 * muap ** 2 * sigma_nl2 + 2. * f * muap ** 2 * sigma_x2
            resummed_w = (
                (self.b1 + f * muap ** 2 * skc_ap - sk_ap) ** 2 * jnp.exp(-0.5 * ksq_ap * d2 * sigma_dd2)
                + 2. * (self.b1 + f * muap ** 2 * skc_ap - sk_ap) * (1. + f * muap ** 2) * sk_ap
                  * jnp.exp(-0.5 * ksq_ap * d2 * sigma_ds2)
                + (1. + f * muap ** 2) ** 2 * sk_ap ** 2 * jnp.exp(-0.5 * ksq_ap * d2 * sigma_ss2)
            )
        else:
            resummed_w = (self.b1 + f * muap ** 2) ** 2 * jnp.exp(-0.5 * ksq_ap * d2 * sigma_dd2)

        # Normalized wiggle component: resummed_w * (pk_dd - pknow_dd) / pknow_dd at kap.
        damped_wiggles = resummed_w * (pk_ap - pknow_ap) / pknow_ap

        # Smooth term in undistorted (or AP-distorted for 'move-all') space.
        if 'move-all' in self.model:
            ks, mus = kap, muap
        else:
            ks, mus = k, mu
        pknow = _interp_loglog(ks, template.k, template.pknow_dd)
        fog = 1. / (1. + 0.5 * (self.sigmas * ks * mus) ** 2) ** 2
        sk = 0.
        if self.mode == 'reciso':
            sk = jnp.exp(-0.5 * (ks * self.smoothing_radius) ** 2)
        pksmooth = (self.b1 + f * mus ** 2 * (1. - sk)) ** 2 * pknow

        if 'fog-damping' in self.model:
            pkmu = pksmooth * fog * (1. + damped_wiggles)
        else:
            pkmu = pksmooth * (fog + damped_wiggles)

        self.poles = self._to_poles(pkmu)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


# ── tracer base (broadband) ───────────────────────────────────────────────────

class _BAOWigglesTracerSpectrum2Poles(Calculator):
    """Base for BAO tracer theories: wraps a bare BAO theory and adds polynomial broadband.

    The broadband is a sum of power-law terms per multipole:
      P_tracer_ell(k) = P_bare_ell(k) + sum_pow al{ell}_{pow} * (k / kp)^pow
    where kp = 2*pi / rs_drag_fid is the BAO scale pivot.

    Subclasses set ``_default_pt_cls`` to the bare theory class.
    """

    _default_pt_cls = None  # set by DampedBAOWigglesTracerSpectrum2Poles etc.

    def __init__(self, k=None, pt=None, ells=(0, 2), broadband_pows=(-3, -2, -1, 0, 1), tracers=None):
        # Broadband Parameters created here (Option B) so they appear in __dict__.
        # ``tracers`` (a single tracer name) namespaces each parameter via apply_tracers.
        _ells = tuple(ells)
        _pows = tuple(broadband_pows)
        self.bb_params = []
        for ell in _ells:
            for pow in _pows:
                self.bb_params.append(
                    Parameter(f'al{ell}_{pow}', value=0., prior=None,
                              ref=dict(dist='norm', loc=0., scale=1.),
                              latex=f'a_{{{ell},{pow}}}')
                )
        self._bb_ell_pow = [(ell, pow) for ell in _ells for pow in _pows]
        apply_tracers(self, tracers)  # namespaces bb_params (no cross)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = _ells
        if pt is None:
            pt = self._default_pt_cls(tracers=tracers)  # bias namespaced to the same tracer
        self.pt = pt  # Calculator dep
        self.pt.update(k=self.k, ells=self.ells)

    def __call__(self):
        # kp from the fiducial rs_drag (numpy scalar, constant under JIT).
        kp = 2. * np.pi / self.pt.template._fiducial.rs_drag
        k = self.k  # (n_k,)
        ell_to_idx = {ell: i for i, ell in enumerate(self.ells)}
        broadband = jnp.zeros((len(self.ells), len(k)))
        for (ell, pow), param in zip(self._bb_ell_pow, self.bb_params):
            ill = ell_to_idx[ell]
            broadband = broadband.at[ill].add(param * (k / kp) ** pow)
        self.poles = self.pt.poles + broadband
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class DampedBAOWigglesTracerSpectrum2Poles(_BAOWigglesTracerSpectrum2Poles):
    r"""
    DampedBAOWigglesPTSpectrum2Poles with additive polynomial broadband.

    Broadband parameters ``al{ell}_{pow}`` are created automatically for each
    (ell, pow) combination. The pivot wavenumber is kp = 2*pi / rs_drag_fid.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    pt : DampedBAOWigglesPTSpectrum2Poles, default=None
        Bare theory. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    broadband_pows : tuple of int, default=(-3, -2, -1, 0, 1)
        Powers of k/kp to include in the broadband.
    """
    _default_pt_cls = DampedBAOWigglesPTSpectrum2Poles


class ResummedBAOWigglesTracerSpectrum2Poles(_BAOWigglesTracerSpectrum2Poles):
    r"""
    ResummedBAOWigglesPTSpectrum2Poles with additive polynomial broadband.

    Broadband parameters ``al{ell}_{pow}`` are created automatically for each
    (ell, pow) combination. The pivot wavenumber is kp = 2*pi / rs_drag_fid.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    pt : ResummedBAOWigglesPTSpectrum2Poles, default=None
        Bare theory. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    broadband_pows : tuple of int, default=(-3, -2, -1, 0, 1)
        Powers of k/kp to include in the broadband.
    """
    _default_pt_cls = ResummedBAOWigglesPTSpectrum2Poles


# ── spectrum -> correlation transformer ──────────────────────────────────────

class SpectrumToCorrelation:
    r"""FFTLog Hankel transform: pk multipoles -> xi multipoles.

    Parameters
    ----------
    s : array
        Output separations [Mpc/h].
    ells : tuple of int
        Multipole orders.
    kin : array
        Input wavenumbers [h/Mpc] at which the power spectrum is evaluated.
    """

    def __init__(self, s, ells, kin):
        from cosmoprimo import PowerToCorrelation
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        self.kin = np.asarray(kin, dtype='f8')
        k_fftlog = np.logspace(-4., 3., 2048)
        mask_high = k_fftlog > kin[-1]
        self.k_mid = k_fftlog[~mask_high]
        self.logk_high = np.log10(k_fftlog[mask_high] / kin[-1])
        self.damp_high = np.exp(-(k_fftlog[mask_high] / kin[-1] - 1.) ** 2 / 200.)
        self.fftlog = PowerToCorrelation(k_fftlog, ell=self.ells, q=0, lowring=True)

    def __call__(self, poles):
        r"""Transform pk multipoles to xi multipoles.

        Parameters
        ----------
        poles : ndarray, shape (n_ells, n_kin)
            Power spectrum multipoles at ``self.kin``.

        Returns
        -------
        ndarray, shape (n_ells, n_s)
        """
        tmp = []
        for pole in poles:
            slope_high = (pole[-1] - pole[-2]) / np.log10(self.kin[-1] / self.kin[-2])
            interp_mid = jnp.interp(np.log10(self.k_mid), np.log10(self.kin), pole)
            extrap_high = (pole[-1] + slope_high * self.logk_high) * self.damp_high
            tmp.append(jnp.concatenate([interp_mid, extrap_high]))
        s_arr, corr_arr = self.fftlog(jnp.vstack(tmp))
        return jnp.stack([jnp.interp(self.s, ss, cc) for ss, cc in zip(s_arr, corr_arr)])


# ── bare correlation function multipoles ─────────────────────────────────────

class _BAOWigglesPTCorrelation2Poles(ExternalCalculator):
    """Base for BAO correlation function theories (FFTLog from spectrum to xi multipoles).

    Subclasses set ``_default_pt_cls`` to the spectrum theory class.
    """

    _default_pt_cls = None

    def __init__(self, s=None, pt=None, ells=(0, 2)):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = self._default_pt_cls()
        self.pt = pt
        self.pt.update(k=np.geomspace(1e-4, 0.6, 300), ells=self.ells)

    def __post_init__(self, s=None, pt=None, ells=(0, 2)):
        # Non-node setup only.
        self._to_correlation = SpectrumToCorrelation(s=self.s, ells=self.ells, kin=np.geomspace(1e-4, 0.6, 300))

    def __call__(self):
        self.poles = self._to_correlation(self.pt.poles)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class DampedBAOWigglesPTCorrelation2Poles(_BAOWigglesPTCorrelation2Poles):
    r"""
    BAO correlation function multipoles with Gaussian-damped wiggles.

    Applies FFTLog Hankel transform to ``DampedBAOWigglesPTSpectrum2Poles`` output.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h]. Defaults to np.linspace(20., 200., 181).
    pt : DampedBAOWigglesPTSpectrum2Poles, default=None
        Bare spectrum theory. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    """
    _default_pt_cls = DampedBAOWigglesPTSpectrum2Poles


class ResummedBAOWigglesPTCorrelation2Poles(_BAOWigglesPTCorrelation2Poles):
    r"""
    BAO correlation function multipoles with EFT-resummed wiggles.

    Applies FFTLog Hankel transform to ``ResummedBAOWigglesPTSpectrum2Poles`` output.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h]. Defaults to np.linspace(20., 200., 181).
    pt : ResummedBAOWigglesPTSpectrum2Poles, default=None
        Bare spectrum theory. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    """
    _default_pt_cls = ResummedBAOWigglesPTSpectrum2Poles


# ── tracer correlation function multipoles ────────────────────────────────────

class _BAOWigglesTracerCorrelation2Poles(ExternalCalculator):
    r"""Base for BAO tracer correlation function theories: FFTLog + s-space polynomial broadband.

    The broadband is:
      xi_tracer_ell(s) = xi_bare_ell(s) + sum_pow al{ell}_{pow} * (s / sp)^pow
    where sp = 2*pi / 0.02 ~ 314 Mpc/h.

    Subclasses set ``_default_pt_cls`` to the bare spectrum theory class.
    """

    _default_pt_cls = None

    def __init__(self, s=None, pt=None, ells=(0, 2), broadband_pows=(-2, -1, 0, 1, 2), tracers=None):
        _ells = tuple(ells)
        _pows = tuple(broadband_pows)
        self.bb_params = []
        for ell in _ells:
            for pow in _pows:
                self.bb_params.append(
                    Parameter(f'al{ell}_{pow}', value=0., prior=None,
                              ref=dict(dist='norm', loc=0., scale=1.),
                              latex=f'a_{{{ell},{pow}}}')
                )
        self._bb_ell_pow = [(ell, pow) for ell in _ells for pow in _pows]
        apply_tracers(self, tracers)  # namespaces bb_params (no cross)
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = _ells
        if pt is None:
            pt = self._default_pt_cls(tracers=tracers)  # bias namespaced to the same tracer
        self.pt = pt
        self.pt.update(k=np.geomspace(1e-4, 0.6, 300), ells=self.ells)

    def __post_init__(self, s=None, pt=None, ells=(0, 2), broadband_pows=(-2, -1, 0, 1, 2), tracers=None):
        # Non-node setup only.
        self._to_correlation = SpectrumToCorrelation(s=self.s, ells=self.ells, kin=np.geomspace(1e-4, 0.6, 300))

    def __call__(self):
        sp = 2. * np.pi / 0.02  # pivot separation [Mpc/h]
        s = self.s
        ell_to_idx = {ell: i for i, ell in enumerate(self.ells)}
        xi_bare = self._to_correlation(self.pt.poles)
        broadband = np.zeros((len(self.ells), len(s)))
        for (ell, pow), param in zip(self._bb_ell_pow, self.bb_params):
            ill = ell_to_idx[ell]
            broadband[ill] += float(param.value) * (s / sp) ** pow
        self.poles = xi_bare + broadband
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class DampedBAOWigglesTracerCorrelation2Poles(_BAOWigglesTracerCorrelation2Poles):
    r"""
    DampedBAOWigglesPTCorrelation2Poles with additive s-space polynomial broadband.

    Broadband parameters ``al{ell}_{pow}`` are created automatically for each
    (ell, pow) combination. The pivot separation is sp = 2*pi / 0.02 ~ 314 Mpc/h.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h].
    pt : DampedBAOWigglesPTSpectrum2Poles, default=None
        Bare spectrum theory. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    broadband_pows : tuple of int, default=(-2, -1, 0, 1, 2)
        Powers of s/sp to include in the broadband.
    """
    _default_pt_cls = DampedBAOWigglesPTSpectrum2Poles


class ResummedBAOWigglesTracerCorrelation2Poles(_BAOWigglesTracerCorrelation2Poles):
    r"""
    ResummedBAOWigglesPTCorrelation2Poles with additive s-space polynomial broadband.

    Broadband parameters ``al{ell}_{pow}`` are created automatically for each
    (ell, pow) combination. The pivot separation is sp = 2*pi / 0.02 ~ 314 Mpc/h.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h].
    pt : ResummedBAOWigglesPTSpectrum2Poles, default=None
        Bare spectrum theory. A default instance is created if None.
    ells : tuple of int, default=(0, 2)
        Multipole orders.
    broadband_pows : tuple of int, default=(-2, -1, 0, 1, 2)
        Powers of s/sp to include in the broadband.
    """
    _default_pt_cls = ResummedBAOWigglesPTSpectrum2Poles
