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
from ...parameter import Parameter, VariableCollection
from .template import BAOSpectrum2Template
from ._multitracer import propose_params_multitracer, assign_params
from .fftlog import PowerToCorrelation as _PowerToCorrelation


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

class ProjectToPoles:
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

    @classmethod
    def propose_params(cls, tracers=None):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer([
            Parameter('b1', value=1., prior=dict(limits=[0., 4.]),
                      ref=dict(dist='norm', loc=1., scale=0.1), latex='b_1'),
            Parameter('dbeta', value=1., prior=dict(limits=[0., 3.]),
                      ref=dict(dist='norm', loc=1., scale=0.05), fd_eps=0.02, latex=r'\delta\beta'),
            Parameter('sigmas', value=0., prior=dict(limits=[-1., 10.]),
                      ref=dict(dist='norm', loc=0., scale=1.), latex=r'\Sigma_s'),
            Parameter('sigmapar', value=9., fixed=True, prior=dict(limits=[0., 25.]),
                      ref=dict(dist='norm', loc=9., scale=1.), latex=r'\Sigma_\parallel'),
            Parameter('sigmaper', value=6., fixed=True, prior=dict(limits=[0., 20.]),
                      ref=dict(dist='norm', loc=6., scale=1.), latex=r'\Sigma_\perp'),
        ], tracers)

    def __init__(self, k=None, template=None, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
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
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
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

    @classmethod
    def propose_params(cls, tracers=None):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer([
            Parameter('b1', value=1., prior=dict(limits=[0., 4.]),
                      ref=dict(dist='norm', loc=1., scale=0.1), latex='b_1'),
            Parameter('dbeta', value=1., prior=dict(limits=[0., 3.]),
                      ref=dict(dist='norm', loc=1., scale=0.05), fd_eps=0.02, latex=r'\delta\beta'),
            Parameter('sigmas', value=0., prior=dict(limits=[-1., 10.]),
                      ref=dict(dist='norm', loc=0., scale=1.), latex=r'\Sigma_s'),
            Parameter('d', value=1., fixed=True, prior=dict(limits=[0., 3.]),
                      ref=dict(dist='norm', loc=1., scale=0.05), latex='d'),
        ], tracers)

    def __init__(self, k=None, template=None, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
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
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
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


# ── broadband helpers ─────────────────────────────────────────────────────────

#: Power-law broadband modes (Pk powers range).
_BB_SPECTRUM_POWER_POWS = (-3, -2, -1, 0, 1)
#: Power-law broadband modes (xi powers range).
_BB_CORRELATION_POWER_POWS = (-2, -1, 0, 1, 2)
#: Kernel node indices (enough for k < 0.4 h/Mpc or reasonable s range).
_BB_SPECTRUM_KERNEL_IKS = tuple(range(-2, 10))
_BB_CORRELATION_KERNEL_IKS = tuple(range(-2, 3))
#: Real-space (bl) correction powers for kernel xi broadband.
_BB_CORRELATION_BL_POWS = (0, 2)

_BB_POWER_MODES = ('power', 'power3', 'even-power')
_BB_KERNEL_MODES = ('ngp', 'cic', 'tsc', 'pcs', 'pcs2')


def _kernel_func(x, kernel='pcs'):
    """Evaluate a 1-D window kernel at |x|.

    Parameters
    ----------
    x : array
        Dimensionless distance from kernel center (|k/kp - ik|).
    kernel : str
        One of ``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``.

    Returns
    -------
    ndarray of same shape as *x*, values in [0, 1].
    """
    x = np.abs(x)
    if kernel == 'ngp':
        return np.where(x < 0.5, 1., 0.)
    if kernel == 'cic':
        return np.where(x < 1., 1. - x, 0.)
    if kernel == 'tsc':
        return np.where(x < 0.5, 0.75 - x**2,
               np.where(x < 1.5, 0.5 * (1.5 - x)**2, 0.))
    if kernel in ('pcs', 'pcs2'):
        # Piecewise cubic spline over [0, 2].
        return np.where(x < 1., (4. - 6.*x**2 + 3.*x**3) / 4.,
               np.where(x < 2., (2. - x)**3 / 4., 0.))
    raise ValueError(f'Unknown broadband kernel {kernel!r}; choose ngp/cic/tsc/pcs.')


def _bb_spectrum_auto_params(ells, broadband):
    """Return the ``Parameter`` list for Pk broadband of the given mode."""
    auto_params = []
    ells = tuple(ells)
    if 'power' in broadband:
        for ell in ells:
            for pow in _BB_SPECTRUM_POWER_POWS:
                fixed = (broadband == 'power3') and (pow not in (-2, -1, 0))
                auto_params.append(Parameter(f'al{ell}_{pow}', value=0., fixed=fixed,
                                             prior=None, ref=dict(dist='norm', loc=0., scale=1.),
                                             fd_eps=0.005, latex=f'a_{{{ell},{pow}}}'))
    else:
        for ell in ells:
            for ik in _BB_SPECTRUM_KERNEL_IKS:
                auto_params.append(Parameter(f'al{ell}_{ik}', value=0.,
                                             prior=dict(dist='norm', loc=0., scale=1e4),
                                             ref=dict(dist='norm', loc=0., scale=1e-2),
                                             fd_eps=0.005, latex=f'a_{{{ell},{ik}}}'))
    return auto_params


def _bb_correlation_auto_params(ells, broadband):
    """Return the ``Parameter`` list for xi broadband of the given mode."""
    auto_params = []
    ells = tuple(ells)
    if 'power' in broadband:
        for ell in ells:
            for pow in _BB_CORRELATION_POWER_POWS:
                fixed = ((broadband == 'power3') and (pow not in (-2, -1, 0))) or \
                        ((broadband == 'even-power') and (pow not in (0, 2)))
                auto_params.append(Parameter(f'al{ell}_{pow}', value=0., fixed=fixed,
                                             prior=None, ref=dict(dist='norm', loc=0., scale=1.),
                                             fd_eps=0.005, latex=f'a_{{{ell},{pow}}}'))
    else:
        for ell in ells:
            for ik in _BB_CORRELATION_KERNEL_IKS:
                fixed = (broadband == 'pcs2') and (ell != 0 or ik not in (0, 1))
                auto_params.append(Parameter(f'al{ell}_{ik}', value=0., fixed=fixed,
                                             prior=None, ref=dict(dist='norm', loc=0., scale=1e2),
                                             fd_eps=0.005, latex=f'a_{{{ell},{ik}}}'))
        for ell in ells:
            for pow in _BB_CORRELATION_BL_POWS:
                auto_params.append(Parameter(f'bl{ell}_{pow}', value=0.,
                                             prior=None, ref=dict(dist='norm', loc=0., scale=1e-3),
                                             fd_eps=0.005, latex=f'b_{{{ell},{pow}}}'))
    return auto_params


# ── tracer base (broadband) ───────────────────────────────────────────────────

class _BAOWigglesTracerSpectrum2Poles(Calculator):
    r"""Base for BAO tracer power spectrum theories: bare model + additive broadband.

    Supports four broadband parameterizations (``broadband=`` kwarg):

    - ``'power'``: :math:`\sum_n a_{\ell,n}\,(k/k_p)^n`, :math:`n \in \{-3,\ldots,1\}`.
    - ``'power3'``: same but only :math:`n \in \{-2,-1,0\}` are free (others fixed at 0).
    - ``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``: kernel basis,
      :math:`\sum_{i} a_{\ell,i}\,W(|k/k_p - i|)\,P_\mathrm{nw}(i\,k_p)`.

    Parameters
    ----------
    k : array, default=None
    pt : bare BAO PT class, default=None
    ells : tuple of int, default=(0, 2)
    broadband : str, default='power'
    kp : float, default=None
        Broadband pivot wavenumber [h/Mpc].  Defaults to :math:`2\pi/r_d^\mathrm{fid}`.
    tracers, params : standard multitracer/override kwargs.

    Subclasses set ``_default_pt_cls`` to the bare theory class.
    """

    _default_pt_cls = None  # set by DampedBAOWigglesTracerSpectrum2Poles etc.

    @classmethod
    def propose_params(cls, tracers=None, ells=(0, 2), broadband='power'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        ells : tuple of int, default=(0, 2)
        broadband : str, default='power'
            Broadband parameterization.  One of ``'power'``, ``'power3'``, ``'ngp'``,
            ``'cic'``, ``'tsc'``, ``'pcs'``.

        Returns
        -------
        VariableCollection
        """
        if broadband not in _BB_POWER_MODES + _BB_KERNEL_MODES:
            raise ValueError(f'Unknown broadband mode {broadband!r}.')
        pt_vc = cls._default_pt_cls.propose_params(tracers=tracers) if cls._default_pt_cls is not None else VariableCollection()
        bb_vc = propose_params_multitracer(_bb_spectrum_auto_params(ells, broadband), tracers)
        return pt_vc + bb_vc

    def __init__(self, k=None, pt=None, ells=(0, 2), broadband='power', kp=None, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        _ells = tuple(ells)
        vc = type(self).propose_params(tracers=tracers, ells=_ells, broadband=broadband)
        if params is not None:
            vc = VariableCollection(params)
        # Separate broadband params (al*) from PT params; route each to the right owner.
        # We bypass assign_params for bb_params because some basenames (e.g. al0_0) are valid
        # identifiers and would be split out of the list, breaking matrix row-index correspondence.
        bb_basenames = {p.basename for p in _bb_spectrum_auto_params(_ells, broadband)}
        bb_vc = VariableCollection([p for p in vc if p.basename in bb_basenames])
        pt_vc = VariableCollection([p for p in vc if p.basename not in bb_basenames])
        self.bb_params = list(bb_vc)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = _ells
        if pt is None:
            pt = self._default_pt_cls(tracers=tracers, params=pt_vc if len(pt_vc) else None, **kwargs)
        self.pt = pt
        self.pt.update(k=self.k, ells=self.ells)

    def __post_init__(self, k=None, pt=None, ells=(0, 2), broadband='power', kp=None, tracers=None, params=None, **kwargs):
        # Non-node setup: build the broadband basis matrix keyed on each param's basename.
        _ells = tuple(ells)
        kp_val = float(kp) if kp is not None else 2. * np.pi / self.pt.template._fiducial.rs_drag
        kernel = None if 'power' in broadband else broadband[:3]  # 'pcs2' -> 'pcs'
        if kernel is not None:
            pk_now_at = lambda ki: float(np.interp(ki, self.pt.template.k, self.pt.template._pknow_dd_fid))
        # Build one matrix row per param, grouped by ell.
        # Basename format: 'al{ell}_{pow_or_ik}' (e.g. 'al0_-2', 'al2_1').
        rows_per_ell = [[] for _ in _ells]
        param_indices_per_ell = [[] for _ in _ells]
        for param_idx, param in enumerate(self.bb_params):
            parts = param.basename.split('_', 1)
            ell = int(parts[0][2:])
            val = int(parts[1])
            ill = _ells.index(ell)
            param_indices_per_ell[ill].append(param_idx)
            if kernel is None:
                rows_per_ell[ill].append((self.k / kp_val) ** val)
            else:
                w = _kernel_func(np.abs(self.k / kp_val - val), kernel=kernel)
                rows_per_ell[ill].append(w * pk_now_at(float(np.clip(val * kp_val, self.k[0], self.k[-1]))))
        self._bb_ell_matrices = [np.stack(rows) if rows else np.zeros((0, len(self.k))) for rows in rows_per_ell]
        self._bb_ell_param_indices = param_indices_per_ell

    def __call__(self):
        broadband = jnp.zeros((len(self.ells), len(self.k)))
        for ill, (indices, mat_ell) in enumerate(zip(self._bb_ell_param_indices, self._bb_ell_matrices)):
            if indices:
                bb_vals = jnp.stack([self.bb_params[idx].value for idx in indices])
                broadband = broadband.at[ill].add(bb_vals.dot(jnp.asarray(mat_ell)))
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
    DampedBAOWigglesPTSpectrum2Poles with additive broadband.

    Supports ``broadband=`` modes: ``'power'`` (default), ``'power3'``,
    ``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``.
    Pivot wavenumber ``kp`` defaults to :math:`2\pi/r_d^\mathrm{fid}`.

    Parameters
    ----------
    k : array, default=None
    pt : DampedBAOWigglesPTSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2)
    broadband : str, default='power'
    kp : float, default=None
    """
    _default_pt_cls = DampedBAOWigglesPTSpectrum2Poles


class ResummedBAOWigglesTracerSpectrum2Poles(_BAOWigglesTracerSpectrum2Poles):
    r"""
    ResummedBAOWigglesPTSpectrum2Poles with additive broadband.

    Supports ``broadband=`` modes: ``'power'`` (default), ``'power3'``,
    ``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``.
    Pivot wavenumber ``kp`` defaults to :math:`2\pi/r_d^\mathrm{fid}`.

    Parameters
    ----------
    k : array, default=None
    pt : ResummedBAOWigglesPTSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2)
    broadband : str, default='power'
    kp : float, default=None
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
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        self.kin = np.asarray(kin, dtype='f8')
        k_fftlog = np.logspace(-4., 3., 2048)
        mask_high = k_fftlog > kin[-1]
        self.k_mid = k_fftlog[~mask_high]
        self.logk_high = np.log10(k_fftlog[mask_high] / kin[-1])
        self.damp_high = np.exp(-(k_fftlog[mask_high] / kin[-1] - 1.) ** 2 / 200.)
        self.fftlog = _PowerToCorrelation(k_fftlog, ell=self.ells, q=0, lowring=True)

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

class _BAOWigglesPTCorrelation2Poles(Calculator):
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

class _BAOWigglesTracerCorrelation2Poles(Calculator):
    r"""Base for BAO tracer correlation function theories.

    Supports the same ``broadband=`` modes as :class:`_BAOWigglesTracerSpectrum2Poles`:

    **Power-law modes** (``'power'``, ``'power3'``, ``'even-power'``):
      :math:`\xi_\ell(s) = \xi_\ell^\mathrm{bare}(s) + \sum_n a_{\ell,n}(s/s_p)^n`
      where :math:`s_p = 2\pi/0.02 \approx 314\,\mathrm{Mpc}/h`.
      Parameters: ``al{ell}_{pow}``.

    **Kernel modes** (``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``, ``'pcs2'``):
      The :math:`a_{\ell,i}` broadband is applied in Fourier space (owned by an inner
      :class:`_BAOWigglesTracerSpectrum2Poles` dep), plus an additive s-space correction
      :math:`\sum_{n \in \{0,2\}} b_{\ell,n}(s/s_p)^n` (``bl{ell}_{pow}`` parameters).

    Subclasses set ``_default_pt_cls`` (bare PT class) and ``_default_tracer_cls``
    (tracer spectrum class used for kernel modes).
    """

    _default_pt_cls = None
    _default_tracer_cls = None  # set by concrete subclasses; used for kernel broadband

    @classmethod
    def propose_params(cls, tracers=None, ells=(0, 2), broadband='power'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        ells : tuple of int, default=(0, 2)
        broadband : str, default='power'
            One of ``'power'``, ``'power3'``, ``'even-power'``, ``'ngp'``,
            ``'cic'``, ``'tsc'``, ``'pcs'``, ``'pcs2'``.

        Returns
        -------
        VariableCollection
        """
        if broadband not in _BB_POWER_MODES + ('even-power',) + _BB_KERNEL_MODES:
            raise ValueError(f'Unknown broadband mode {broadband!r}.')
        pt_vc = cls._default_pt_cls.propose_params(tracers=tracers) if cls._default_pt_cls is not None else VariableCollection()
        bb_vc = propose_params_multitracer(_bb_correlation_auto_params(ells, broadband), tracers)
        return pt_vc + bb_vc

    def __init__(self, s=None, pt=None, ells=(0, 2), broadband='power', sp=None, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        _ells = tuple(ells)
        vc = type(self).propose_params(tracers=tracers, ells=_ells, broadband=broadband)
        if params is not None:
            vc = VariableCollection(params)

        # Separate broadband params from PT params; route each to the right owner.
        bb_basenames = {p.basename for p in _bb_correlation_auto_params(_ells, broadband)}
        pt_vc = VariableCollection([p for p in vc if p.basename not in bb_basenames])

        if 'power' in broadband or broadband == 'even-power':
            # Power-law: al params live on this class; pt is a bare PT.
            # Store as ordered list; build_graph discovers them via self.bb_params.
            bb_vc = VariableCollection([p for p in vc if p.basename in bb_basenames])
            self.bb_params = list(bb_vc)
            if pt is None:
                pt = self._default_pt_cls(tracers=tracers, params=pt_vc if len(pt_vc) else None, **kwargs)
            self.pt = pt
            self.pt.update(k=np.geomspace(1e-4, 0.6, 300), ells=_ells)
        else:
            # Kernel: al params + PT params go to the inner tracer spectrum dep; bl params stay here.
            # pt (if provided) is a bare PT, forwarded to the tracer spectrum — matching bak.
            al_vc = VariableCollection([p for p in vc if p.basename in bb_basenames and p.basename.startswith('al') and not p.fixed])
            bl_vc = VariableCollection([p for p in vc if p.basename in bb_basenames and p.basename.startswith('bl')])
            self.pt = self._default_tracer_cls(broadband=broadband, tracers=tracers, pt=pt, params=al_vc + pt_vc, **kwargs)
            self.pt.update(k=np.geomspace(1e-4, 0.6, 300), ells=_ells)
            # bl params: real-space correction; stored as list so build_graph discovers them.
            self.bl_params = list(bl_vc)

        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = _ells

    def __post_init__(self, s=None, pt=None, ells=(0, 2), broadband='power', sp=None, tracers=None, params=None, **kwargs):
        # Non-node setup: FFTLog transformer + broadband basis matrix keyed on each param's basename.
        _ells = tuple(ells)
        self._to_correlation = SpectrumToCorrelation(s=self.s, ells=_ells, kin=np.geomspace(1e-4, 0.6, 300))
        sp_val = float(sp) if sp is not None else 2. * np.pi / 0.02
        # Power modes: al* params live on this class; kernel mode: bl* params live here, al* are in self.pt.
        bb_params_flat = self.bb_params if 'power' in broadband or broadband == 'even-power' else self.bl_params
        # Build one matrix row per param from its 'al/bl{ell}_{pow}' basename.
        rows_per_ell = [[] for _ in _ells]
        param_indices_per_ell = [[] for _ in _ells]
        for param_idx, param in enumerate(bb_params_flat):
            parts = param.basename.split('_', 1)
            ell = int(parts[0][2:])
            val = int(parts[1])
            ill = _ells.index(ell)
            param_indices_per_ell[ill].append(param_idx)
            rows_per_ell[ill].append((self.s / sp_val) ** val)
        self._bb_ell_matrices = [np.stack(rows) if rows else np.zeros((0, len(self.s))) for rows in rows_per_ell]
        self._bb_ell_param_indices = param_indices_per_ell
        self._bb_params_flat = bb_params_flat

    def __call__(self):
        xi_bare = self._to_correlation(self.pt.poles)
        broadband = jnp.zeros((len(self.ells), len(self.s)))
        for ill, (indices, mat_ell) in enumerate(zip(self._bb_ell_param_indices, self._bb_ell_matrices)):
            if indices:
                bb_vals = jnp.stack([self._bb_params_flat[idx].value for idx in indices])
                broadband = broadband.at[ill].add(bb_vals.dot(jnp.asarray(mat_ell)))
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
    BAO correlation function with damped wiggles and additive broadband.

    Supports ``broadband=`` modes: ``'power'`` (default), ``'power3'``,
    ``'even-power'``, ``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``, ``'pcs2'``.
    Pivot separation ``sp`` defaults to :math:`2\pi/0.02 \approx 314\,\mathrm{Mpc}/h`.

    Parameters
    ----------
    s : array, default=None
    pt : DampedBAOWigglesPTSpectrum2Poles or DampedBAOWigglesTracerSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2)
    broadband : str, default='power'
    sp : float, default=None
    """
    _default_pt_cls = DampedBAOWigglesPTSpectrum2Poles
    _default_tracer_cls = DampedBAOWigglesTracerSpectrum2Poles


class ResummedBAOWigglesTracerCorrelation2Poles(_BAOWigglesTracerCorrelation2Poles):
    r"""
    BAO correlation function with resummed wiggles and additive broadband.

    Supports ``broadband=`` modes: ``'power'`` (default), ``'power3'``,
    ``'even-power'``, ``'ngp'``, ``'cic'``, ``'tsc'``, ``'pcs'``, ``'pcs2'``.
    Pivot separation ``sp`` defaults to :math:`2\pi/0.02 \approx 314\,\mathrm{Mpc}/h`.

    Parameters
    ----------
    s : array, default=None
    pt : ResummedBAOWigglesPTSpectrum2Poles or ResummedBAOWigglesTracerSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2)
    broadband : str, default='power'
    sp : float, default=None
    """
    _default_pt_cls = ResummedBAOWigglesPTSpectrum2Poles
    _default_tracer_cls = ResummedBAOWigglesTracerSpectrum2Poles
