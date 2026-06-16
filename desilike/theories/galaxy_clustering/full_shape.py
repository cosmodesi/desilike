"""
Full-shape power spectrum and correlation function multipoles.

Classes
-------
KaiserPTSpectrum2Poles
    Kaiser (linear) matter power spectrum multipoles with AP distortion and Gaussian FoG damping.
KaiserTracerSpectrum2Poles
    KaiserPTSpectrum2Poles with linear bias b1 and shot noise.
KaiserTracerCorrelation2Poles
    KaiserTracerSpectrum2Poles Fourier-transformed to configuration space via FFTLog.
TNSPTSpectrum2Poles
    TNS 1-loop matter power spectrum multipoles (Taruya, Nishimichi & Saito 2010).
TNSTracerSpectrum2Poles
    TNSPTSpectrum2Poles with full 1-loop bias expansion.
TNSTracerCorrelation2Poles
    TNSTracerSpectrum2Poles Fourier-transformed to configuration space via FFTLog.
"""

import numpy as np
import jax
import jax.numpy as jnp
import interpax

import os

from ...base import Calculator
from ...parameter import Parameter, VariableCollection
from ..primordial_cosmology import CosmoprimoCosmology, _get_fiducial
from .bao import ProjectToPoles, SpectrumToCorrelation
from .template import DirectSpectrum2Template, _ap_k_mu
from ._multitracer import propose_params_multitracer, assign_params


# ── utilities ─────────────────────────────────────────────────────────────────

def _velocileptors_default_params(prior_basis):
    """Return the 11 default auto_params for LPT/REPT Velocileptors tracer classes."""
    if prior_basis == 'physical':
        return [
            Parameter('b1p', value=1., prior=dict(dist='uniform', limits=[0., 3.]), ref=dict(dist='norm', loc=1., scale=0.1), latex=r"b_1'"),
            Parameter('b2p', value=0., prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_2'"),
            Parameter('bsp', value=0., prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_s'"),
            Parameter('b3p', value=0., fixed=True, latex=r"b_3'"),
            Parameter('alpha0p', value=0., prior=dict(dist='norm', loc=0., scale=12.5), ref=dict(dist='norm', loc=0., scale=1.), latex=r"\alpha_0'"),
            Parameter('alpha2p', value=0., prior=dict(dist='norm', loc=0., scale=12.5), ref=dict(dist='norm', loc=0., scale=1.), latex=r"\alpha_2'"),
            Parameter('alpha4p', value=0., prior=dict(dist='norm', loc=0., scale=12.5), ref=dict(dist='norm', loc=0., scale=1.), latex=r"\alpha_4'"),
            Parameter('alpha6p', value=0., fixed=True, latex=r"\alpha_6'"),
            Parameter('sn0p', value=0., prior=dict(dist='norm', loc=0., scale=2.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,0}'"),
            Parameter('sn2p', value=0., prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,2}'"),
            Parameter('sn4p', value=0., prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,4}'"),
        ]
    return [
        Parameter('b1', value=1., prior=dict(limits=[-1., 10.]), ref=dict(limits=[0.4, 0.6]), latex='b_1'),
        Parameter('b2', value=0., prior=dict(dist='norm', loc=0., scale=10.), ref=dict(dist='norm', loc=0., scale=0.5), latex='b_2'),
        Parameter('bs', value=0., prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=0.5), latex='b_s'),
        Parameter('b3', value=0., fixed=True, latex='b_3'),
        Parameter('alpha0', value=0., prior=dict(dist='norm', loc=0., scale=30.), ref=dict(dist='norm', loc=0., scale=1.), latex=r'\alpha_0'),
        Parameter('alpha2', value=0., prior=dict(dist='norm', loc=0., scale=50.), ref=dict(dist='norm', loc=0., scale=1.), latex=r'\alpha_2'),
        Parameter('alpha4', value=0., prior=dict(dist='norm', loc=0., scale=50.), ref=dict(dist='norm', loc=0., scale=1.), latex=r'\alpha_4'),
        Parameter('alpha6', value=0., fixed=True, latex=r'\alpha_6'),
        Parameter('sn0', value=0., prior=dict(dist='norm', loc=0., scale=4.), ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,0}'),
        Parameter('sn2', value=0., prior=dict(dist='norm', loc=0., scale=100.), ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,2}'),
        Parameter('sn4', value=0., prior=dict(dist='norm', loc=0., scale=500.), ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,4}'),
    ]


def get_nthreads(nthreads=None):
    """Number of threads for external (velocileptors) calls; defaults to ``$OMP_NUM_THREADS`` or 1."""
    if nthreads is None:
        nthreads = os.getenv('OMP_NUM_THREADS', '1')
    return int(nthreads)


#: Valid FOLPS prior_basis options (mirrors FOLPSv2 in desilike_bak).
_FOLPS_PRIOR_BASES = ('standard', 'physical', 'physical_aap', 'tcm_chudaykin_aap')


def get_physical_stochastic_settings(tracer=None):
    """Per-tracer satellite fraction ``fsat`` and velocity dispersion ``sigv`` for the
    physical_aap stochastic terms (Mark Maus, Ruiyang Zhao). ``tracer=None`` gives generic defaults."""
    if tracer is not None:
        tracer = str(tracer).upper()
        settings = {'BGS': {'fsat': 0.13, 'sigv': 150 / 70. * 10**(1 / 3) * (1 + 0.2)**0.5},
                    'LRG': {'fsat': 0.13, 'sigv': 150 / 70. * 10**(1 / 3) * (1 + 0.8)**0.5},
                    'ELG': {'fsat': 0.06, 'sigv': 150 / 70. * 2.1**0.5},
                    'QSO': {'fsat': 0.2, 'sigv': 150 / 70. * 10**(0.7 / 3) * 2.4**0.5}}
        try:
            settings = settings[tracer]
        except KeyError:
            raise ValueError('unknown tracer: {}, please use any of {}'.format(tracer, list(settings.keys())))
    else:
        settings = {'fsat': 0.1, 'sigv': 5.}
    return settings


def _velocileptors_kvec(k, boost_prec=2):
    """Internal velocileptors evaluation k-grid spanning ``[k[0], k[-1]]`` with margin for the
    cubic AP interpolation below/above (and numerical noise at the endpoint)."""
    k = np.asarray(k, dtype='f8')
    return np.concatenate([[min(0.0005, k[0])],
                           np.geomspace(0.0015, 0.025, 10 * boost_prec, endpoint=True),
                           np.arange(0.03, max(0.5, k[-1]) + 0.015 / boost_prec, 0.01 / boost_prec)])


def _velocileptors_physical_to_standard(b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6,
                                   sn0, sn2, sn4, f, fsat, sigv, nbar, A=1., A_AP=1., rept=False):
    r"""Convert physical-basis parameters to the standard velocileptors bias vector ``[b1, b2, bs, b3, alpha0, alpha2,
    alpha4, alpha6, sn0, sn2, sn4]``.

    The first four are Lagrangian bias for LPT; for REPT (``rept=True``) they are converted to
    the Eulerian basis (:math:`b_1 = 1 + b_1^L,\ b_2 = 8/21\,b_1^L + b_2^L,\ b_s = b_s^L,\ b_3 = b_3^L`).
    ``alpha6`` is unused (``alpha6 = f^2 alpha4``), kept for a uniform signature.
    """
    b1E = b1 / (A * A_AP**0.5)
    b1L = b1E - 1.
    # b2E shifted by + 8. / 21. * b1L to center the prior on 0 (coevolution)
    b2E = b2 / (A**2 * A_AP**0.5) + 8. / 21. * b1L
    bK2 = bs / (A**2 * A_AP**0.5)
    # It looks like REPT also used bsL, whose prior is already centered on 0 (coevolution)
    bsL = bK2
    # Same about b3
    b3L = b3 / (A**4 * A_AP)
    if rept:  # REPT: Eulerian bias
        bias = [b1E, b2E, bsL, b3L]
    else:  # LPT: Lagrangian bias
        bias = [b1L, b2E - 8. / 21. * b1L, bsL, b3L]
    alpha0, alpha2, alpha4 = alpha0 / (A**2 * A_AP), alpha2 / (A**2 * A_AP), alpha4 / (A**2 * A_AP)
    alphas = [(1. + b1L)**2 * alpha0,
              f * (1. + b1L) * (alpha0 + alpha2),
              f * (f * alpha2 + (1. + b1L) * alpha4),
              f**2 * alpha4]
    sn0, sn2, sn4 = sn0 / A_AP, sn2 / A_AP, sn4 / A_AP
    stoch = [sn0 / nbar, sn2 / nbar * fsat * sigv**2, sn4 / nbar * fsat * sigv**4]
    return jnp.array(bias + alphas + stoch)


def _velocileptors_combine_bias_terms_spectrum2_poles(pktable, pars, nd=1e-4):
    """Contract a velocileptors bias table ``(n_ells, n_k, 19)`` with the 11 bias parameters."""
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4 = pars
    bias_monomials = jnp.array([1., b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3,
                                alpha0, alpha2, alpha4, alpha6, sn0 / nd, sn2 / nd, sn4 / nd])
    return jnp.sum(pktable * bias_monomials, axis=-1)


def _weights_trapz(x):
    return np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.


def _interp_loglog(k_query, k_knots, pk_knots):
    """Cubic spline interpolation in log10(k) space."""
    shape = jnp.shape(k_query)
    flat = jnp.ravel(k_query)
    result = interpax.interp1d(jnp.log10(flat), jnp.log10(k_knots), pk_knots, method='cubic', extrap=True)
    return jnp.reshape(result, shape)


# ── TNS perturbation theory ───────────────────────────────────────────────────

def tns_kernels(k, q, wq):
    """Precompute numpy kernel arrays for 1-loop TNS integrals at wavenumbers k."""
    jq = q**2 * wq / (4. * np.pi**2)
    k = k[:, None]
    x = q / k

    def kernel_ff(x):
        x = np.array(x)
        toret = (6. / x**2 - 79. + 50. * x**2 - 21. * x**4 + 0.75 * (1. / x - x)**3 * (2. + 7. * x**2) * 2 * np.log(np.abs((x - 1.) / (x + 1.)))) / 504.
        mask = x > 10.
        toret[mask] = - 61. / 630. + 2. / 105. / x[mask]**2 - 10. / 1323. / x[mask]**4
        dx = x - 1.
        mask = np.abs(dx) < 0.01
        toret[mask] = - 11. / 126. + dx[mask] / 126. - 29. / 252. * dx[mask]**2
        return toret / x**2

    def kernel_gg(x):
        x = np.array(x)
        toret = (6. / x**2 - 41. + 2. * x**2 - 3. * x**4 + 0.75 * (1. / x - x)**3 * (2. + x**2) * 2 * np.log(np.abs((x - 1.) / (x + 1.)))) / 168.
        mask = x > 10.
        toret[mask] = - 3. / 10. + 26. / 245. / x[mask]**2 - 38. / 2205. / x[mask]**4
        dx = x - 1.
        mask = np.abs(dx) < 0.01
        toret[mask] = - 3. / 14. - 5. / 42. * dx[mask] - 1. / 84. * dx[mask]**2
        return toret / x**2

    kernels = [2 * jq * kernel_ff(x), 2 * jq * kernel_gg(x)]

    def kernel_a(x):
        toret = np.zeros((5,) + x.shape, dtype='f8')
        logx = np.zeros_like(x)
        mask = np.abs(x - 1) > 1e-16
        logx[mask] = np.log(np.abs((x[mask] + 1) / (x[mask] - 1)))
        toret[0] = -1. / 84. / x * (2 * x * (19 - 24 * x**2 + 9 * x**4) - 9 * (x**2 - 1)**3 * logx)
        toret[1] = 1. / 112. / x**3 * (2 * x * (x**2 + 1) * (3 - 14 * x**2 + 3 * x**4) - 3 * (x**2 - 1)**4 * logx)
        toret[2] = 1. / 336. / x**3 * (2 * x * (9 - 185 * x**2 + 159 * x**4 - 63 * x**6) + 9 * (x**2 - 1)**3 * (7 * x**2 + 1) * logx)
        toret[4] = 1. / 336. / x**3 * (2 * x * (9 - 109 * x**2 + 63 * x**4 - 27 * x**6) + 9 * (x**2 - 1)**3 * (3 * x**2 + 1) * logx)
        mask = x < 1e-4
        xm = x[mask]
        toret[0][mask] = 8 * xm**8 / 735 + 24 * xm**6 / 245 - 24 * xm**4 / 35 + 8 * xm**2 / 7 - 2. / 3
        toret[1][mask] = - 16 * xm**8 / 8085 - 16 * xm**6 / 735 + 48 * xm**4 / 245 - 16 * xm**2 / 35
        toret[2][mask] = 32 * xm**8 / 1617 + 128 * xm**6 / 735 - 288 * xm**4 / 245 + 64 * xm**2 / 35 - 4. / 3
        toret[4][mask] = 24 * xm**8 / 2695 + 8 * xm**6 / 105 - 24 * xm**4 / 49 + 24 * xm**2 / 35 - 2. / 3
        mask = x > 1e2
        xm = x[mask]
        toret[0][mask] = 2. / 105 - 24 / (245 * xm**2) - 8 / (735 * xm**4) - 8 / (2695 * xm**6) - 8 / (7007 * xm**8)
        toret[1][mask] = -16. / 35 + 48 / (245 * xm**2) - 16 / (735 * xm**4) - 16 / (8085 * xm**6) - 16 / (35035 * xm**8)
        toret[2][mask] = -44. / 105 - 32 / (735 * xm**4) - 64 / (8085 * xm**6) - 96 / (35035 * xm**8)
        toret[4][mask] = -46. / 105 + 24 / (245 * xm**2) - 8 / (245 * xm**4) - 8 / (1617 * xm**6) - 8 / (5005 * xm**8)
        toret[3] = toret[1]
        return toret / x**2

    kernels.append(jq * kernel_a(x))
    return kernels


@jax.jit
def tns_pt(k, q, wq, pk_q, kernel13_d, kernel13_t, kernel_a):
    """1-loop TNS power spectrum components (JAX-jitted)."""
    k11 = k
    k = k[:, None]
    jq = q**2 * wq / (4. * np.pi**2)
    x = q / k

    # GL quadrature over mu in [0, 1] (symmetric half of [-1, 1]).
    _xf, _wf = np.polynomial.legendre.leggauss(20)
    mus = _xf[10:]
    wmus = (_wf[10:] + _wf[9::-1]) / 2.

    pk_k = jnp.interp(k11, q, pk_q)

    def get_terms(mu, wmu):
        kdq = k * q * mu
        kq2 = k**2 - 2. * kdq + q**2
        qdkq = kdq - q**2
        F2_d = 5. / 7. + 1. / 2. * qdkq * (1. / q**2 + 1. / kq2) + 2. / 7. * qdkq**2 / (q**2 * kq2)
        F2_t = 3. / 7. + 1. / 2. * qdkq * (1. / q**2 + 1. / kq2) + 4. / 7. * qdkq**2 / (q**2 * kq2)
        S = qdkq**2 / (q**2 * kq2) - 1. / 3.
        D = 2. / 7. * (mu**2 - 1.)
        pk_kq = jnp.interp(kq2**0.5, q, pk_q, left=0., right=0.)
        jq_pk_q_pk_kq = jq * pk_q * pk_kq

        _pk_b2d = wmu * jnp.sum(jq_pk_q_pk_kq * F2_d, axis=-1)
        _pk_bs2d = wmu * jnp.sum(jq_pk_q_pk_kq * F2_d * S, axis=-1)
        _pk_b2t = wmu * jnp.sum(jq_pk_q_pk_kq * F2_t, axis=-1)
        _pk_bs2t = wmu * jnp.sum(jq_pk_q_pk_kq * F2_t * S, axis=-1)
        _sig3sq = wmu * jnp.sum(105. / 16. * jq * pk_q * (D * S + 8. / 63.), axis=-1)
        _pk_b22 = wmu / 2. * jnp.sum(jq * pk_q * (pk_kq - pk_q), axis=-1)
        _pk_b2s2 = wmu / 2. * jnp.sum(jq * pk_q * (pk_kq * S - 2. / 3. * pk_q), axis=-1)
        _pk_bs22 = wmu / 2. * jnp.sum(jq * pk_q * (pk_kq * S**2 - 4. / 9. * pk_q), axis=-1)
        _pk22_dd = 2 * wmu * jnp.sum(F2_d**2 * jq_pk_q_pk_kq, axis=-1)
        _pk22_dt = 2 * wmu * jnp.sum(F2_d * F2_t * jq_pk_q_pk_kq, axis=-1)
        _pk22_tt = 2 * wmu * jnp.sum(F2_t * F2_t * jq_pk_q_pk_kq, axis=-1)

        xmu = kq2 / k**2
        kernel_A = [0] * 5
        kernel_tA = [0] * 5
        kernel_A[0] = - x**3 / 7. * (mu + 6 * mu**3 + x**2 * mu * (-3 + 10 * mu**2) + x * (-3 + mu**2 - 12 * mu**4))
        kernel_A[1] = x**4 / 14. * (mu**2 - 1) * (-1 + 7 * x * mu - 6 * mu**2)
        kernel_A[2] = x**3 / 14. * (x**2 * mu * (13 - 41 * mu**2) - 4 * (mu + 6 * mu**3) + x * (5 + 9 * mu**2 + 42 * mu**4))
        kernel_A[3] = kernel_A[1]
        kernel_A[4] = x**3 / 14. * (1 - 7 * x * mu + 6 * mu**2) * (-2 * mu + x * (-1 + 3 * mu**2))
        kernel_tA[0] = 1. / 7. * (mu + x - 2 * x * mu**2) * (3 * x + 7 * mu - 10 * x * mu**2)
        kernel_tA[1] = x / 14. * (mu**2 - 1) * (3 * x + 7 * mu - 10 * x * mu**2)
        kernel_tA[2] = 1. / 14. * (28 * mu**2 + x * mu * (25 - 81 * mu**2) + x**2 * (1 - 27 * mu**2 + 54 * mu**4))
        kernel_tA[3] = x / 14. * (1 - mu**2) * (x - 7 * mu + 6 * x * mu**2)
        kernel_tA[4] = 1. / 14. * (x - 7 * mu + 6 * x * mu**2) * (-2 * mu - x + 3 * x * mu**2)
        _A = wmu * jnp.sum(jq / x**2 * (jnp.array(kernel_A) * pk_k[:, None] + jnp.array(kernel_tA) * pk_q) * pk_kq / xmu**2, axis=-1)

        jq_pk_q_pk_kq /= x**2 * xmu
        _B = [0.] * 12
        _B[0] = wmu * jnp.sum(x**2 * (mu**2 - 1.) / 2. * jq_pk_q_pk_kq, axis=-1)
        _B[1] = wmu * jnp.sum(3. * x**2 * (mu**2 - 1.)**2 / 8. * jq_pk_q_pk_kq, axis=-1)
        _B[2] = wmu * jnp.sum(3. * x**4 * (mu**2 - 1.)**2 / xmu / 8. * jq_pk_q_pk_kq, axis=-1)
        _B[3] = wmu * jnp.sum(5. * x**4 * (mu**2 - 1.)**3 / xmu / 16. * jq_pk_q_pk_kq, axis=-1)
        _B[4] = wmu * jnp.sum(x * (x + 2. * mu - 3. * x * mu**2) / 2. * jq_pk_q_pk_kq, axis=-1)
        _B[5] = wmu * jnp.sum(- 3. * x * (mu**2 - 1.) * (-x - 2. * mu + 5. * x * mu**2) / 4. * jq_pk_q_pk_kq, axis=-1)
        _B[6] = wmu * jnp.sum(3. * x**2 * (mu**2 - 1.) * (-2. + x**2 + 6. * x * mu - 5. * x**2 * mu**2) / xmu / 4. * jq_pk_q_pk_kq, axis=-1)
        _B[7] = wmu * jnp.sum(- 3. * x**2 * (mu**2 - 1.)**2 * (6. - 5. * x**2 - 30. * x * mu + 35. * x**2 * mu**2) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)
        _B[8] = wmu * jnp.sum(x * (4. * mu * (3. - 5. * mu**2) + x * (3. - 30. * mu**2 + 35. * mu**4)) / 8. * jq_pk_q_pk_kq, axis=-1)
        _B[9] = wmu * jnp.sum(x * (-8. * mu + x * (-12. + 36. * mu**2 + 12. * x * mu * (3. - 5. * mu**2) + x**2 * (3. - 30. * mu**2 + 35. * mu**4))) / xmu / 8. * jq_pk_q_pk_kq, axis=-1)
        _B[10] = wmu * jnp.sum(3. * x * (mu**2 - 1.) * (-8. * mu + x * (-12. + 60. * mu**2 + 20. * x * mu * (3. - 7. * mu**2) + 5. * x**2 * (1. - 14. * mu**2 + 21. * mu**4))) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)
        _B[11] = wmu * jnp.sum(x * (8. * mu * (-3. + 5. * mu**2) - 6. * x * (3. - 30. * mu**2 + 35. * mu**4) + 6. * x**2 * mu * (15. - 70. * mu**2 + 63 * mu**4) + x**3 * (5. - 21. * mu**2 * (5. - 15. * mu**2 + 11. * mu**4))) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)
        return jnp.stack([_pk_b2d, _pk_bs2d, _pk_b2t, _pk_bs2t, _sig3sq, _pk_b22, _pk_b2s2, _pk_bs22, _pk22_dd, _pk22_dt, _pk22_tt] + list(_A) + _B)

    res = jnp.sum(jax.vmap(get_terms)(mus, wmus), axis=0)
    pk_b2d, pk_bs2d, pk_b2t, pk_bs2t, sig3sq, pk_b22, pk_b2s2, pk_bs22, pk22_dd, pk22_dt, pk22_tt = res[:11]
    A, B = res[11:16], res[16:]
    A += pk_k * jnp.sum(kernel_a * pk_q, axis=-1)
    pk13_dd = 2. * jnp.sum(kernel13_d * pk_q, axis=-1) * pk_k
    pk13_tt = 2. * jnp.sum(kernel13_t * pk_q, axis=-1) * pk_k
    pk13_dt = (pk13_dd + pk13_tt) / 2.
    pk_sig3sq = sig3sq * pk_k
    pk_dd = pk_k + pk22_dd + pk13_dd
    pk_dt = pk_k + pk22_dt + pk13_dt
    pk_tt = pk_k + pk22_tt + pk13_tt
    return [pk_k, pk_dd, pk_b2d, pk_bs2d, pk_sig3sq, pk_b22, pk_b2s2, pk_bs22, pk_dt, pk_b2t, pk_bs2t, pk_tt, A, B]


# ── Kaiser model ──────────────────────────────────────────────────────────────

class KaiserPTSpectrum2Poles(Calculator):
    r"""
    Kaiser power spectrum multipoles.

    AP distortion, optional Gaussian FoG damping, and GL projection to multipoles.
    Exposes ``table`` with keys ``pk_dd``, ``pk_dt``, ``pk_tt`` (and ``pk11 = pk_dd``).

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to np.linspace(0.01, 0.2, 101).
    template : template calculator, default=None
        Power spectrum template. A default ``DirectSpectrum2Template()`` is created if None.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders to compute.
    mu : int, default=8
        Number of Gauss-Legendre mu-bins in [0, 1].
    """

    def __init__(self, k=None, template=None, ells=(0, 2, 4), mu=8, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        self.sigmapar = Parameter('sigmapar', value=0., fixed=True, latex=r'\Sigma_\parallel')
        self.sigmaper = Parameter('sigmaper', value=0., fixed=True, latex=r'\Sigma_\perp')
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        k_min = min(1e-4, self.k[0] / 2.)
        k_max = max(1., self.k[-1] * 2.)
        self.template.update(k=np.geomspace(k_min, k_max, 500))

    def __post_init__(self, k=None, template=None, ells=(0, 2, 4), mu=8, **kwargs):
        # Non-node setup only.
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        self._mu = self._to_poles.mu

    def __call__(self):
        k = self.k[:, None]
        mu = self._mu
        jac, kap, muap = self.template.ap_k_mu(k, mu)
        f = self.template.f
        sigmanl2 = kap**2 * (self.sigmapar**2 * muap**2 + self.sigmaper**2 * (1. - muap**2))
        damping = jnp.exp(-sigmanl2 / 2.)
        pkt = jac * damping * _interp_loglog(kap, self.template.k, self.template.pk_dd)
        self.table = {
            'pk_dd': self._to_poles(pkt),
            'pk_dt': self._to_poles(f * muap**2 * pkt),
            'pk_tt': self._to_poles(f**2 * muap**4 * pkt),
        }
        self.table['pk11'] = self.table['pk_dd']

    def tree_flatten(self):
        return ([self.table['pk_dd'], self.table['pk_dt'], self.table['pk_tt']],
                {'k': self.k, 'ells': self.ells})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.table = {'pk_dd': children[0], 'pk_dt': children[1], 'pk_tt': children[2]}
        obj.table['pk11'] = obj.table['pk_dd']
        obj.k = aux['k']
        obj.ells = aux['ells']
        return obj


class KaiserTracerSpectrum2Poles(Calculator):
    r"""
    Kaiser tracer power spectrum multipoles.

    Combines ``KaiserPTSpectrum2Poles`` components with linear bias ``b1`` and shot noise ``sn0``.
    For the matter (unbiased) power spectrum set b1=1 and sn0=0.

    For cross-spectra between tracers :math:`X` and :math:`Y` the model is
    :math:`b_1^X b_1^Y P_{dd} + (b_1^X + b_1^Y) P_{d\theta} + P_{\theta\theta} + s_n`.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    pt : KaiserPTSpectrum2Poles, default=None
        Matter PT module. A default instance is created if None.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders.
    template : template calculator, default=None
        Passed to the pt module if provided.
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3]. ``sn0`` parameter is in units of ``1/nbar``.
    tracers : str, (str, str), or None, default=None
        Tracer namespacing of the bias parameters (auto, namespaced auto, or cross).
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
                      ref=dict(limits=[1., 2.]), latex='b_1'),
            Parameter('sn0', value=0., prior=dict(dist='norm', loc=0., scale=1000.),
                      ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,0}'),
        ], tracers, stochastic=('sn0',), cross=True)

    def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, nbar=1e-4, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = KaiserPTSpectrum2Poles(**kwargs)
        self.pt = pt
        self.pt.update(k=self.k, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, nbar=1e-4, tracers=None, params=None, **kwargs):
        # Non-node setup only.
        self._nbar = float(nbar)

    def __call__(self):
        sn = jnp.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * self.sn0.value / self._nbar
        pk_dd, pk_dt, pk_tt = self.pt.table['pk_dd'], self.pt.table['pk_dt'], self.pt.table['pk_tt']
        if isinstance(self.b1, tuple):
            b1_X, b1_Y = self.b1
            self.poles = b1_X * b1_Y * pk_dd + (b1_X + b1_Y) * pk_dt + pk_tt + sn
        else:
            self.poles = self.b1**2 * pk_dd + 2. * self.b1 * pk_dt + pk_tt + sn
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class KaiserTracerCorrelation2Poles(Calculator):
    r"""
    Kaiser tracer correlation function multipoles via FFTLog.

    propose_params delegates to :class:`KaiserTracerSpectrum2Poles`.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h]. Defaults to np.linspace(20., 200., 181).
    pt : KaiserTracerSpectrum2Poles, default=None
        Tracer spectrum module. A default instance is created if None.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders.
    template : template calculator, default=None
        Passed to the pt module if provided.
    """

    @classmethod
    def propose_params(cls, tracers=None, **kwargs):
        """Delegate to :meth:`KaiserTracerSpectrum2Poles.propose_params`."""
        return KaiserTracerSpectrum2Poles.propose_params(tracers=tracers, **kwargs)

    def __init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, tracers=None, params=None, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        kin = np.geomspace(1e-4, 0.6, 300)
        if pt is None:
            pt = KaiserTracerSpectrum2Poles(tracers=tracers, params=params, **kwargs)
        self.pt = pt
        self.pt.update(k=kin, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, tracers=None, params=None, **kwargs):
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


# ── TNS model ─────────────────────────────────────────────────────────────────

_TNS_TABLE_NAMES = ['pk11', 'pk_dd', 'pk_b2d', 'pk_bs2d', 'pk_sig3sq',
                      'pk_b22', 'pk_b2s2', 'pk_bs22', 'pk_dt', 'pk_b2t', 'pk_bs2t', 'pk_tt', 'A', 'B']


class TNSPTSpectrum2Poles(Calculator):
    r"""
    TNS 1-loop matter power spectrum multipoles.

    Implements the model of Taruya, Nishimichi & Saito 2010 (arXiv:0912.0244).
    TNS loop kernels are precomputed at compile time (``__post_init__``).

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc]. Defaults to np.linspace(0.01, 0.2, 101).
    template : template calculator, default=None
        Power spectrum template. A default ``DirectSpectrum2Template()`` is created if None.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders to compute.
    mu : int, default=8
        Number of Gauss-Legendre mu-bins in [0, 1].
    fog : str, default='lorentzian'
        Finger-of-God damping kernel: 'lorentzian' or 'gaussian'.
    """

    @classmethod
    def propose_params(cls, tracers=None):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory."""
        return propose_params_multitracer([
            Parameter('sigmav', value=3., prior=dict(dist='norm', loc=0., scale=20., limits=[0., 10.]),
                      ref=dict(dist='norm', loc=0., scale=0.5), fd_eps=2., latex=r'\sigma_v'),
        ], tracers)

    def __init__(self, k=None, template=None, ells=(0, 2, 4), mu=8, fog='lorentzian', tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        kin = np.geomspace(1e-3, max(2., self.k[-1] * 2.), 500)
        self.template.update(k=kin)

    def __post_init__(self, k=None, template=None, ells=(0, 2, 4), mu=8, fog='lorentzian', tracers=None, params=None, **kwargs):
        # Non-node setup only (the template node already ran __post_init__ via update()).
        self._fog = str(fog)
        q = self.template.k
        wq = _weights_trapz(q)
        self._k11 = np.linspace(self.k[0] * 0.7, self.k[-1] * 1.3, int(len(self.k) * 1.6 + 0.5))
        self._q = q
        self._wq = wq
        self._kernels = tns_kernels(self._k11, q, wq)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        self._mu = self._to_poles.mu

    def __call__(self):
        k = self.k[:, None]
        mu = self._mu
        jac, kap, muap = self.template.ap_k_mu(k, mu)
        f = self.template.f
        if self._fog == 'lorentzian':
            damping = 1. / (1. + (self.sigmav * kap * muap)**2 / 2.)**2.
        else:
            damping = jnp.exp(-(self.sigmav * kap * muap)**2)

        tns_result = tns_pt(self._k11, self._q, self._wq, self.template.pk_dd, *self._kernels)
        table = jnp.concatenate([x[None, :] for x in tns_result[:-2]] + tns_result[-2:], axis=0)
        # table shape: (29, n_k11); interpolate and apply AP + FoG.
        kap_flat = jnp.log10(jnp.ravel(kap))
        table_interp = interpax.interp1d(kap_flat, jnp.log10(self._k11), table.T, method='cubic', extrap=True)
        table = jac * damping * jnp.moveaxis(jnp.reshape(table_interp, kap.shape + (29,)), [0, 1], [1, 2])
        # table shape: (29, n_k, n_mu)

        A_raw = table[12:17]   # (5, n_k, n_mu)
        B_raw = table[17:]     # (12, n_k, n_mu)
        A = jnp.stack([f * A_raw[0] * muap**2,
                       f**2 * (A_raw[1] * muap**2 + A_raw[2] * muap**4),
                       f**3 * (A_raw[3] * muap**4 + A_raw[4] * muap**6)])
        B = jnp.stack([f**2 * (B_raw[0] * muap**2 + B_raw[4] * muap**4),
                       -f**3 * ((B_raw[1] + B_raw[2]) * muap**2 + (B_raw[5] + B_raw[6]) * muap**4 + (B_raw[8] + B_raw[9]) * muap**6),
                       f**4 * (B_raw[3] * muap**2 + B_raw[7] * muap**4 + B_raw[10] * muap**6 + B_raw[11] * muap**8)])

        group1 = self._to_poles(table[:8, None])                        # (8, n_ells, n_k)
        group2 = self._to_poles(f * muap**2 * table[8:11, None])        # (3, n_ells, n_k)
        group3 = self._to_poles(f**2 * muap**4 * table[11:12, None])   # (1, n_ells, n_k)
        A_poles = self._to_poles(A[:, None, :, :])                        # (3, n_ells, n_k)
        B_poles = self._to_poles(B[:, None, :, :])                        # (3, n_ells, n_k)

        self.table = {}
        for pk in group1: self.table[_TNS_TABLE_NAMES[len(self.table)]] = pk
        for pk in group2: self.table[_TNS_TABLE_NAMES[len(self.table)]] = pk
        for pk in group3: self.table[_TNS_TABLE_NAMES[len(self.table)]] = pk
        self.table['A'] = A_poles
        self.table['B'] = B_poles

    def tree_flatten(self):
        return [self.table[n] for n in _TNS_TABLE_NAMES], {'k': self.k, 'ells': self.ells}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.table = dict(zip(_TNS_TABLE_NAMES, children))
        obj.k = aux['k']
        obj.ells = aux['ells']
        return obj


class TNSTracerSpectrum2Poles(Calculator):
    r"""
    TNS tracer power spectrum multipoles.

    Combines ``TNSPTSpectrum2Poles`` components with a full 1-loop bias expansion
    (b1, b2, bs, b3) plus shot noise.
    For the matter (unbiased) power spectrum set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    pt : TNSPTSpectrum2Poles, default=None
        Matter PT module. A default instance is created if None.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders.
    template : template calculator, default=None
        Passed to the pt module if provided.
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3]. ``sn0`` parameter is in units of ``1/nbar``.
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
                      ref=dict(limits=[1., 2.]), latex='b_1'),
            Parameter('b2', value=0., prior=dict(dist='norm', loc=0., scale=15.),
                      ref=dict(dist='norm', loc=0., scale=0.5), latex='b_2'),
            Parameter('bs', value=0., fixed=True, prior=dict(dist='norm', loc=0., scale=15.),
                      ref=dict(dist='norm', loc=0., scale=0.5), latex='b_s'),
            Parameter('b3', value=0., fixed=True, latex='b_3'),
            Parameter('sn0', value=0., prior=dict(dist='norm', loc=0., scale=1000.),
                      ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,0}'),
            Parameter('sigmav', value=3., prior=dict(dist='norm', loc=0., scale=20., limits=[0., 10.]),
                      ref=dict(dist='norm', loc=0., scale=0.5), fd_eps=2., latex=r'\sigma_v'),
        ], tracers)

    def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, nbar=1e-4, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers)
        if params is not None:
            vc = vc + VariableCollection(params)
        # sigmav is owned by the PT; separate it and route to PT.
        sigmav_vc = vc.select(basename='sigmav')
        assign_params(self, vc - sigmav_vc, tracers)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = TNSPTSpectrum2Poles(tracers=tracers, params=sigmav_vc if len(sigmav_vc) else None, **kwargs)
        self.pt = pt
        self.pt.update(k=self.k, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, nbar=1e-4, tracers=None, params=None, **kwargs):
        # Non-node setup only.
        self._nbar = float(nbar)

    def __call__(self):
        b1, b2, bs, b3 = self.b1, self.b2, self.bs, self.b3
        bs2 = bs - 4. / 7. * (b1 - 1.)
        b3nl = b3 + 32. / 315. * (b1 - 1.)
        sn = jnp.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * self.sn0.value / self._nbar
        self.poles = (b1**2 * self.pt.table['pk_dd'] + 2. * b1 * self.pt.table['pk_dt']
                      + self.pt.table['pk_tt'] + sn)
        self.poles += (2 * b1 * b2 * self.pt.table['pk_b2d'] + 2. * b1 * bs2 * self.pt.table['pk_bs2d']
                       + 2 * b1 * b3nl * self.pt.table['pk_sig3sq'] + b2**2 * self.pt.table['pk_b22']
                       + 2 * b2 * bs2 * self.pt.table['pk_b2s2'] + bs2**2 * self.pt.table['pk_bs22']
                       + b2 * self.pt.table['pk_b2t'] + b3nl * self.pt.table['pk_sig3sq'])
        self.poles += b1**2 * (self.pt.table['A'][0] + self.pt.table['B'][0])
        self.poles += b1 * (self.pt.table['A'][1] + self.pt.table['B'][1])
        self.poles += self.pt.table['A'][2] + self.pt.table['B'][2]
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class TNSTracerCorrelation2Poles(Calculator):
    r"""
    TNS tracer correlation function multipoles via FFTLog.

    The FFTLog Hankel transform is linear, so a transformation matrix is precomputed
    at compile time and applied as a JAX einsum in ``__call__``.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h]. Defaults to np.linspace(20., 200., 181).
    pt : TNSTracerSpectrum2Poles, default=None
        Tracer spectrum module. A default instance is created if None.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders.
    template : template calculator, default=None
        Passed to the pt module if provided.
    """

    @classmethod
    def propose_params(cls, tracers=None, **kwargs):
        """Delegate to :meth:`TNSTracerSpectrum2Poles.propose_params`."""
        return TNSTracerSpectrum2Poles.propose_params(tracers=tracers, **kwargs)

    def __init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, tracers=None, params=None, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        kin = np.geomspace(1e-4, 0.6, 300)
        if pt is None:
            pt = TNSTracerSpectrum2Poles(tracers=tracers, params=params, **kwargs)
        self.pt = pt
        self.pt.update(k=kin, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, tracers=None, params=None, **kwargs):
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


class LPTVelocileptorsPTSpectrum2Poles(Calculator):
    r"""
    Velocileptors LPT matter power spectrum multipoles (non-JAX).

    Wraps ``velocileptors.LPT.lpt_rsd_fftw.LPT_RSD``.
    Exposes ``table`` (shape ``(n_ells, n_k, 19)``), ``sigma8``, ``fsigma8``.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    template : DirectSpectrum2Template, default=None
    ells : tuple of int, default=(0, 2, 4)
    mu : int, default=4
        Gauss-Legendre mu order for AP integration.
    **kwargs :
        Velocileptors options: ``use_Pzel``, ``kIR``, ``cutoff``, ``extrap_min``, ``extrap_max``, ``N``, ``jn``, ``nthreads``.
    """

    _is_external = True
    _lpt_defaults = dict(use_Pzel=False, kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, jn=5)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/sfschen/velocileptors')

    def __init__(self, k=None, template=None, ells=(0, 2, 4), mu=4, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if k is None:
            k = _velocileptors_kvec(np.linspace(0.01, 0.5, 200))
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        self.template.update(k=np.geomspace(min(1e-4, self.k[0] / 2.), max(2., self.k[-1] * 2.), 500))

    def __post_init__(self, k=None, template=None, ells=(0, 2, 4), mu=4, **kwargs):
        # Non-node setup only.
        self.nmu = int(mu)
        self._options = {name: kwargs.get(name, val) for name, val in self._lpt_defaults.items()}
        self._options['threads'] = get_nthreads(kwargs.get('nthreads', None))

    def __call__(self):
        from scipy.interpolate import interp1d as _interp1d
        from velocileptors.LPT import lpt_rsd_fftw
        lpt_rsd_fftw.interp1d = lambda x, y: _interp1d(x, y, kind='cubic', assume_sorted=True)
        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        pt = LPT_RSD(np.asarray(self.template.k), np.asarray(self.template.pk_dd), **self._options)
        pt.make_pltable(float(self.template.f), kv=np.asarray(self.k),
                        apar=float(self.template.qpar), aperp=float(self.template.qper), ngauss=self.nmu)
        pktable = {0: pt.p0ktable, 2: pt.p2ktable, 4: pt.p4ktable}
        self.table = np.array([pktable[ell] for ell in self.ells])  # (n_ells, n_k, 19)
        self.qpar = float(self.template.qpar)
        self.qper = float(self.template.qper)
        self.sigma8 = float(self.template.sigma8)
        self.fsigma8 = float(self.template.fsigma8)
        self.sigma8_fid = float(self.template.sigma8_fid)

    def tree_flatten(self):
        return ([jnp.asarray(self.table), jnp.asarray(self.qpar), jnp.asarray(self.qper),
                 jnp.asarray(self.sigma8), jnp.asarray(self.fsigma8), jnp.asarray(self.sigma8_fid)],
                {'k': self.k, 'ells': self.ells})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.table, obj.qpar, obj.qper, obj.sigma8, obj.fsigma8, obj.sigma8_fid = children
        obj.k = aux['k']
        obj.ells = aux['ells']
        return obj


class LPTVelocileptorsTracerSpectrum2Poles(Calculator):
    r"""
    Velocileptors LPT tracer power spectrum multipoles.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    pt : LPTVelocileptorsPTSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    prior_basis : str, default='physical'
        ``'physical'``: parameters ``b1p, b2p, bsp, b3p, alpha0p, ..., sn0p, sn2p, sn4p``.
        Otherwise: ``b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4``.
    fsat : float, default=None
        Satellite fraction for the physical stochastic terms.  Defaults to
        ``get_physical_stochastic_settings()['fsat']``.  Pass the output of
        :func:`get_physical_stochastic_settings` directly for a specific tracer.
    sigv : float, default=None
        Velocity dispersion for the physical stochastic terms.  Defaults to
        ``get_physical_stochastic_settings()['sigv']``.
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3]. Stochastic terms are in units of ``1/nbar``.
    """

    @classmethod
    def propose_params(cls, tracers=None, prior_basis='physical'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        prior_basis : str, default='physical'

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer(_velocileptors_default_params(prior_basis), tracers)

    def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical',
                 fsat=None, sigv=None, nbar=1e-4, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers, prior_basis=prior_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        physical = prior_basis == 'physical'
        assign_params(self, vc, tracers, mapping=(lambda name: name[:-1]) if physical else None)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = LPTVelocileptorsPTSpectrum2Poles(**kwargs)
        self.pt = pt
        self.pt.update(k=_velocileptors_kvec(self.k), ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical', fsat=None, sigv=None, nbar=1e-4, tracers=None, **kwargs):
        # Non-node setup only.
        self._prior_basis = prior_basis
        self._nbar = float(nbar)
        settings = get_physical_stochastic_settings()
        self._fsat = float(fsat) if fsat is not None else settings['fsat']
        self._sigv = float(sigv) if sigv is not None else settings['sigv']

    def __call__(self):
        if self._prior_basis == 'standard':
            pars = jnp.array([self.b1, self.b2, self.bs, self.b3,
                               self.alpha0, self.alpha2, self.alpha4, self.alpha6,
                               self.sn0, self.sn2, self.sn4])
        else:
            f = self.pt.fsigma8 / self.pt.sigma8
            A = self.pt.sigma8 / self.pt.sigma8_fid
            qpar = self.pt.qpar
            qper = self.pt.qper
            A_AP = 1. / (qper**2 * qpar) if 'aap' in self._prior_basis else 1.
            pars = _velocileptors_physical_to_standard(self.b1, self.b2, self.bs, self.b3,
                                                  self.alpha0, self.alpha2, self.alpha4, self.alpha6,
                                                  self.sn0, self.sn2, self.sn4,
                                                  f, self._fsat, self._sigv, self._nbar, A=A, A_AP=A_AP, rept=False)
        raw = _velocileptors_combine_bias_terms_spectrum2_poles(self.pt.table, pars, nd=1.)  # (n_ells, n_k_pt)
        self.poles = interpax.interp1d(self.k, self.pt.k, raw.T, method='cubic', extrap=True).T
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class LPTVelocileptorsTracerCorrelation2Poles(Calculator):
    r"""
    Velocileptors LPT tracer correlation function multipoles via FFTLog.

    Parameters
    ----------
    s : array, default=None
        Output separations [Mpc/h].
    pt : LPTVelocileptorsTracerSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    prior_basis : str, default='physical'
    """

    @classmethod
    def propose_params(cls, tracers=None, **kwargs):
        """Delegate to :meth:`LPTVelocileptorsTracerSpectrum2Poles.propose_params`."""
        return LPTVelocileptorsTracerSpectrum2Poles.propose_params(tracers=tracers, **kwargs)

    def __init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical', tracers=None, params=None, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        kin = np.geomspace(1e-4, 0.6, 300)
        if pt is None:
            pt = LPTVelocileptorsTracerSpectrum2Poles(prior_basis=prior_basis, tracers=tracers, params=params, **kwargs)
        self.pt = pt
        self.pt.update(k=kin, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical', tracers=None, params=None, **kwargs):
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


class REPTVelocileptorsPTSpectrum2Poles(Calculator):
    r"""
    Velocileptors REPT matter power spectrum multipoles (non-JAX).

    Wraps ``velocileptors.EPT.ept_fullresum_varyDz_nu_fftw.REPT``.
    Exposes ``table`` (shape ``(n_ells, n_k, 19)``), ``sigma8``, ``fsigma8``.

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers [h/Mpc].
    template : DirectSpectrum2Template, default=None
    ells : tuple of int, default=(0, 2, 4)
    mu : int, default=4
    **kwargs :
        REPT options: ``rbao``, ``sbao``, ``beyond_gauss``, ``one_loop``, ``shear``, ``cutoff``, ``jn``, ``N``, ``extrap_min``, ``extrap_max``, ``import_wisdom``, ``nthreads``.
    """

    _is_external = True
    _rept_defaults = dict(rbao=110, sbao=None, beyond_gauss=True, one_loop=True, shear=True, cutoff=20, jn=5, N=4000, extrap_min=-4, extrap_max=3, import_wisdom=False)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/sfschen/velocileptors')

    def __init__(self, k=None, template=None, ells=(0, 2, 4), mu=4, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if k is None:
            k = _velocileptors_kvec(np.linspace(0.01, 0.5, 200))
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        self.template.update(with_now='peakaverage')
        self.template.update(k=np.geomspace(min(1e-4, self.k[0] / 2.), max(2., self.k[-1] * 2.), 500))

    def __post_init__(self, k=None, template=None, ells=(0, 2, 4), mu=4, **kwargs):
        # Non-node setup only.
        self.nmu = int(mu)
        self._options = {name: kwargs.get(name, val) for name, val in self._rept_defaults.items()}
        self._options['threads'] = get_nthreads(kwargs.get('nthreads', None))

    def __call__(self):
        from scipy.interpolate import interp1d as _interp1d
        from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT
        pk_dd = np.asarray(self.template.pk_dd)
        pknow_dd = np.asarray(self.template.pknow_dd)
        opts = {k: v for k, v in self._options.items() if v is not None}
        pt = REPT(np.asarray(self.template.k), pk_dd, pnw=pknow_dd, kmin=self.k[0], kmax=self.k[-1], nk=200, **opts)
        log10_ktempl = np.log10(np.asarray(self.template.k))
        log10_fk = np.log10(np.clip(np.asarray(self.template.fk), 1e-30, None))
        fk = 10.**_interp1d(log10_ktempl, log10_fk, kind='cubic', fill_value='extrapolate', assume_sorted=True)(np.log10(pt.kv))
        pks = pt.compute_redshift_space_power_multipoles_tables(fk, apar=float(self.template.qpar), aperp=float(self.template.qper), ngauss=self.nmu)[1:]
        pktable_kv = np.array([pks[list([0, 2, 4]).index(ell)] for ell in self.ells])  # (n_ells, n_kv, 19)
        self.table = _interp1d(pt.kv, pktable_kv, kind='cubic', fill_value='extrapolate', axis=1, assume_sorted=True)(self.k)
        self.qpar = float(self.template.qpar)
        self.qper = float(self.template.qper)
        self.sigma8 = float(self.template.sigma8)
        self.fsigma8 = float(self.template.fsigma8)
        self.sigma8_fid = float(self.template.sigma8_fid)

    def tree_flatten(self):
        return ([self.table, self.qpar, self.qper, self.sigma8, self.fsigma8, self.sigma8_fid],
                {'k': self.k, 'ells': self.ells})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.table, obj.qpar, obj.qper, obj.sigma8, obj.fsigma8, obj.sigma8_fid = children
        obj.k = aux['k']
        obj.ells = aux['ells']
        return obj


class REPTVelocileptorsTracerSpectrum2Poles(Calculator):
    r"""
    Velocileptors REPT tracer power spectrum multipoles.

    Differs from LPT in the physical-prior bias conversion and co-evolution correction applied to ``bs``/``b3``.

    Parameters
    ----------
    k : array, default=None
    pt : REPTVelocileptorsPTSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    prior_basis : str, default='physical'
    fsat, sigv, nbar : same as LPTVelocileptorsTracerSpectrum2Poles.
    """

    @classmethod
    def propose_params(cls, tracers=None, prior_basis='physical'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        prior_basis : str, default='physical'

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer(_velocileptors_default_params(prior_basis), tracers)

    def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical',
                 fsat=None, sigv=None, nbar=1e-4, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers, prior_basis=prior_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        physical = prior_basis == 'physical'
        assign_params(self, vc, tracers, mapping=(lambda name: name[:-1]) if physical else None)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = REPTVelocileptorsPTSpectrum2Poles(**kwargs)
        self.pt = pt
        self.pt.update(k=_velocileptors_kvec(self.k), ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical', fsat=None, sigv=None, nbar=1e-4, tracers=None, **kwargs):
        # Non-node setup only.
        self._prior_basis = prior_basis
        self._nbar = float(nbar)
        settings = get_physical_stochastic_settings()
        self._fsat = float(fsat) if fsat is not None else settings['fsat']
        self._sigv = float(sigv) if sigv is not None else settings['sigv']

    def __call__(self):
        if self._prior_basis == 'standard':
            pars = jnp.array([self.b1, self.b2, self.bs - (2. / 7.) * (self.b1 - 1.), 3. * self.b3 + (self.b1 - 1.),
                               self.alpha0, self.alpha2, self.alpha4, self.alpha6,
                               self.sn0, self.sn2, self.sn4])
        else:
            f = self.pt.fsigma8 / self.pt.sigma8
            A = self.pt.sigma8 / self.pt.sigma8_fid
            qpar = self.pt.qpar
            qper = self.pt.qper
            A_AP = 1. / (qper**2 * qpar) if 'aap' in self._prior_basis else 1.
            pars = _velocileptors_physical_to_standard(self.b1, self.b2, self.bs, self.b3,
                                                  self.alpha0, self.alpha2, self.alpha4, self.alpha6,
                                                  self.sn0, self.sn2, self.sn4,
                                                  f, self._fsat, self._sigv, self._nbar, A=A, A_AP=A_AP, rept=True)
        raw = _velocileptors_combine_bias_terms_spectrum2_poles(self.pt.table, pars, nd=1.)
        self.poles = interpax.interp1d(self.k, self.pt.k, raw.T, method='cubic', extrap=True).T
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class REPTVelocileptorsTracerCorrelation2Poles(Calculator):
    r"""
    Velocileptors REPT tracer correlation function multipoles via FFTLog.

    Parameters
    ----------
    s : array, default=None
    pt : REPTVelocileptorsTracerSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    prior_basis : str, default='physical'
    """

    @classmethod
    def propose_params(cls, tracers=None, **kwargs):
        """Delegate to :meth:`REPTVelocileptorsTracerSpectrum2Poles.propose_params`."""
        return REPTVelocileptorsTracerSpectrum2Poles.propose_params(tracers=tracers, **kwargs)

    def __init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical', tracers=None, params=None, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        kin = np.geomspace(1e-4, 0.6, 300)
        if pt is None:
            pt = REPTVelocileptorsTracerSpectrum2Poles(prior_basis=prior_basis, tracers=tracers, params=params, **kwargs)
        self.pt = pt
        self.pt.update(k=kin, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical', tracers=None, params=None, **kwargs):
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


class PyBirdPTSpectrum2Poles(Calculator):
    r"""
    PyBird matter power spectrum multipoles (non-JAX).

    Wraps ``pybird.bird.Bird`` + pybird loop integrals.
    Exposes ``P11l``, ``Ploopl``, ``Pctl``, ``Pstl``, ``Pnnlol`` arrays and metadata.

    Parameters
    ----------
    k : array, default=None
    template : DirectSpectrum2Template, default=None
    ells : tuple of int, default=(0, 2, 4)
    km, kr : float, default=0.7, 0.25
    accboost, fftaccboost : int, default=1
    fftbias : float, default=-1.6
    with_nnlo_counterterm : bool, default=False
    with_stoch : bool, default=True
    with_resum : str or bool, default='full'
    with_ap : bool, default=True
    eft_basis : str, default='eftoflss'
    """

    _is_external = True

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/pierrexyz/pybird')

    def __init__(self, k=None, template=None, ells=(0, 2, 4), km=0.7, kr=0.25,
                 accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False,
                 with_stoch=True, with_resum='full', with_ap=True, eft_basis='eftoflss', **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        if with_nnlo_counterterm:
            self.template.update(with_now='peakaverage')

    def __post_init__(self, k=None, template=None, ells=(0, 2, 4), km=0.7, kr=0.25,
                      accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False,
                      with_stoch=True, with_resum='full', with_ap=True, eft_basis='eftoflss', **kwargs):
        # Non-node setup only (pybird Common/NonLinear/Resum/Projection are not Nodes).
        self._with_stoch = bool(with_stoch)
        self._with_nnlo = bool(with_nnlo_counterterm)
        self._with_resum = with_resum
        self._with_ap = bool(with_ap)
        self.km = tuple(km) if hasattr(km, '__len__') else (float(km),) * 2
        self.kr = tuple(kr) if hasattr(kr, '__len__') else (float(kr),) * 2
        from pybird.common import Common
        from pybird.nonlinear import NonLinear
        from pybird.resum import Resum
        from pybird.projection import Projection
        eft = eft_basis if eft_basis not in (None, 'velocileptors') else 'eftoflss'
        if self.k[0] * 0.8 < 1e-3:
            import warnings
            warnings.warn('pybird does not predict P(k) for k < 0.001 h/Mpc; nan will be replaced by 0')
        self._co = Common(Nl=len(self.ells), kmin=1e-3, kmax=self.k[-1] * 1.3,
                          km=min(self.km), kr=min(self.kr), nd=1e-4, eft_basis=eft,
                          halohalo=True, with_cf=False, with_time=True,
                          accboost=float(accboost), optiresum=(with_resum == 'opti'),
                          with_uvmatch=False, exact_time=False, quintessence=False,
                          with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False)
        self._nonlinear = NonLinear(load=False, save=False, NFFT=256 * int(fftaccboost), fftbias=fftbias, co=self._co)
        self._resum = Resum(co=self._co)
        self._nnlo = None
        if with_nnlo_counterterm:
            from pybird.nnlo import NNLO_counterterm
            self._nnlo = NNLO_counterterm(co=self._co)
        self._projection = Projection(self.k, with_ap=with_ap, H_fid=None, D_fid=None, co=self._co)

    def __call__(self):
        from pybird.bird import Bird
        from scipy.interpolate import interp1d as _interp1d
        cosmo = {'kk': np.asarray(self.template.k), 'pk_lin': np.asarray(self.template.pk_dd),
                 'pk_lin_2': None, 'f': float(self.template.f), 'DA': 1., 'H': 1.}
        self._pt = Bird(cosmo, with_bias=False, eft_basis=self._co.eft_basis, with_stoch=self._with_stoch,
                        with_nnlo_counterterm=self._nnlo is not None, co=self._co)
        if self._nnlo is not None:
            self._nnlo.Ps(self._pt, _interp1d(np.log(np.asarray(self.template.k)),
                                               np.log(np.clip(np.asarray(self.template.pknow_dd), 1e-30, None)),
                                               fill_value='extrapolate', assume_sorted=True))
        self._nonlinear.PsCf(self._pt)
        self._pt.setPsCfl()
        if self._with_resum:
            self._resum.PsCf(self._pt, makeIR=True, makeQ=True, setIR=True, setPs=True, setCf=False)
        if self._with_ap:
            self._projection.AP(self._pt, q=(float(self.template.qper), float(self.template.qpar)))
        self._projection.xdata(self._pt)

    def tree_flatten(self):
        _z = jnp.zeros((len(self.ells), 1, len(self.k)))
        P11l = jnp.asarray(self._pt.P11l)
        Ploopl = jnp.asarray(self._pt.Ploopl)
        Pctl = jnp.asarray(self._pt.Pctl)
        Pstl = jnp.asarray(self._pt.Pstl) if self._with_stoch else _z
        Pnnlol = jnp.asarray(self._pt.Pnnlol) if self._with_nnlo else _z
        return ([P11l, Ploopl, Pctl, Pstl, Pnnlol],
                {'k': self.k, 'ells': self.ells, 'km': self.km, 'kr': self.kr,
                 'f': float(self._pt.f), 'eft_basis': self._pt.eft_basis,
                 'with_stoch': self._with_stoch, 'with_nnlo': self._with_nnlo, 'co': self._co})

    @classmethod
    def tree_unflatten(cls, aux, children):
        from pybird.bird import Bird
        obj = object.__new__(cls)
        pt = Bird.__new__(Bird)
        pt.P11l, pt.Ploopl, pt.Pctl, pt.Pstl, pt.Pnnlol = children
        pt.f = aux['f']
        pt.eft_basis = aux['eft_basis']
        pt.with_stoch = aux['with_stoch']
        pt.with_nnlo_counterterm = aux['with_nnlo']
        pt.with_bias = False
        pt.co = aux['co']
        pt.with_tidal_alignments = pt.co.with_tidal_alignments
        obj._pt = pt
        obj.k = aux['k']
        obj.ells = aux['ells']
        obj.km = aux['km']
        obj.kr = aux['kr']
        obj._with_stoch = aux['with_stoch']
        obj._with_nnlo = aux['with_nnlo']
        obj._co = aux['co']
        return obj


class PyBirdTracerSpectrum2Poles(Calculator):
    r"""
    PyBird tracer power spectrum multipoles.

    Parameters
    ----------
    k : array, default=None
    pt : PyBirdPTSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    eft_basis : str, default='eftoflss'
        One of ``'eftoflss'``, ``'westcoast'``, ``'eastcoast'``, ``'velocileptors'``.
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3].
    """

    @classmethod
    def _auto_params(cls, eft_basis):
        """Return default auto_params list for the given EFT basis (shared with Correlation variant)."""
        eft = eft_basis if eft_basis not in (None, 'velocileptors') else 'eftoflss'
        if eft in ('eftoflss', 'velocileptors'):
            bias = [
                Parameter('b1', value=1.6, prior=dict(limits=[0., 4.]), ref=dict(dist='norm', loc=1.6, scale=0.1), latex='b_1'),
                Parameter('b2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_2'),
                Parameter('b3', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_3'),
                Parameter('b4', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_4'),
            ]
        elif eft == 'westcoast':
            bias = [
                Parameter('b1', value=1.6, prior=dict(limits=[0., 4.]), ref=dict(dist='norm', loc=1.6, scale=0.1), latex='b_1'),
                Parameter('b2p4', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_{2+4}'),
                Parameter('b2m4', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_{2-4}'),
                Parameter('b3', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_3'),
            ]
        else:  # eastcoast
            bias = [
                Parameter('b1', value=1.6, prior=dict(limits=[0., 4.]), ref=dict(dist='norm', loc=1.6, scale=0.1), latex='b_1'),
                Parameter('b2t', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_{2t}'),
                Parameter('b2g', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_{2g}'),
                Parameter('b3g', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_{3g}'),
            ]
        return bias + [
            Parameter('cct', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='c_{ct}'),
            Parameter('cr1', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='c_{r1}'),
            Parameter('cr2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='c_{r2}'),
            Parameter('ce0', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r'\epsilon_0'),
            Parameter('ce1', value=0., fixed=True, latex=r'\epsilon_1'),
            Parameter('ce2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r'\epsilon_2'),
        ]

    @classmethod
    def propose_params(cls, tracers=None, eft_basis='eftoflss', **kwargs):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        eft_basis : str, default='eftoflss'

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer(cls._auto_params(eft_basis), tracers, stochastic=('ce0', 'ce1', 'ce2'), cross=True)

    def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, eft_basis='eftoflss', nbar=1e-4, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers, eft_basis=eft_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self._eft_basis = eft_basis if eft_basis not in (None, 'velocileptors') else 'eftoflss'
        if pt is None:
            pt = PyBirdPTSpectrum2Poles(**kwargs)
        self.pt = pt
        self.pt.update(k=self.k, ells=self.ells, eft_basis=self._eft_basis)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, eft_basis='eftoflss', nbar=1e-4, tracers=None, **kwargs):
        # Non-node setup only.
        self._nbar = float(nbar)

    def _build_params(self, idx=None):
        """Bias dict for pybird, with **raw** counterterms.

        pybird's ``setBias`` divides ``cct``/``cr1``/``cr2`` by ``co.km**2``/``co.kr**2``
        (and ``ce1``/``ce2`` by ``co.km**2``) once, so we pass the raw parameters here.

        For cross-spectra, ``idx`` in ``{0, 1}`` selects tracer X or Y from the
        tuple-valued (per-tracer) bias attributes; shared stochastic terms are scalars
        and are returned as-is.
        """
        def get(name):
            val = getattr(self, name)
            return val[idx] if (idx is not None and isinstance(val, tuple)) else val
        eft = self._eft_basis
        b1 = get('b1')
        if eft == 'westcoast':
            b2 = (get('b2p4') + get('b2m4')) / 2.**0.5
            b4 = (get('b2p4') - get('b2m4')) / 2.**0.5
            b3 = get('b3')
        elif eft == 'eastcoast':
            b2g, b2t, b3g = get('b2g'), get('b2t'), get('b3g')
            b2 = b1 + 7./2.*b2g
            b3 = b1 + 15.*b2g + 6.*b3g
            b4 = 0.5*b2t - 7./2.*b2g
        else:
            b2, b3, b4 = get('b2'), get('b3'), get('b4')
        if eft in ('eftoflss', 'velocileptors', 'westcoast'):
            return {'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4,
                    'cct': get('cct'), 'cr1': get('cr1'), 'cr2': get('cr2'),
                    'ce0': get('ce0'), 'ce1': get('ce1'), 'ce2': get('ce2')}
        return {'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4,
                'c0': get('cct'), 'c2': get('cr1'), 'c4': get('cr2'),
                'ce0': get('ce0'), 'ce1': get('ce1'), 'ce2': get('ce2')}

    def _fullps_cross(self, bird, biasX, biasY):
        r"""Cross power-spectrum multipoles for two tracers X, Y.

        Follows https://arxiv.org/abs/2308.06206 eq.(13): the shared matter loop
        tables (``bird.P11l``/``Ploopl``/``Pctl``/``Pstl``) are contracted with
        symmetric (X<->Y) bias vectors that reduce to the auto vectors when X == Y.
        Counterterms are divided by ``km**2``/``kr**2`` here (single division, as in
        :meth:`_build_params` the values are raw); stochastic terms are shared.
        """
        f = bird.f
        b1X, b2X, b3X, b4X = (biasX[f'b{i:d}'] for i in (1, 2, 3, 4))
        b1Y, b2Y, b3Y, b4Y = (biasY[f'b{i:d}'] for i in (1, 2, 3, 4))
        kmX, kmY = self.pt.km
        krX, krY = self.pt.kr
        if bird.eft_basis in ('eftoflss', 'westcoast'):
            b5X, b6X, b7X = (biasX[n] / ks**2 for n, ks in zip(('cct', 'cr1', 'cr2'), (kmX, krX, krX)))
            b5Y, b6Y, b7Y = (biasY[n] / ks**2 for n, ks in zip(('cct', 'cr1', 'cr2'), (kmY, krY, krY)))
            bct = jnp.array([b1X * b5Y + b1Y * b5X, b1Y * b6X + b1X * b6Y, b1Y * b7X + b1X * b7Y,
                             (b5X + b5Y) * f, (b6X + b6Y) * f, (b7X + b7Y) * f])
        else:  # eastcoast (inversion of eq. 2.23 of arXiv:2004.10607)
            ct0X = biasX['c0'] - f / 3. * biasX['c2'] + 3. / 35. * f**2 * biasX['c4']
            ct2X = biasX['c2'] - 6. / 7. * f * biasX['c4']
            ct4X = biasX['c4']
            ct0Y = biasY['c0'] - f / 3. * biasY['c2'] + 3. / 35. * f**2 * biasY['c4']
            ct2Y = biasY['c2'] - 6. / 7. * f * biasY['c4']
            ct4Y = biasY['c4']
            bct = -jnp.array([ct0X + ct0Y, f * (ct2X + ct2Y), f**2 * (ct4X + ct4Y)])
        if bird.with_nnlo_counterterm:
            raise NotImplementedError('PyBird cross-power spectrum with nnlo counterterm is not implemented.')
        b11 = jnp.array([b1X * b1Y, (b1X + b1Y) * f, f**2])
        bloop = jnp.array([1., 0.5 * (b1X + b1Y), 0.5 * (b2X + b2Y), 0.5 * (b3X + b3Y), 0.5 * (b4X + b4Y),
                           b1X * b1Y, 0.5 * (b1X * b2Y + b1Y * b2X), 0.5 * (b1X * b3Y + b1Y * b3X),
                           0.5 * (b1X * b4Y + b1Y * b4X), b2X * b2Y, 0.5 * (b2X * b4Y + b2Y * b4X), b4X * b4Y])
        Ps0 = jnp.einsum('b,lbx->lx', b11, bird.P11l)
        Ps1 = jnp.einsum('b,lbx->lx', bloop, bird.Ploopl) + jnp.einsum('b,lbx->lx', bct, bird.Pctl)
        if bird.with_stoch:
            # Match pybird's setBias: stochastic terms divided by co.nd (the number density).
            bst = jnp.array([biasX['ce0'], biasX['ce1'] / (kmX * kmY), biasX['ce2'] / (kmX * kmY)]) / bird.co.nd
            Ps1 = Ps1 + jnp.einsum('b,lbx->lx', bst, bird.Pstl)
        return jnp.nan_to_num(Ps0 + Ps1, nan=0., posinf=jnp.inf, neginf=-jnp.inf)

    def __call__(self):
        bird = self.pt._pt  # underlying pybird Bird (self.pt is the External wrapper)
        if isinstance(self.b1, tuple):  # cross-spectrum of two tracers
            self.poles = self._fullps_cross(bird, self._build_params(0), self._build_params(1))
        else:
            import pybird.bird as bird_module
            bird_module.np = jnp
            self._pt = bird
            bird.co.nbar = self._nbar
            bird.setreducePslb(self._build_params(), what='full')
            bird_module.np = np
            self.poles = jnp.nan_to_num(bird.fullPs, nan=0., posinf=jnp.inf, neginf=-jnp.inf)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class PyBirdPTCorrelation2Poles(Calculator):
    r"""
    PyBird matter correlation function multipoles (non-JAX).

    Parameters
    ----------
    s : array, default=None
    template : DirectSpectrum2Template, default=None
    ells : tuple of int, default=(0, 2, 4)
    km, kr, accboost, fftaccboost, fftbias, with_nnlo_counterterm, with_stoch, with_resum, with_ap, eft_basis : same as PyBirdPTSpectrum2Poles.
    """

    _is_external = True

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/pierrexyz/pybird')

    def __init__(self, s=None, template=None, ells=(0, 2, 4), km=0.7, kr=0.25,
                 accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False,
                 with_stoch=False, with_resum='full', with_ap=True, eft_basis='eftoflss', **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        if with_nnlo_counterterm:
            self.template.update(with_now='peakaverage')

    def __post_init__(self, s=None, template=None, ells=(0, 2, 4), km=0.7, kr=0.25,
                      accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False,
                      with_stoch=False, with_resum='full', with_ap=True, eft_basis='eftoflss', **kwargs):
        # Non-node setup only (pybird Common/NonLinear/Resum/Projection are not Nodes).
        self._with_stoch = bool(with_stoch)
        self._with_nnlo = bool(with_nnlo_counterterm)
        self._with_resum = with_resum
        self._with_ap = bool(with_ap)
        self.km = tuple(km) if hasattr(km, '__len__') else (float(km),) * 2
        self.kr = tuple(kr) if hasattr(kr, '__len__') else (float(kr),) * 2
        from pybird.common import Common
        from pybird.nonlinear import NonLinear
        from pybird.resum import Resum
        from pybird.projection import Projection
        eft = eft_basis if eft_basis not in (None, 'velocileptors') else 'eftoflss'
        self._co = Common(Nl=len(self.ells), kmin=1e-3, kmax=0.25, km=min(self.km), kr=min(self.kr), nd=1e-4,
                          eft_basis=eft, halohalo=True, with_cf=True, with_time=True,
                          accboost=float(accboost), optiresum=(with_resum == 'opti'),
                          with_uvmatch=False, exact_time=False, quintessence=False,
                          with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False)
        self._nonlinear = NonLinear(load=False, save=False, NFFT=256 * int(fftaccboost), fftbias=fftbias, co=self._co)
        self._resum = Resum(co=self._co)
        self._nnlo = None
        if with_nnlo_counterterm:
            from pybird.nnlo import NNLO_counterterm
            self._nnlo = NNLO_counterterm(co=self._co)
        self._projection = Projection(self.s, with_ap=with_ap, H_fid=None, D_fid=None, co=self._co)

    def __call__(self):
        from pybird.bird import Bird
        from scipy.interpolate import interp1d as _interp1d
        cosmo = {'kk': np.asarray(self.template.k), 'pk_lin': np.asarray(self.template.pk_dd),
                 'pk_lin_2': None, 'f': float(self.template.f), 'DA': 1., 'H': 1.}
        self._pt = Bird(cosmo, with_bias=False, eft_basis=self._co.eft_basis, with_stoch=self._with_stoch,
                        with_nnlo_counterterm=self._nnlo is not None, co=self._co)
        if self._nnlo is not None:
            self._nnlo.Cf(self._pt, _interp1d(np.log(np.asarray(self.template.k)),
                                               np.log(np.clip(np.asarray(self.template.pknow_dd), 1e-30, None)),
                                               fill_value='extrapolate', assume_sorted=True))
        self._nonlinear.PsCf(self._pt)
        self._pt.setPsCfl()
        if self._with_resum:
            self._resum.PsCf(self._pt, makeIR=True, makeQ=True, setIR=True, setPs=True, setCf=True)
        if self._with_ap:
            self._projection.AP(self._pt, q=(float(self.template.qper), float(self.template.qpar)))
        self._projection.xdata(self._pt)

    def tree_flatten(self):
        # Expose both Cf and Ps loop arrays: setreduceCflb ends with a call to
        # setreducePslb (NNLO bookkeeping), which needs the P-arrays even though
        # the tracer only reads fullCf.
        _zc = jnp.zeros((len(self.ells), 1, len(self.s)))
        C11l = jnp.asarray(self._pt.C11l)
        Cloopl = jnp.asarray(self._pt.Cloopl)
        Cctl = jnp.asarray(self._pt.Cctl)
        Cstl = jnp.asarray(self._pt.Cstl) if self._with_stoch else _zc
        Cnnlol = jnp.asarray(self._pt.Cnnlol) if self._with_nnlo else _zc
        P11l = jnp.asarray(self._pt.P11l)
        Ploopl = jnp.asarray(self._pt.Ploopl)
        Pctl = jnp.asarray(self._pt.Pctl)
        _zp = jnp.zeros((len(self.ells), 1, P11l.shape[-1]))
        Pstl = jnp.asarray(self._pt.Pstl) if self._with_stoch else _zp
        Pnnlol = jnp.asarray(self._pt.Pnnlol) if self._with_nnlo else _zp
        return ([C11l, Cloopl, Cctl, Cstl, Cnnlol, P11l, Ploopl, Pctl, Pstl, Pnnlol],
                {'s': self.s, 'ells': self.ells, 'km': self.km, 'kr': self.kr,
                 'f': float(self._pt.f), 'eft_basis': self._pt.eft_basis,
                 'with_stoch': self._with_stoch, 'with_nnlo': self._with_nnlo, 'co': self._co})

    @classmethod
    def tree_unflatten(cls, aux, children):
        from pybird.bird import Bird
        obj = object.__new__(cls)
        pt = Bird.__new__(Bird)
        (pt.C11l, pt.Cloopl, pt.Cctl, pt.Cstl, pt.Cnnlol,
         pt.P11l, pt.Ploopl, pt.Pctl, pt.Pstl, pt.Pnnlol) = children
        pt.f = aux['f']
        pt.eft_basis = aux['eft_basis']
        pt.with_stoch = aux['with_stoch']
        pt.with_nnlo_counterterm = aux['with_nnlo']
        pt.with_bias = False
        pt.co = aux['co']
        pt.with_tidal_alignments = pt.co.with_tidal_alignments
        obj._pt = pt
        obj.s = aux['s']
        obj.ells = aux['ells']
        obj.km = aux['km']
        obj.kr = aux['kr']
        obj._with_stoch = aux['with_stoch']
        obj._with_nnlo = aux['with_nnlo']
        obj._co = aux['co']
        return obj


class PyBirdTracerCorrelation2Poles(Calculator):
    r"""
    PyBird tracer correlation function multipoles.

    Parameters
    ----------
    s : array, default=None
    pt : PyBirdPTCorrelation2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    eft_basis : str, default='eftoflss'
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3].
    """

    @classmethod
    def propose_params(cls, tracers=None, eft_basis='eftoflss'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Cross-correlations are not supported for the correlation function; use a single tracer name.

        Parameters
        ----------
        tracers : str or None, default=None
        eft_basis : str, default='eftoflss'

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer(PyBirdTracerSpectrum2Poles._auto_params(eft_basis),
                                           tracers, stochastic=('ce0', 'ce1', 'ce2'))  # no cross

    def __init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, eft_basis='eftoflss', nbar=1e-4, tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers, eft_basis=eft_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, tracers)
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        self._eft_basis = eft_basis if eft_basis not in (None, 'velocileptors') else 'eftoflss'
        if pt is None:
            pt = PyBirdPTCorrelation2Poles(**kwargs)
        self.pt = pt
        self.pt.update(s=self.s, ells=self.ells, eft_basis=self._eft_basis)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, eft_basis='eftoflss', nbar=1e-4, tracers=None, **kwargs):
        # Non-node setup only.
        self._nbar = float(nbar)

    _build_params = PyBirdTracerSpectrum2Poles._build_params

    def __call__(self):
        import pybird.bird as bird_module
        bird_module.np = jnp
        self._pt = self.pt._pt  # underlying pybird Bird (self.pt is the External wrapper)
        self._pt.co.nbar = self._nbar
        self._pt.setreduceCflb(self._build_params(), what='full')
        bird_module.np = np
        self.poles = self._pt.fullCf
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class FOLPSPTSpectrum2Poles(Calculator):
    r"""
    FOLPS matter power spectrum multipoles.

    Wraps ``folps.NonLinearPowerSpectrumCalculator`` loop tables.
    Exposes ``kap``, ``muap``, ``jac``, ``table``, ``table_now``,
    ``f``, ``f0``, ``qpar``, ``qper``, ``sigma8``, ``fsigma8``.

    Parameters
    ----------
    k : array, default=None
    template : DirectSpectrum2Template, default=None
    ells : tuple of int, default=(0, 2, 4)
    mu : int, default=6
    kernels : str, default='fk'
    rbao : float, default=104.
    A_full : bool, default=True
    remove_DeltaP : bool, default=False
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/FolpsD')

    def __init__(self, k=None, template=None, ells=(0, 2, 4), mu=6, kernels='fk', rbao=104., A_full=True, remove_DeltaP=False, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if template is None:
            template = DirectSpectrum2Template()
        self.template = template
        self.template.update(with_now='peakaverage')

    def __post_init__(self, k=None, template=None, ells=(0, 2, 4), mu=6, kernels='fk', rbao=104., A_full=True, remove_DeltaP=False, **kwargs):
        # Non-node setup only.
        self._kernels = str(kernels)
        self._rbao = float(rbao)
        self._A_full = bool(A_full)
        self._remove_DeltaP = bool(remove_DeltaP)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        os.environ.setdefault('FOLPS_BACKEND', 'jax')
        import folps as folpsv2
        self._matrices = folpsv2.MatrixCalculator(A_full=A_full, use_TNS_model=remove_DeltaP).get_mmatrices()

    def __call__(self):
        import folps as folpsv2
        cosmo_params = {'pkttlin': self.template.pk_dd * self.template.fk**2,
                        'f0': self.template.f0}
        folps_nlps = folpsv2.NonLinearPowerSpectrumCalculator(
            mmatrices=self._matrices, kernels=self._kernels, rbao=self._rbao, **cosmo_params)
        table, table_now = folps_nlps.calculate_loop_table(
            k=self.template.k, pklin=self.template.pk_dd,
            pknow=self.template.pknow_dd, **cosmo_params)
        jac, kap, muap = self.template.ap_k_mu(self.k[:, None], self._to_poles.mu)
        self.kap = kap
        self.muap = muap
        self.jac = jac
        # FOLPS returns mixed shapes: most loop terms are (nk_table,) arrays but the
        # trailing entries (sigma2w, f0, and the NW sigma2/delta_sigma2) are scalars,
        # which interp_table passes through unchanged.  Keep them as a tuple of
        # per-element arrays (0-d for scalars) rather than a single rectangular array.
        self.table = tuple(jnp.asarray(t) for t in table)
        self.table_now = tuple(jnp.asarray(t) for t in table_now)
        self.f = self.template.f
        self.f0 = self.template.f0
        self.qpar = self.template.qpar
        self.qper = self.template.qper
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.fsigma8
        self.sigma8_fid = self.template.sigma8_fid

    def tree_flatten(self):
        # table / table_now are tuples of per-element arrays; flatten each element
        # as a separate child so JAX preserves their individual shapes.
        table = list(self.table)
        table_now = list(self.table_now)
        children = ([self.kap, self.muap, self.jac] + table + table_now
                    + [self.f, self.f0, self.qpar, self.qper, self.sigma8, self.fsigma8, self.sigma8_fid])
        aux = {'k': self.k, 'ells': self.ells, 'mu': self._to_poles.mu, 'wmu': self._to_poles.wmu,
               'A_full': self._A_full, 'remove_DeltaP': self._remove_DeltaP,
               'n_table': len(table), 'n_table_now': len(table_now)}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        it = iter(children)
        obj.kap = next(it)
        obj.muap = next(it)
        obj.jac = next(it)
        obj.table = tuple(next(it) for _ in range(aux['n_table']))
        obj.table_now = tuple(next(it) for _ in range(aux['n_table_now']))
        obj.f = next(it)
        obj.f0 = next(it)
        obj.qpar = next(it)
        obj.qper = next(it)
        obj.sigma8 = next(it)
        obj.fsigma8 = next(it)
        obj.sigma8_fid = next(it)
        obj.k = aux['k']
        obj.ells = aux['ells']
        obj._A_full = aux['A_full']
        obj._remove_DeltaP = aux['remove_DeltaP']
        obj._to_poles = ProjectToPoles.__new__(ProjectToPoles)
        obj._to_poles.mu = aux['mu']
        obj._to_poles.wmu = aux['wmu']
        obj._to_poles.ells = aux['ells']
        return obj

    def combine_bias_terms_spectrum2_poles(self, pars, bias_scheme, damping):
        """Evaluate power-spectrum multipoles for *pars*.

        Reads only from attributes set by ``__call__`` (or ``tree_unflatten`` when
        emulated) — no access to ``self.template``.
        """
        # For emulator
        os.environ.setdefault('FOLPS_BACKEND', 'jax')
        import folps as folpsv2
        import folps.folps as _folps_module
        _folps_module.A_full_status = self._A_full
        _folps_module.use_TNS_model_status = self._remove_DeltaP
        folps_rsdmps = folpsv2.RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
        pars = folps_rsdmps.set_bias_scheme(pars=pars, bias_scheme=bias_scheme)
        pkmu = self.jac * folps_rsdmps.get_rsd_pkmu(self.kap, self.muap, pars, tuple(self.table), tuple(self.table_now), IR_resummation=True, damping=damping)
        return self._to_poles(pkmu)

    def combine_bias_terms_spectrum3_poles(self, pars, k1k2, multipoles, **options):
        """Evaluate bispectrum multipoles for *pars*.

        Builds the ``[k, pk_lin, pk_lin_now, fk]`` input from ``self.table``/``self.table_now``
        — no access to ``self.template``, so this works with emulated calculators.
        ``self.table[0]`` is the loop k-grid; ``self.table[1]`` is pk_lin;
        ``self.table_now[1]`` is pk_lin_now; ``self.table[2] * self.f0`` is fk.
        """
        # For emulator
        os.environ.setdefault('FOLPS_BACKEND', 'jax')
        k_pkl_pklnw_fk = jnp.array([self.table[0], self.table[1], self.table_now[1], self.table[2] * self.f0])
        return _get_spectrum3poles_folps(pars, k1k2, k_pkl_pklnw_fk, self.f0, self.qpar, self.qper, multipoles=multipoles, **options)


class FOLPSTracerSpectrum2Poles(Calculator):
    r"""
    FOLPS tracer power spectrum multipoles.

    Parameters
    ----------
    k : array, default=None
    pt : FOLPSPTSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    prior_basis : str, default='physical_aap'
        Bias / counterterm / stochastic parameterization (mirrors FOLPSv2 in desilike_bak):

        - ``'standard'``: standard Eulerian bias as in the FOLPS paper (arXiv:2404.07269);
          parameters ``b1, b2, bs, b3, alpha0, alpha2, alpha4, ct, sn0, sn2, X_FoG``.
        - ``'physical'``: physical (velocileptors-DR1) Lagrangian basis, **no** AP rescaling;
          parameters carry a ``p`` suffix (``b1p, b2p, ...``).
        - ``'physical_aap'`` (default): physical basis with AP rescaling (2pt3pt prior document).
        - ``'tcm_chudaykin_aap'``: physical basis with AP rescaling and the class-PT counterterm
          basis (Chudaykin et al.); uses ``bias_scheme='classpt'``.
    fsat : float, default=None
        Satellite fraction for the physical stochastic terms.  Defaults to
        ``get_physical_stochastic_settings()['fsat']``.  Pass the output of
        :func:`get_physical_stochastic_settings` directly for a specific tracer.
    sigv : float, default=None
        Velocity dispersion for the physical stochastic terms.  Defaults to
        ``get_physical_stochastic_settings()['sigv']``.
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3]. Stochastic parameters are in units of ``1/nbar``.
    mu : int, default=6
        Number of :math:`\mu` bins for multipole integration.
    damping : str, default='lor'
        Damping for Finger-of-God effect.
    """

    @classmethod
    def propose_params(cls, tracers=None, prior_basis='physical_aap', **kwargs):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str, (str, str), or None, default=None
        prior_basis : str, default='physical_aap'
            One of ``'standard'``, ``'physical'``, ``'physical_aap'``, ``'tcm_chudaykin_aap'``.

        Returns
        -------
        VariableCollection
        """
        if prior_basis not in _FOLPS_PRIOR_BASES:
            raise ValueError(f"Unknown prior_basis={prior_basis!r}; valid: {list(_FOLPS_PRIOR_BASES)}.")
        physical = (prior_basis != 'standard')
        if physical:
            auto_params = [
                Parameter('b1p', value=1., prior=dict(dist='uniform', limits=[0.1, 8.]), ref=dict(dist='norm', loc=1.5, scale=0.1), latex=r"b_1'"),
                Parameter('b2p', value=0., prior=dict(dist='norm', loc=0., scale=20.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_2'"),
                Parameter('bsp', value=0., prior=dict(dist='norm', loc=0., scale=20.),
                          ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_s'"),
                Parameter('b3p', value=0., prior=dict(dist='norm', loc=0., scale=1.),
                          ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_3'"),
                Parameter('alpha0p', value=0., prior=dict(dist='norm', loc=0., scale=50.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"\alpha_0'"),
                Parameter('alpha2p', value=0., prior=dict(dist='norm', loc=0., scale=50.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"\alpha_2'"),
                Parameter('alpha4p', value=0., prior=dict(dist='norm', loc=0., scale=50.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"\alpha_4'"),
                Parameter('ctp', value=0., fixed=True, latex=r"c_t'"),
                Parameter('X_FoGp', value=0., fixed=True, prior=dict(dist='uniform', limits=[0, 10]), latex=r"X_\mathrm{FoG}''"),
                Parameter('sn0p', value=0., prior=dict(dist='norm', loc=0., scale=2.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,0}'"),
                Parameter('sn2p', value=0., prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,2}'"),
            ]
        else:
            auto_params = [
                Parameter('b1', value=1., prior=dict(limits=[0., 10.]), ref=dict(limits=[1.4, 1.6]), latex='b_1'),
                Parameter('b2', value=0., prior=dict(limits=[-50., 50.]), ref=dict(limits=[-1., 1.]), latex='b_2'),
                Parameter('bs', value=0., prior=None, ref=dict(limits=[-1., 1.]), latex='b_s'),
                Parameter('b3', value=0., fixed=True, latex='b_3'),
                Parameter('alpha0', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r'\alpha_0'),
                Parameter('alpha2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r'\alpha_2'),
                Parameter('alpha4', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r'\alpha_4'),
                Parameter('ct', value=0., fixed=True, latex='c_t'),
                Parameter('X_FoG', value=0., fixed=True, prior=dict(dist='uniform', limits=[0, 10]), latex=r'X_\mathrm{FoG}'),
                Parameter('sn0', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,0}'),
                Parameter('sn2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=0.1), latex='s_{n,2}'),
            ]
        return propose_params_multitracer(auto_params, tracers)

    def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical_aap',
                 fsat=None, sigv=None, nbar=1e-4, mu=6, damping='lor',
                 tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers, prior_basis=prior_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        physical = prior_basis != 'standard'
        assign_params(self, vc, tracers, mapping=(lambda name: name[:-1]) if physical else None)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        if pt is None:
            pt = FOLPSPTSpectrum2Poles(**kwargs)
        self.pt = pt
        self.pt.update(k=self.k, ells=self.ells, mu=mu)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical_aap',
                      fsat=None, sigv=None, nbar=1e-4, mu=6, damping='lor',
                      tracers=None, **kwargs):
        # Non-node setup only.
        self._prior_basis = str(prior_basis)
        self._damping = str(damping)
        self._nbar = float(nbar)
        # Physical stochastic settings: pass fsat/sigv directly (e.g. the output of
        # get_physical_stochastic_settings); defaults are the generic settings.
        settings = get_physical_stochastic_settings()
        self._fsat = float(fsat) if fsat is not None else settings['fsat']
        self._sigv = float(sigv) if sigv is not None else settings['sigv']
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)

    def __call__(self):
        sigma8 = self.pt.sigma8
        fsigma8 = self.pt.fsigma8
        f = fsigma8 / sigma8
        qpar = self.pt.qpar
        qper = self.pt.qper
        A_AP = 1. / (qper**2 * qpar)
        # Amplitude rescaling (Class-PT style)
        A = sigma8 / self.pt.sigma8_fid

        bias_scheme = 'folps'
        if self._prior_basis == 'standard':
            b1, b2, bs, b3 = self.b1.value, self.b2.value, self.bs.value, self.b3.value
            alpha0, alpha2, alpha4, ct = self.alpha0.value, self.alpha2.value, self.alpha4.value, self.ct.value
            sn0, sn2, X_FoG = self.sn0.value, self.sn2.value, self.X_FoG.value
            pars = [b1, b2, bs, b3, alpha0, alpha2, alpha4, ct, sn0, sn2, 1. / self._nbar, X_FoG]

        elif self._prior_basis in ['physical', 'physical_aap']:  # physical basis with AP rescaling
            if 'aap' not in self._prior_basis: A_AP = 1.
            b1L = self.b1.value / (A * A_AP**0.5) - 1.
            b2L = self.b2.value / (A**2 * A_AP**0.5)
            b1E = 1. + b1L
            b2E = b2L
            bK2 = self.bs.value / (A**2 * A_AP**0.5) - 2. / 7. * b1L
            btd = self.b3.value / (A**4 * A_AP) + 23. / 42. * b1L
            bsE = 2. * bK2
            b3E = 64. / 105. * (-5. / 4. * bsE - btd)
            a0t, a2t, a4t = self.alpha0.value / (A**2 * A_AP), self.alpha2.value / (A**2 * A_AP), self.alpha4.value / (A**2 * A_AP)
            alpha0 = b1E**2 * a0t
            alpha2 = b1E * f * (a0t + a2t)
            alpha4 = f**2 * a2t + b1E * f * a4t
            sn0 = self.sn0.value / A_AP / self._nbar
            sn2 = self.sn2.value / A_AP / self._nbar * self._fsat * self._sigv**2
            pars = [b1E, b2E, bsE, b3E, alpha0, alpha2, alpha4, self.ct.value,
                               sn0, sn2, 1., self.X_FoG.value]

        else:  # 'tcm_chudaykin_aap': physical + AP with the class-PT counterterm basis
            bias_scheme = 'classpt'
            b1L = self.b1.value / A - 1.
            b2L = self.b2.value / A**2
            bsL = self.bs.value / A**2
            b3 = self.b3.value / A
            c0, c2, c4 = self.alpha0.value / (A**2 * A_AP), self.alpha2.value / (A**2 * A_AP), self.alpha4.value / (A**2 * A_AP)
            ct0 = -2. / 105. * (105. * c0 - 35. * c2 * f + 9. * c4 * f**2)
            ct2 = -2. / 7. * f * (7. * c2 - 6. * f * c4)
            ct4 = -2. * f**2 * c4
            sn0 = self.sn0.value / self._nbar
            sn2 = self.sn2.value / self._nbar * self._fsat * self._sigv**2
            pars = [1. + b1L, b2L, bsL, b3, ct0, ct2, ct4, 0.,
                               sn0, sn2, 1., self.X_FoG.value]

        self.poles = self.pt.combine_bias_terms_spectrum2_poles(pars, bias_scheme, self._damping)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


class FOLPSTracerCorrelation2Poles(Calculator):
    r"""
    FOLPS tracer correlation function multipoles via FFTLog.

    Parameters
    ----------
    s : array, default=None
    pt : FOLPSTracerSpectrum2Poles, default=None
    ells : tuple of int, default=(0, 2, 4)
    template : template calculator, default=None
    prior_basis : str, default='physical_aap'
        See :class:`FOLPSTracerSpectrum2Poles`.
    fsat, sigv, nbar : forwarded to :class:`FOLPSTracerSpectrum2Poles`.
    """

    @classmethod
    def propose_params(cls, tracers=None, **kwargs):
        """Delegate to :meth:`FOLPSTracerSpectrum2Poles.propose_params`."""
        return FOLPSTracerSpectrum2Poles.propose_params(tracers=tracers, **kwargs)

    def __init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical_aap', tracers=None, **kwargs):
        # Nodes (Calculator deps) and their update() live in __init__.
        if s is None:
            s = np.linspace(20., 200., 181)
        self.s = np.asarray(s, dtype='f8')
        self.ells = tuple(ells)
        kin = np.geomspace(1e-4, 0.6, 300)
        if pt is None:
            pt = FOLPSTracerSpectrum2Poles(prior_basis=prior_basis, tracers=tracers, **kwargs)
        self.pt = pt
        self.pt.update(k=kin, ells=self.ells)
        if template is not None:
            self.pt.update(template=template)

    def __post_init__(self, s=None, pt=None, ells=(0, 2, 4), template=None, prior_basis='physical_aap', tracers=None, **kwargs):
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

#@jax.jit(static_argnames=['multipoles', 'precision', 'damping', 'interpolation_method', 'bias_scheme', 'model', 'renormalized'])
def _get_spectrum3poles_folps(pars, k1k2, k_pkl_pklnw_fk,
                              f0, qpar, qper, multipoles=['B000', 'B202'],
                              precision=(8, 10, 10), damping='lor',
                              interpolation_method='linear',
                              bias_scheme='folps', model='FOLPSD', renormalized=True):
    import folps as folpsv2
    # folpsv2.MatrixCalculator(A_full=True, use_TNS_model=False)
    # folps_bispectrum_class = folpsv2.BispectrumCalculator_fk(model='FOLPSD')
    f0 = jnp.asarray(f0)
    bpars = jnp.asarray(pars)

    ells = []
    _ells = ['B000', 'B110', 'B220', 'B112', 'B202']
    provided = []
    for ell in multipoles:
        if ell in _ells:
            ells.append(ell)
            provided.append((ell, False))
        elif (ell_swap:=ell[0] + ell[2:0:-1] + ell[3:]) in _ells:
            ells.append(ell_swap)
            provided.append((ell_swap, True))
        else:
            provided.append((False, False))

    bispectrum = folpsv2.BispectrumCalculator(model=model)
    result = bispectrum.Sugiyama_Bell(
        f=f0,
        bpars=bpars,
        k_pkl_pklnw=k_pkl_pklnw_fk,
        k1k2pairs=k1k2,
        qpar=qpar,
        qper=qper,
        precision=precision,
        damping=damping,
        multipoles=ells,
        bias_scheme=bias_scheme,
        renormalize=renormalized,
        interpolation_method=interpolation_method
    )

    toret = []
    for ell, (_ell, swap) in zip(multipoles, provided):
        if _ell:
            tmp = result[ells.index(_ell)]
            if swap: tmp = tmp.T
            toret.append(tmp)
        else:
            toret.append(jnp.zeros(len(k1k2)))
    folpsv2.BispectrumCalculator._tables_cache = {}  # to avoid leak
    return jnp.array(toret)


class FOLPSTracerSpectrum3Poles(Calculator):
    r"""
    FOLPS tracer bispectrum multipoles.

    Computes the redshift-space bispectrum multipoles ``B_{l1 l2 L}(k1, k2)`` from the
    linear power spectrum via ``folps.BispectrumCalculator.Sugiyama_Bell``.

    Parameters
    ----------
    k : array, shape (N, 2), default=None
        Output ``(k1, k2)`` wavenumber pairs [h/Mpc].  Defaults to a diagonal grid
        ``k1 == k2`` over ``np.linspace(0.01, 0.1, 11)`` (the case handled by Sugiyama_Bell).
    pt : FOLPSPTSpectrum2Poles, default=None
        PT calculator providing ``sigma8``, ``fsigma8``, ``qpar``, ``qper`` and the
        underlying template.  Defaults to a new :class:`FOLPSPTSpectrum2Poles`.
    template : template calculator, default=None
        Forwarded to ``pt`` if given.  Defaults to :class:`DirectSpectrum2Template`.
    ells : tuple of (int, int, int), default=((0, 0, 0), (2, 0, 2))
        Bispectrum multipole triplets ``(l1, l2, L)``.  Available: (0,0,0), (1,1,0),
        (2,2,0), (0,2,2), (1,1,2).
    prior_basis : str, default='physical_aap'
        Bias / counterterm / stochastic parameterization:

        - ``'standard'``: Eulerian FOLPS bias; parameters ``b1, b2, bs, c1, c2, sn0, snb0, X_FoG``.
        - ``'physical'``: physical (velocileptors-DR1) Lagrangian basis, no AP rescaling;
          parameters carry a ``p`` suffix.
        - ``'physical_aap'`` (default): physical basis with AP rescaling (2pt3pt prior document).
        - ``'tcm_chudaykin_aap'``: physical basis with AP rescaling and class-PT counterterm basis.
    fsat : float, default=None
        Satellite fraction for the physical stochastic terms.
    sigv : float, default=None
        Velocity dispersion for the physical stochastic terms.
    nbar : float, default=1e-4
        Number density [(Mpc/h)^-3]. Stochastic parameters are in units of ``1/nbar``.
    model : str, default='FOLPSD'
    damping : str, default='lor'
    precision : tuple, default=(8, 10, 10)
    renormalized : bool, default=True
    interpolation_method : str, default='linear'

    Reference
    ---------
    arXiv:2404.07269
    """
    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/FolpsD')

    @classmethod
    def propose_params(cls, tracers=None, prior_basis='physical_aap'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this theory.

        Parameters
        ----------
        tracers : str or None, default=None
        prior_basis : str, default='physical_aap'

        Returns
        -------
        VariableCollection
        """
        if prior_basis not in _FOLPS_PRIOR_BASES:
            raise ValueError(f"Unknown prior_basis={prior_basis!r}; valid: {list(_FOLPS_PRIOR_BASES)}.")
        physical = (prior_basis != 'standard')
        if physical:
            auto_params = [
                Parameter('b1p', value=1., prior=dict(dist='uniform', limits=[0.1, 8.]), ref=dict(dist='norm', loc=1.5, scale=0.1), latex=r"b_1'"),
                Parameter('b2p', value=0., prior=dict(dist='norm', loc=0., scale=20.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_2'"),
                Parameter('bsp', value=0., prior=dict(dist='norm', loc=0., scale=20.),
                          ref=dict(dist='norm', loc=0., scale=1.), latex=r"b_s'"),
                Parameter('c1p', value=0., prior=dict(dist='norm', loc=0., scale=1.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"c_1'"),
                Parameter('c2p', value=0., fixed=True, prior=dict(dist='norm', loc=0., scale=20.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"c_2'"),
                Parameter('sn0p', value=0., prior=dict(dist='norm', loc=0., scale=2.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,0}'"),
                Parameter('sn0bp', value=0., prior=dict(dist='norm', loc=0., scale=1.), ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{nb,0}'"),
                Parameter('X_FoGp', value=0., fixed=True, latex=r"X'_\mathrm{FoG}"),
            ]
        else:
            auto_params = [
                Parameter('b1', value=1., prior=dict(limits=[0., 10.]), ref=dict(limits=[1.4, 1.6]), latex='b_1'),
                Parameter('b2', value=0., prior=dict(limits=[-50., 50.]), ref=dict(limits=[-1., 1.]), latex='b_2'),
                Parameter('bs', value=0., prior=None, ref=dict(limits=[-1., 1.]), latex='b_s'),
                Parameter('c1', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='c_1'),
                Parameter('c2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='c_2'),
                Parameter('sn0', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{n,0}"),
                Parameter('sn0b', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex=r"s_{nb,0}"),
                Parameter('X_FoG', value=0., fixed=True, latex=r'X_\mathrm{FoG}'),
            ]
        return propose_params_multitracer(auto_params, tracers)  # no cross (bispectra not implemented)

    def __init__(self, k=None, pt=None, ells=((0, 0, 0), (2, 0, 2)), template=None,
                 prior_basis='physical_aap', tracers=None, params=None, **kwargs):
        # Nodes (Parameters + Calculator deps) and their update() live in __init__.
        vc = type(self).propose_params(tracers=tracers, prior_basis=prior_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        physical = prior_basis != 'standard'
        assign_params(self, vc, tracers, mapping=(lambda name: name[:-1]) if physical else None)
        if k is None:
            k = np.column_stack([np.linspace(0.01, 0.1, 11)] * 2)
        self.k = np.atleast_2d(np.asarray(k, dtype='f8'))
        self.ells = tuple(tuple(int(e) for e in ell) for ell in ells)
        if pt is None:
            pt = FOLPSPTSpectrum2Poles()
        self.pt = pt
        if template is not None:
            self.pt.update(template=template)
        self.pt.template.update(with_now='peakaverage')

    def __post_init__(self, k=None, pt=None, ells=((0, 0, 0), (2, 0, 2)), template=None,
                      prior_basis='physical_aap', fsat=None, sigv=None,
                      nbar=1e-4, model='FOLPSD', damping='lor', precision=(8, 10, 10),
                      renormalized=True, interpolation_method='linear', tracers=None, **kwargs):
        # Non-node setup only.
        self._prior_basis = str(prior_basis)
        self._nbar = float(nbar)
        settings = get_physical_stochastic_settings()
        self._fsat = float(fsat) if fsat is not None else settings['fsat']
        self._sigv = float(sigv) if sigv is not None else settings['sigv']
        self._options = dict(model=str(model), damping=str(damping),
                             precision=tuple(precision), renormalized=bool(renormalized),
                             interpolation_method=str(interpolation_method))

    def __call__(self):
        sigma8 = self.pt.sigma8
        qpar = self.pt.qpar
        qper = self.pt.qper
        A_AP = 1. / (qper**2 * qpar)
        A = sigma8 / self.pt.sigma8_fid

        bias_scheme = 'folps'
        kNL = 0.3

        if self._prior_basis == 'standard':
            pars = [self.b1.value, self.b2.value, self.bs.value, self.c1.value, self.c2.value,
                    self.sn0.value, self.sn0b.value, self.X_FoG.value]

        elif self._prior_basis in ['physical', 'physical_aap']:  # physical basis with AP rescaling
            if 'aap' not in self._prior_basis: A_AP = 1.
            b1L = self.b1.value / (A * A_AP**0.5) - 1.
            b1E = 1. + b1L
            b2E = self.b2.value / (A**2 * A_AP**0.5)
            bK2 = self.bs.value / (A**2 * A_AP**0.5) - 2. / 7. * b1L
            bsE = 2. * bK2
            c1 = self.c1.value / kNL**2 / (A**2 * A_AP)
            c2 = self.c2.value / kNL**2 / (A**2 * A_AP)
            pars = [b1E, b2E, bsE, c1, c2,
                    self.sn0.value / A_AP / self._nbar, self.sn0b.value / A_AP / self._nbar, self.X_FoG.value]

        else:  # 'tcm_chudaykin_aap'
            bias_scheme = 'classpt'
            b1L = self.b1.value / A - 1.
            b2L = self.b2.value / A**2
            bsL = self.bs.value / A**2
            c1 = self.c1.value / kNL**2 / (A**2 * A_AP)
            c2 = self.c2.value / kNL**2 / (A**2 * A_AP)
            pars = [1. + b1L, b2L, bsL, c1, c2,
                    self.sn0.value / self._nbar, self.sn0b.value / self._nbar, self.X_FoG.value]

        multipoles = tuple('B{:d}{:d}{:d}'.format(*ell) for ell in self.ells)
        self.poles = self.pt.combine_bias_terms_spectrum3_poles(pars, self.k, multipoles, bias_scheme=bias_scheme, **self._options)
        return self.poles

    def tree_flatten(self):
        return [self.poles], {'k': self.k, 'ells': self.ells}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        obj.k = aux['k']
        obj.ells = aux['ells']
        return obj


class JAXEffortTracerSpectrum2Poles(Calculator):
    r"""
    Tracer power-spectrum multipoles from a JAXEffort emulator.

    Cosmological parameters are supplied via a :class:`~desilike.theories.primordial_cosmology.CosmoprimoCosmology`
    dependency (``cosmo``), accessed as ``cosmo['h']``, ``cosmo['omega_cdm']``, etc.
    Growth ``D(z)``, ``f(z)`` and Alcock-Paczynski distortion are computed from a JAXEffort
    ``w0waCDMCosmology`` object built from those parameters.

    Parameters
    ----------
    cosmo : CosmoprimoCosmology or None, default=None
        Cosmology calculator.  When ``None`` a default :class:`~desilike.theories.primordial_cosmology.CosmoprimoCosmology`
        with ``engine='eisenstein_hu'`` is created internally.
    k : array, default=None
        Output wavenumbers [h/Mpc].  Defaults to ``np.linspace(0.01, 0.2, 101)``.
    ells : tuple of int, default=(0, 2, 4)
        Multipole orders.
    z : float, default=0.5
        Effective redshift.
    mu : int, default=8
        Number of Gauss-Legendre mu-bins in [0, 1].
    model : str, default='velocileptors_rept_mnuw0wacdm'
        JAXEffort trained-emulator key.
    prior_basis : str, default='standard'
        ``'physical'``: sigma8-normalized Lagrangian bias (same as :class:`REPTVelocileptorsTracerSpectrum2Poles`);
        params carry a ``p`` suffix and are stored as ``b1``, ``b2``, etc. via :func:`assign_params` mapping.
        ``'standard'``: standard velocileptors REPT basis.
    fsat, sigv, nbar : floats
        Physical-basis stochastic settings; forwarded to :func:`_velocileptors_physical_to_standard`.

    Reference
    ---------
    https://github.com/CosmologicalEmulators/jaxeffort
    """
    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/CosmologicalEmulators/jaxeffort')

    @classmethod
    def propose_params(cls, tracers=None, prior_basis='standard'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for the bias parameters.

        Cosmological parameters come from the :class:`~desilike.theories.primordial_cosmology.CosmoprimoCosmology`
        dependency and are not included here.

        Parameters
        ----------
        tracers : str or None, default=None
        prior_basis : str, default='standard'

        Returns
        -------
        VariableCollection
        """
        if prior_basis == 'physical':
            return propose_params_multitracer(_velocileptors_default_params('physical'), tracers)
        return propose_params_multitracer([
            Parameter('b1', value=2., prior=dict(limits=[0., 4.]),
                      ref=dict(dist='norm', loc=2., scale=0.1), latex='b_1'),
            Parameter('b2', value=0., prior=dict(dist='norm', loc=0., scale=2.),
                      ref=dict(dist='norm', loc=0., scale=1.), latex='b_2'),
            Parameter('bs', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='b_s'),
            Parameter('b3', value=0., fixed=True, latex='b_3'),
            Parameter('alpha0', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=10.), latex=r'\alpha_0'),
            Parameter('alpha2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=10.), latex=r'\alpha_2'),
            Parameter('alpha4', value=0., fixed=True, latex=r'\alpha_4'),
            Parameter('alpha6', value=0., fixed=True, latex=r'\alpha_6'),
            Parameter('sn0', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='s_{n,0}'),
            Parameter('sn2', value=0., prior=None, ref=dict(dist='norm', loc=0., scale=1.), latex='s_{n,2}'),
            Parameter('sn4', value=0., fixed=True, latex='s_{n,4}'),
        ], tracers)

    def __init__(self, *args, cosmo=None, prior_basis='standard', model='velocileptors_rept_mnuw0wacdm', tracers=None, params=None, **kwargs):
        # ── cosmology dep (provides logA, n_s, h, omega_b, omega_cdm, m_ncdm, w0_fld, wa_fld) ──
        if cosmo is None:
            cosmo = CosmoprimoCosmology(engine='eisenstein_hu')
        self.cosmo = cosmo  # Calculator dep; build_graph discovers it from __dict__
        # ── velocileptors bias ──
        vc = type(self).propose_params(tracers=tracers, prior_basis=prior_basis)
        if params is not None:
            vc = vc + VariableCollection(params)
        physical = prior_basis == 'physical'
        assign_params(self, vc, tracers, mapping=(lambda name: name[:-1]) if physical else None)

    def __post_init__(self, k=None, ells=(0, 2, 4), z=0.5, mu=8, prior_basis='standard',
                      fsat=None, sigv=None, nbar=1e-4,
                      model='velocileptors_rept_mnuw0wacdm', fiducial='DESI', **kwargs):
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self.z = float(z)
        self._prior_basis = str(prior_basis)
        self._model = str(model)
        settings = get_physical_stochastic_settings()
        self._fsat = float(fsat) if fsat is not None else settings['fsat']
        self._sigv = float(sigv) if sigv is not None else settings['sigv']
        self._nbar = float(nbar)
        self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        from scipy import special
        # Legendre coefficients (highest power first, for jnp.polyval) per multipole.
        self._legendre_coeffs = [np.asarray(special.legendre(ell).c) for ell in self.ells]
        import jaxeffort
        self._emulators = [jaxeffort.trained_emulators[self._model][str(ell)] for ell in self.ells]
        self._kgrid = np.asarray(self._emulators[0].P11.k_grid)
        # Fiducial distances for AP (fixed); same distance formulas as in __call__
        fiducial = _get_fiducial(fiducial)
        # Let's just use the same cosmology class, for numerical error cancellation around the fiducial
        fiducial = jaxeffort.W0WaCDMCosmology(ln10As=fiducial['logA'], ns=fiducial['n_s'],
                                         h=fiducial['h'], omega_b=fiducial['omega_b'],
                                         omega_c=fiducial['omega_cdm'], m_nu=fiducial['m_ncdm_tot'],
                                         w0=fiducial['w0_fld'], wa=fiducial['wa_fld'])
        self._E_fid = fiducial.E_z(self.z)
        self._dM_fid = fiducial.r_z(self.z)
        self._D_fid, self._f_fid = fiducial.D_f_z(z)

    def __call__(self):
        import jaxeffort
        # Read cosmological parameters from the CosmoprimoCosmology dep.
        z = self.z
        jax_cosmo = jaxeffort.W0WaCDMCosmology(ln10As=self.cosmo['logA'], ns=self.cosmo['n_s'],
                                               h=self.cosmo['h'], omega_b=self.cosmo['omega_b'],
                                               omega_c=self.cosmo['omega_cdm'], m_nu=self.cosmo['m_ncdm_tot'],
                                               w0=self.cosmo['w0_fld'], wa=self.cosmo['wa_fld'])
        theta = jnp.array([z, jax_cosmo.ln10As, jax_cosmo.ns, 100. * jax_cosmo.h, jax_cosmo.omega_b, jax_cosmo.omega_c, jax_cosmo.m_nu, jax_cosmo.w0, jax_cosmo.wa])
        D, f = jax_cosmo.D_f_z(z)
        # Alcock-Paczynski from the JAXEffort cosmology distances (qpar = D_H/D_H_fid, qper = D_M/D_M_fid).
        qpar = (self._E_fid) / (jax_cosmo.E_z(z))
        qper = jax_cosmo.r_z(z) / self._dM_fid
        A_AP = 1. / (qper**2 * qpar)

        if self._prior_basis in ['physical', 'physical_aap']:
            biases = _velocileptors_physical_to_standard(
                self.b1.value, self.b2.value, self.bs.value, self.b3.value,
                self.alpha0.value, self.alpha2.value, self.alpha4.value, self.alpha6.value,
                self.sn0.value, self.sn2.value, self.sn4.value,
                f, self._fsat, self._sigv, self._nbar, A=D / self._D_fid,
                A_AP=A_AP if 'aap' in self._prior_basis else 1., rept='rept' in self._model)
        else:
            b1, b2 = self.b1.value, self.b2.value
            bs, b3 = self.bs.value, self.b3.value
            a0, a2, a4, a6 = self.alpha0.value, self.alpha2.value, self.alpha4.value, self.alpha6.value
            sn0, sn2, sn4 = self.sn0.value, self.sn2.value, self.sn4.value
            if 'rept' in self._model:  # velocileptors REPT co-evolution of bs / b3
                bs, b3 = bs - (2. / 7.) * (b1 - 1.), 3. * b3 + (b1 - 1.)
            biases = jnp.array([b1, b2, bs, b3, a0, a2, a4, a6, sn0, sn2, sn4])

        poles = [emu.get_Pl(theta, biases, D) for emu in self._emulators]  # each on self._kgrid

        jac, kap, muap = _ap_k_mu(self.k[:, None], self._to_poles.mu, qpar, qper)  # (n_k, n_mu)
        pkmu = jnp.zeros_like(kap)
        for leg_coeffs, pole in zip(self._legendre_coeffs, poles):
            pole_at_kap = interpax.interp1d(kap.ravel(), self._kgrid, pole, method='cubic', extrap=True).reshape(kap.shape)
            pkmu = pkmu + pole_at_kap * jnp.polyval(leg_coeffs, muap)
        pkmu = jac * pkmu
        self.poles = self._to_poles(pkmu)
        return self.poles

    def tree_flatten(self):
        return [self.poles], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj
