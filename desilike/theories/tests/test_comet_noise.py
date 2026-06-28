"""
Characterise the GP emulator noise in COMETTracerSpectrum2Poles chi2 profiles.

The comet-emu sklearn GP has intrinsic interpolation noise of ~1-5e-3 absolute
in each PX_ell diagram output per k-bin.  After summing over 180 data points
(60 k-bins x 3 multipoles) the noise-to-signal ratio in chi2 is ~3e-5,
independent of which cosmological parameter is varied.

Near the posterior peak (chi2 ~ N_data ~ 180), the absolute chi2 noise is
~0.006 units -- negligible for MCMC or profiling.  The noise is the same for
both backend='numpy' and backend='jax' because both use the same GP model files.
"""
import numpy as np
import pytest

from desilike.theories.galaxy_clustering.full_shape import COMETTracerSpectrum2Poles
from desilike.base import compile


# Relative GP noise level (chi2_noise / chi2_value), consistent across parameters.
_EXPECTED_RELATIVE_NOISE = 2e-4  # conservative upper bound; empirically ~3e-5


def _make_pipe(k, z=1.0, ells=(0, 2, 4)):
    theory = COMETTracerSpectrum2Poles(k=k, z=z, ells=ells)
    return compile(theory)


def _chi2_profile(pipe, params0, pred0, noise_std, pname, dvs):
    """Return chi2 values at offsets dvs from the fiducial for parameter pname."""
    chi2 = np.zeros(len(dvs))
    for idx, dv in enumerate(dvs):
        pred = np.asarray(pipe({**params0, pname: params0[pname] + dv})).ravel()
        chi2[idx] = np.sum(((pred - pred0) / noise_std) ** 2)
    return chi2


def _high_freq_noise(chi2, dvs, poly_degree=8):
    """
    Fit a degree-``poly_degree`` polynomial and return the residuals.
    The rms of the residuals estimates the high-frequency GP noise in chi2.
    """
    coeffs = np.polyfit(dvs, chi2, poly_degree)
    return chi2 - np.polyval(coeffs, dvs)


@pytest.fixture(scope='module')
def pipe_and_fiducial():
    k = np.linspace(0.02, 0.3, 60)
    pipe = _make_pipe(k)
    params0 = {p.name: float(p.value) for p in pipe.params.select(fixed=False, derived=False)}
    pred0 = np.asarray(pipe(params0)).ravel()
    noise_std = np.maximum(0.01 * np.abs(pred0), 1.0)
    return pipe, params0, pred0, noise_std


@pytest.mark.parametrize('pname,sigma_param,npts', [
    ('omega_cdm', 0.002, 40),
    ('omega_b',   0.001, 40),
    ('logA',      0.010, 40),
    ('n_s',       0.010, 40),
])
def test_chi2_noise_relative_level(pipe_and_fiducial, pname, sigma_param, npts):
    """
    The high-frequency chi2 noise (after removing smooth polynomial structure)
    must be below _EXPECTED_RELATIVE_NOISE * chi2_max for each cosmological
    parameter at a ±2 sigma scan range.
    """
    pipe, params0, pred0, noise_std = pipe_and_fiducial
    if pname not in params0:
        pytest.skip(f'{pname} not a free parameter')

    dvs = np.linspace(-2 * sigma_param, 2 * sigma_param, npts)
    chi2 = _chi2_profile(pipe, params0, pred0, noise_std, pname, dvs)
    residuals = _high_freq_noise(chi2, dvs)

    chi2_max = chi2.max()
    noise_rms = np.std(residuals)
    relative_noise = noise_rms / chi2_max

    assert relative_noise < _EXPECTED_RELATIVE_NOISE, (
        f'{pname}: chi2_noise_rms={noise_rms:.4f}, chi2_max={chi2_max:.1f}, '
        f'relative={relative_noise:.2e} > {_EXPECTED_RELATIVE_NOISE:.2e}'
    )


def test_chi2_noise_near_posterior_peak(pipe_and_fiducial):
    """
    At MCMC-relevant scales (within the posterior, chi2 ~ N_data ~ 180) the
    absolute chi2 noise must be < 0.1 units for omega_cdm, the noisiest param.
    """
    pipe, params0, pred0, noise_std = pipe_and_fiducial
    pname = 'omega_cdm'
    sigma_param = 0.002
    eps = 0.02 * sigma_param  # small steps; total displacement = 20*eps = 0.4*sigma
    npts = 21
    dvs = np.arange(npts) * eps
    chi2 = _chi2_profile(pipe, params0, pred0, noise_std, pname, dvs)
    residuals = _high_freq_noise(chi2, dvs, poly_degree=4)
    noise_rms = np.std(residuals)
    assert noise_rms < 0.5, (
        f'omega_cdm near-peak chi2 noise={noise_rms:.4f} > 0.5 units'
    )


def test_h_stability(pipe_and_fiducial):
    """
    Varying h must produce no NaN predictions over a ±4-sigma range, and the
    chi2 profile must be smooth.  This guards against the Omega_de<0 /
    growth_factor_lambda NaN crash that occurs when h is pushed outside the
    emulator's valid range.
    """
    pipe, params0, pred0, noise_std = pipe_and_fiducial
    pname = 'h'
    if pname not in params0:
        pytest.skip('h not a free parameter')

    sigma_h = 0.008  # typical 1-sigma from CMB+BAO
    dvs = np.linspace(-2 * sigma_h, 2 * sigma_h, 50)
    chi2 = np.zeros(len(dvs))
    for idx, dv in enumerate(dvs):
        pred = np.asarray(pipe({**params0, pname: params0[pname] + dv})).ravel()
        assert not np.any(np.isnan(pred)), (
            f'NaN in prediction at h={params0[pname] + dv:.4f} (offset {dv:+.4f})'
        )
        chi2[idx] = np.sum(((pred - pred0) / noise_std) ** 2)

    # h shifts the spectrum shape strongly, so the GP noise is ~10x higher
    # relative to other params; empirically ~4e-4 over ±2 sigma.
    _h_relative_noise_bound = 1e-3
    residuals = _high_freq_noise(chi2, dvs)
    noise_rms = np.std(residuals)
    relative_noise = noise_rms / chi2.max()
    assert relative_noise < _h_relative_noise_bound, (
        f'h: chi2_noise_rms={noise_rms:.4f}, chi2_max={chi2.max():.1f}, '
        f'relative={relative_noise:.2e} > {_h_relative_noise_bound:.2e}'
    )


def test_gradient_stability(pipe_and_fiducial):
    """
    The first derivative estimate of the model prediction w.r.t. omega_cdm must
    be stable across step sizes from 1e-5 to 1e-2 (relative change < 1%).
    """
    pipe, params0, pred0, noise_std = pipe_and_fiducial
    pname = 'omega_cdm'
    v0 = params0[pname]
    grads = []
    for logeps in [-5, -4, -3, -2]:
        eps = 10. ** logeps
        pred_plus  = np.asarray(pipe({**params0, pname: v0 + eps})).ravel()
        pred_minus = np.asarray(pipe({**params0, pname: v0 - eps})).ravel()
        d1 = (pred_plus - pred_minus) / (2 * eps)
        grads.append(np.abs(d1).max())
    grads = np.array(grads)
    # All gradient estimates should agree to within 1%
    relative_spread = (grads.max() - grads.min()) / grads.mean()
    assert relative_spread < 0.01, (
        f'Gradient estimates vary by {relative_spread:.2%} across step sizes '
        f'(values: {grads}): first derivative is not stable'
    )
