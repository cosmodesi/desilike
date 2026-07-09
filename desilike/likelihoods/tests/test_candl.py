"""Tests for the generic candl CMB likelihood wrapper and the clipy-based
Planck PR3 wrappers.

candl tests use a minimal synthetic data set (one 'TT 90x90' spectrum, 3 bins)
written to a temporary directory.  clipy integration tests download the real
Sroll2 ``.clik`` file via the :meth:`install` method (network required).

Harmonic :math:`C_\\ell` require a real Boltzmann engine (the JAX-native
``eisenstein_hu`` engine cannot compute them), so ``cosmo`` is built with
``engine='camb'`` throughout.
"""

import numpy as np
import jax
import pytest

from desilike.base import compile, get_params, Posterior
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.likelihoods.cmb import CandlLikelihood, CandlLensLikelihood

pytest.importorskip('candl')


def _write_candl_fixture(data_dir):
    """Write a minimal candl data set: one 'TT 90x90' spectrum (3 bins, theory ell
    2-10), a single calibration nuisance parameter ('Tcal') with a Gaussian prior."""
    with open(data_dir / 'test_candl.yaml', 'w') as file:
        file.write(
            'name: "Test candl dataset"\n'
            'spectra_info:\n'
            '  - "TT 90x90": 3\n'
            'band_power_file: band_powers.txt\n'
            'covariance_file: covariance.txt\n'
            'window_functions_folder: windows/\n'
            'data_model:\n'
            '  - Module: common.CalibrationSingleScalar\n'
            '    cal_param: Tcal\n'
            'priors:\n'
            '  - par_names: [Tcal]\n'
            '    central_value: 1.0\n'
            '    prior_std: 0.01\n'
        )
    np.savetxt(data_dir / 'band_powers.txt', np.array([1000., 600., 300.]))
    np.savetxt(data_dir / 'covariance.txt', np.diag([50., 30., 15.]) ** 2)

    (data_dir / 'windows').mkdir(exist_ok=True)
    ells = np.arange(2, 11)
    window = np.zeros((9, 4))
    window[:, 0] = ells
    window[0:3, 1] = 1. / 3
    window[3:6, 2] = 1. / 3
    window[6:9, 3] = 1. / 3
    np.savetxt(data_dir / 'windows' / 'TT_90x90_window_functions.txt', window)


def _write_candl_fixture_with_cosmo_prior(data_dir):
    """Like _write_candl_fixture, but with an extra prior on 'tau', a cosmological
    parameter (not a data-model nuisance parameter), exercising cosmo_params."""
    _write_candl_fixture(data_dir)
    with open(data_dir / 'test_candl.yaml', 'a') as file:
        file.write(
            '  - par_names: [tau]\n'
            '    central_value: 0.0566\n'
            '    prior_std: 0.0058\n'
        )


def _write_candl_lens_fixture(data_dir):
    """Write a minimal candl lensing data set: a single 'kk' spectrum (3 bins,
    theory ell 2-10), no nuisance parameters, no priors, no data_model."""
    with open(data_dir / 'test_candl_lens.yaml', 'w') as file:
        file.write(
            'name: "Test candl lensing dataset"\n'
            'spectra_info:\n'
            '  - kk: 3\n'
            'band_power_file: band_powers.txt\n'
            'covariance_file: covariance.txt\n'
            'window_functions_folder: windows/\n'
        )
    np.savetxt(data_dir / 'band_powers.txt', np.array([1e-7, 5e-8, 2e-8]))
    np.savetxt(data_dir / 'covariance.txt', np.diag([1e-8, 5e-9, 2e-9]) ** 2)

    (data_dir / 'windows').mkdir(exist_ok=True)
    ells = np.arange(2, 11)
    window = np.zeros((9, 4))
    window[:, 0] = ells
    window[0:3, 1] = 1. / 3
    window[3:6, 2] = 1. / 3
    window[6:9, 3] = 1. / 3
    np.savetxt(data_dir / 'windows' / 'kk_window_functions.txt', window)


def test_likelihood(tmp_path):
    """CandlLikelihood constructs, compiles, and evaluates to a finite logpdf,
    matching between direct and jit-compiled evaluation."""
    _write_candl_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo)

    assert like.ndata == 3
    assert set(like.params) == {'Tcal'}

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}

    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)

    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))


def test_required_nuisance_parameter_uses_candl_prior(tmp_path):
    """The 'Tcal' Parameter picks up value/ref from candl's declared GaussianPrior,
    but has no desilike-level prior (candl.Like.log_like applies it internally)."""
    _write_candl_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo)

    param = like.params['Tcal']
    assert not param.fixed
    assert param.value == 1.
    assert np.isclose(param.ref.std(), 0.01)
    assert not param.prior.is_proper()  # no desilike-level prior: avoid double-counting


def test_split_diag_priors(tmp_path):
    """With split_diag_priors=True, candl.Like.priors is left untouched (so any
    off-diagonal correlation is still handled by candl), and the same central
    value/std are *also* attached as a proper desilike Parameter prior."""
    _write_candl_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo, split_diag_priors=True)

    assert len(like.like.priors) == 1  # untouched, unlike the old clear_internal_priors design
    param = like.params['Tcal']
    assert not param.fixed
    assert param.value == 1.
    assert param.prior.is_proper()
    assert np.isclose(param.prior.std(), 0.01)

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}
    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)


def test_split_diag_priors_numerically_equivalent(tmp_path):
    """The total posterior (likelihood + prior) with split_diag_priors=True must
    exactly match the split_diag_priors=False total: the diagonal piece moved to
    desilike's Parameter.prior is precisely subtracted from the candl likelihood
    call, so summing it back in via Prior reconstructs the original total."""
    _write_candl_fixture(tmp_path)

    cosmo_nonsplit = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like_nonsplit = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo_nonsplit, split_diag_priors=False)
    posterior_nonsplit = Posterior(like_nonsplit)

    cosmo_split = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like_split = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo_split, split_diag_priors=True)
    posterior_split = Posterior(like_split)

    # Evaluate away from the prior mode so the (otherwise zero-at-mode) prior term is non-trivial.
    params = get_params(posterior_nonsplit)
    defaults = {p.name: p._value for p in params}
    defaults['Tcal'] = 1.02

    logpdf_nonsplit = compile(posterior_nonsplit)(defaults)
    logpdf_split = compile(posterior_split)(defaults)
    assert np.isclose(float(logpdf_nonsplit), float(logpdf_split))


def test_params_override(tmp_path):
    """Passing params= overrides the auto-built nuisance parameter."""
    _write_candl_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    from desilike.parameter import Parameter
    like = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo, params=[Parameter('Tcal', value=1.05, fixed=True)])

    assert like.params['Tcal'].fixed
    assert like.params['Tcal'].value == 1.05


def test_cosmo_params_external_prior(tmp_path):
    """A candl prior on a parameter not in required_nuisance_parameters ('tau') is
    resolved via cosmo_params (candl name -> cosmoprimo name) and read directly from
    cosmo, rather than creating a new desilike Parameter."""
    _write_candl_fixture_with_cosmo_prior(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = CandlLikelihood(str(tmp_path / 'test_candl.yaml'), cosmo=cosmo, cosmo_params={'tau': 'tau_reio'})

    assert like._cosmo_prior_names == ['tau']
    assert 'tau' not in like.params  # not a new desilike Parameter

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}
    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)


def test_lens_likelihood(tmp_path):
    """CandlLensLikelihood constructs, compiles, and evaluates to a finite logpdf
    for a 'kk'-only synthetic data set (no nuisance parameters, no data_model)."""
    _write_candl_lens_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = CandlLensLikelihood(str(tmp_path / 'test_candl_lens.yaml'), cosmo=cosmo)

    assert like.ndata == 3
    assert like.params == {}
    assert like._ellmax_standard == 0
    assert like._ellmax_potential == 10

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}

    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)

    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))


# ── clipy / clik_candl integration tests ─────────────────────────────────────

pytest.importorskip('clipy')


def test_planck_pr3_lowl_ee_sroll2(tmp_path, monkeypatch):
    """Install PlanckPR3LowlEESroll2Likelihood to a temporary directory and run it.

    Downloads the Sroll2 ``.clik`` file (~590 KB) from INFN, installs clipy-like,
    and verifies the compiled logpdf is finite at fiducial parameters.

    Both DESILIKE_CONFIG_DIR and DESILIKE_INSTALL_DIR are redirected to ``tmp_path``
    so the download goes there and does not touch the user's real install.
    """
    from desilike.install import Installer
    from desilike.likelihoods.cmb.candl import PlanckPR3LowlEESroll2Likelihood

    monkeypatch.setenv('DESILIKE_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('DESILIKE_INSTALL_DIR', str(tmp_path))

    PlanckPR3LowlEESroll2Likelihood.install(Installer())

    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=30, non_linear='mead')))
    like = PlanckPR3LowlEESroll2Likelihood(cosmo=cosmo)

    pipe = compile(like)
    defaults = {p.name: p._value for p in get_params(like)}
    logpdf = pipe(defaults)
    assert np.isfinite(float(logpdf)), f'logpdf not finite: {logpdf}'


def test_planck_pr3_lowl_ee_sroll2_ace(tmp_path, monkeypatch):
    """Same Planck PR3 low-ell EE (Sroll2) likelihood served by ACECosmology(engine='ace'),
    i.e. the packaged jaxcapse 'camb_lcdm' Cl emulator: the compiled logpdf must be finite,
    match the camb-based CosmoprimoCosmology one at fiducial parameters, respond to
    tau_reio, and reject (-inf) parameters outside the emulator training ranges."""
    from desilike.install import Installer
    from desilike.likelihoods.cmb.candl import PlanckPR3LowlEESroll2Likelihood
    from desilike.theories.primordial_cosmology import ACECosmology

    monkeypatch.setenv('DESILIKE_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('DESILIKE_INSTALL_DIR', str(tmp_path))

    PlanckPR3LowlEESroll2Likelihood.install(Installer())

    cosmo_camb = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=30, non_linear='mead')))
    like_camb = PlanckPR3LowlEESroll2Likelihood(cosmo=cosmo_camb)
    pipe_camb = compile(like_camb)
    defaults_camb = {p.name: p._value for p in get_params(like_camb)}
    logpdf_camb = float(pipe_camb(defaults_camb))

    cosmo_ace = ACECosmology(engine='ace', fiducial='DESI')
    like_ace = PlanckPR3LowlEESroll2Likelihood(cosmo=cosmo_ace)
    pipe_ace = compile(like_ace)
    defaults_ace = {p.name: p._value for p in get_params(like_ace)}
    logpdf_ace = float(pipe_ace(defaults_ace))

    assert np.isfinite(logpdf_ace), f'logpdf not finite: {logpdf_ace}'
    assert np.isclose(logpdf_ace, logpdf_camb, atol=0.5), f'ACE {logpdf_ace} vs camb {logpdf_camb}'

    # tau_reio dependence, by finite difference: clipy's simall is a floor-indexed table
    # lookup, piecewise-constant in Cl, so jax.grad is exactly 0 by construction (as in the
    # original clik) -- gradient-based use must rely on desilike's finite-difference machinery.
    eps = 1e-3
    fd_tau = (float(pipe_ace({**defaults_ace, 'tau_reio': defaults_ace['tau_reio'] + eps}))
              - float(pipe_ace({**defaults_ace, 'tau_reio': defaults_ace['tau_reio'] - eps}))) / (2. * eps)
    assert np.isfinite(fd_tau) and abs(fd_tau) > 1., f'no tau_reio response: {fd_tau}'

    # out-of-training-range parameters: NaN-masked Cl must reject the sample (-inf), not
    # feed finite garbage through clipy's NaN-clamping table lookup
    assert float(pipe_ace({**defaults_ace, 'h': 0.95})) == -np.inf


if __name__ == '__main__':

    import os
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_likelihood(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_required_nuisance_parameter_uses_candl_prior(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_split_diag_priors(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_split_diag_priors_numerically_equivalent(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_params_override(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cosmo_params_external_prior(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_lens_likelihood(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['DESILIKE_CONFIG_DIR'] = tmp
        os.environ['DESILIKE_INSTALL_DIR'] = tmp
        test_planck_pr3_lowl_ee_sroll2(Path(tmp), type('_MP', (), {'setenv': lambda self, k, v: None})())
    print('All tests passed.')
