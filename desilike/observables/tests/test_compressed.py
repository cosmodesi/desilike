"""Tests for compressed (summary-statistic) galaxy clustering observables."""

import numpy as np
import pytest


def _make_fiducial():
    import cosmoprimo.fiducial as fid
    return fid.DESI(engine='eisenstein_hu')


def _make_cosmo(fiducial):
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    return CosmoprimoCosmology(engine='eisenstein_hu', fiducial=fiducial)


def _run(obs):
    from desilike.base import compile
    pipe = compile(obs)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)
    return pipe_params


# ── BAOCompressionObservable ─────────────────────────────────────────────────

def test_bao_compression_at_fiducial():
    """BAOCompressionObservable: q-ratios equal 1 when cosmo matches the fiducial exactly."""
    from desilike.observables.galaxy_clustering import BAOCompressionObservable

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    parameters = ['DH_over_rd', 'DM_over_rd', 'DV_over_rd', 'DH_over_DM', 'qpar', 'qper', 'qiso', 'qap']
    obs = BAOCompressionObservable(data=None, parameters=parameters, cosmo=cosmo, z=0.5, fiducial=fiducial)

    _run(obs)

    assert obs.flattheory.shape == (len(parameters),)
    assert obs.flatdata.shape == (len(parameters),)
    np.testing.assert_array_equal(obs.flatdata, np.zeros(len(parameters)))
    np.testing.assert_allclose(np.asarray(obs.flattheory)[-4:], 1., rtol=1e-8)
    assert obs.name == 'bao'


def test_bao_compression_with_data_and_covariance():
    """BAOCompressionObservable: array data/covariance are formatted and kept in parameter order."""
    from desilike.observables.galaxy_clustering import BAOCompressionObservable

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    parameters = ['qpar', 'qper']
    data = [1.01, 0.99]
    cov = np.diag([1e-4, 2e-4])
    obs = BAOCompressionObservable(data=data, covariance=cov, parameters=parameters,
                                    cosmo=cosmo, z=0.5, fiducial=fiducial, name='my_bao')

    _run(obs)

    np.testing.assert_array_equal(obs.flatdata, data)
    assert obs.covariance.shape == (2, 2)
    np.testing.assert_allclose(np.diag(obs.covariance), np.diag(cov))
    assert obs.name == 'my_bao'


def test_bao_compression_prebuilt_theory_kwargs_forwarded():
    """BAOCompressionObservable: a prebuilt theory is reused, and extra kwargs update it in place."""
    from desilike.observables.galaxy_clustering import BAOCompressionObservable
    from desilike.theories.galaxy_clustering.template import BAOTheory

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    theory = BAOTheory(z=0.3, fiducial=fiducial, cosmo=cosmo)
    obs = BAOCompressionObservable(parameters=['qpar'], theory=theory, z=0.8)

    assert obs.theory is theory
    _run(obs)
    assert obs.theory.z == 0.8


# ── BAOPhaseShiftCompressionObservable ───────────────────────────────────────

def test_bao_phaseshift_compression_at_fiducial():
    """BAOPhaseShiftCompressionObservable: qpar and baoshift equal 1 at the fiducial cosmology."""
    from desilike.observables.galaxy_clustering import BAOPhaseShiftCompressionObservable

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    parameters = ['qpar', 'qper', 'N_eff', 'baoshift']
    obs = BAOPhaseShiftCompressionObservable(data=None, parameters=parameters,
                                              cosmo=cosmo, z=0.5, fiducial=fiducial)

    _run(obs)

    values = dict(zip(parameters, np.asarray(obs.flattheory)))
    np.testing.assert_allclose(values['qpar'], 1., rtol=1e-8)
    np.testing.assert_allclose(values['qper'], 1., rtol=1e-8)
    np.testing.assert_allclose(values['baoshift'], 1., rtol=1e-8)
    assert obs.name == 'baoshift'


# ── TurnOverCompressionObservable ────────────────────────────────────────────

def test_turnover_compression_at_fiducial():
    """TurnOverCompressionObservable: qap equals 1 exactly, qto is close to 1 (grid discretization)."""
    from desilike.observables.galaxy_clustering import TurnOverCompressionObservable

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    parameters = ['qto', 'qap']
    obs = TurnOverCompressionObservable(data=None, parameters=parameters, cosmo=cosmo, z=0.5, fiducial=fiducial)

    _run(obs)

    values = dict(zip(parameters, np.asarray(obs.flattheory)))
    np.testing.assert_allclose(values['qap'], 1., rtol=1e-8)
    np.testing.assert_allclose(values['qto'], 1., atol=1e-3)
    assert obs.name == 'turnover'


# ── Data / covariance formatting ─────────────────────────────────────────────

def test_missing_parameters_raises():
    """BaseCompressionObservable: array/None data without explicit parameters raises."""
    from desilike.observables.galaxy_clustering.compressed import _format_compression_data

    with pytest.raises(ValueError):
        _format_compression_data(data=None, covariance=None, parameters=None)
    with pytest.raises(ValueError):
        _format_compression_data(data=[1., 2.], covariance=None, parameters=None)


def test_covariance_shape_mismatch_raises():
    """BaseCompressionObservable: covariance shape inconsistent with data size raises."""
    from desilike.observables.galaxy_clustering.compressed import _format_compression_data

    with pytest.raises(ValueError):
        _format_compression_data(data=[1., 2.], covariance=np.eye(3), parameters=['qpar', 'qper'])


def test_lsstypes_observable_data_input():
    """BaseCompressionObservable: lsstypes ObservableLike input sets parameters and flatdata directly."""
    import lsstypes as types
    from desilike.observables.galaxy_clustering import BAOCompressionObservable

    parameters = ['qpar', 'qper']
    values = [1.02, 0.98]
    leaves = [types.ObservableLeaf(value=np.atleast_1d(v)) for v in values]
    data = types.ObservableTree(leaves, parameters=parameters)

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    obs = BAOCompressionObservable(data=data, cosmo=cosmo, z=0.5, fiducial=fiducial)

    assert obs.parameters == parameters
    np.testing.assert_array_equal(obs.flatdata, values)


# ── ObservablesGaussianLikelihood ────────────────────────────────────────────

def test_gaussian_likelihood_with_bao_compression():
    """ObservablesGaussianLikelihood: BAOCompressionObservable plugs in like any other observable."""
    from desilike.observables.galaxy_clustering import BAOCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile

    fiducial = _make_fiducial()
    cosmo = _make_cosmo(fiducial)
    parameters = ['qpar', 'qper']
    cov = np.diag([1e-4, 1e-4])
    # Data matches the fiducial prediction exactly (q-ratios are 1 there).
    obs = BAOCompressionObservable(data=[1., 1.], covariance=cov, parameters=parameters,
                                    cosmo=cosmo, z=0.5, fiducial=fiducial)
    like = ObservablesGaussianLikelihood(observables=obs)

    pipe = compile(like)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert like.flattheory.shape == (2,)
    np.testing.assert_allclose(like.flattheory, like.flatdata, atol=1e-8)
    np.testing.assert_allclose(like.precision, np.linalg.inv(cov), rtol=1e-10)
    assert np.isfinite(float(like.logpdf))


def test_gaussian_likelihood_multi_compressed_observable():
    """ObservablesGaussianLikelihood: BAO and turn-over compressed observables are concatenated correctly."""
    from desilike.observables.galaxy_clustering import BAOCompressionObservable, TurnOverCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile

    fiducial = _make_fiducial()
    cosmo_bao = _make_cosmo(fiducial)
    cosmo_to = _make_cosmo(fiducial)

    obs_bao = BAOCompressionObservable(data=[1., 1.], parameters=['qpar', 'qper'],
                                        cosmo=cosmo_bao, z=0.5, fiducial=fiducial, name='bao')
    obs_to = TurnOverCompressionObservable(data=[1.], parameters=['qap'],
                                            cosmo=cosmo_to, z=0.5, fiducial=fiducial, name='turnover')

    cov = np.diag([1e-4, 1e-4, 1e-4])
    like = ObservablesGaussianLikelihood(observables=[obs_bao, obs_to], covariance=cov)

    pipe = compile(like)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert like.flatdata.shape == (3,)
    assert like.flattheory.shape == (3,)
    assert np.isfinite(float(like.logpdf))


def test_posterior_camb_invalid_omega_cdm():
    """Posterior + BAOCompressionObservable with engine='camb': an unphysical point
    (omega_cdm < 0) degrades to -inf, it does not crash.

    engine='camb' is an *external* Boltzmann code, run through pure_callback with concrete
    (non-Tracer) values. Under jax.jit, Posterior always builds the full graph regardless of
    the prior (see Posterior.__call__), so this exercises CosmoprimoCosmology's external-engine
    fallback for unphysical input (CosmologyInputError -> NaN, instead of raising) even though
    omega_cdm=-0.05 is also outside the default prior support.
    """
    import jax
    import cosmoprimo.fiducial as fid
    from desilike.observables.galaxy_clustering import BAOCompressionObservable
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile, Posterior, get_params

    fiducial = fid.DESI(engine='camb')
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=fiducial)
    obs = BAOCompressionObservable(data=[1., 1.], parameters=['qpar', 'qper'],
                                    cosmo=cosmo, z=0.5, fiducial=fiducial)
    like = ObservablesGaussianLikelihood(observables=obs, covariance=np.diag([1e-4, 1e-4]))
    post = Posterior(like)

    pipe = compile(post)
    defaults = {p.name: float(p._value) for p in get_params(pipe)}

    # Sanity: the fiducial point is finite.
    logpdf_fiducial = pipe(defaults)
    assert np.isfinite(float(logpdf_fiducial))

    bad = dict(defaults)
    bad['omega_cdm'] = -0.05

    logpdf_jit = jax.jit(pipe)(bad)
    assert float(logpdf_jit) == -np.inf
