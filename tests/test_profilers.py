import numpy as np
import pytest
from jax import numpy as jnp

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.profilers import optimize, Profiler

MEAN_PRIOR = np.array([+0.2, -0.1])
SD_PRIOR = np.array([0.1, 0.05])
COV_PRIOR = np.diag(SD_PRIOR**2)
MEAN_LIKELIHOOD = np.array([-0.2, +0.2])
COV_LIKELIHOOD = np.array([[0.01, 0.01], [0.01, 0.02]])
SD_LIKELIHOOD = np.sqrt(np.diag(COV_LIKELIHOOD))
COV_POSTERIOR = np.linalg.inv(np.linalg.inv(COV_PRIOR) +
                              np.linalg.inv(COV_LIKELIHOOD))
SD_POSTERIOR = np.sqrt(np.diag(COV_POSTERIOR))
MEAN_POSTERIOR = COV_POSTERIOR @ (
    np.linalg.inv(COV_PRIOR) @ MEAN_PRIOR +
    np.linalg.inv(COV_LIKELIHOOD) @ MEAN_LIKELIHOOD)

OPTIMIZE = dict(
    bobyqa=optimize.optimize_bobyqa,
    dual_annealing=optimize.optimize_dual_annealing,
    minuit=optimize.optimize_minuit,
    optax=optimize.optimize_optax,
    scipy=optimize.optimize_scipy)
OPTIMIZE_KWARGS = dict(
    bobyqa=None,
    dual_annealing=dict(maxiter=10),
    minuit=None,
    optax=None,
    scipy=None)


@pytest.fixture
def likelihood():

    class Likelihood(BaseGaussianLikelihood):

        def calculate(self, **kwargs):
            self.flattheory = jnp.array([kwargs[name] for name in ['a', 'b']])
            super().calculate()

    likelihood = Likelihood(MEAN_LIKELIHOOD, covariance=COV_LIKELIHOOD)
    likelihood.init.params = dict(
        a=dict(prior=dict(dist='norm', limits=[-3, +3], loc=MEAN_PRIOR[0],
                          scale=SD_PRIOR[0])),
        b=dict(prior=dict(dist='norm', limits=[-3, +3], loc=MEAN_PRIOR[1],
                          scale=SD_PRIOR[1])))

    return likelihood


@pytest.mark.mpi
@pytest.mark.parametrize('posterior', [True, False])
@pytest.mark.parametrize('key', OPTIMIZE.keys())
def test_accuracy(likelihood, posterior, key):
    # Test that the profiler returns the correct result.

    profiler = Profiler(likelihood, rng=42, posterior=posterior)
    profiler.add_optimize_all()
    profiler.add_single_sample(dict(a=0.35))
    profiler.add_manual_grid(dict(a=np.linspace(-1, +1, 3)))
    profiler.add_manual_grid(
        dict(a=np.linspace(-1, +1, 4), b=np.linspace(-1, +1, 5)))
    samples = profiler.run(
        optimize=OPTIMIZE[key], optimize_kwargs=OPTIMIZE_KWARGS[key])

    if posterior:
        key = 'log_posterior'
    else:
        key = 'log_likelihood'

    if posterior:
        # Correct normalization such that the log posterior of the best fit
        # is 0.
        samples[key] -= likelihood(dict(
            a=MEAN_POSTERIOR[0], b=MEAN_POSTERIOR[1]))

    # Check the maximum likelihood/posterior has been found.
    assert np.isclose(np.amax(samples[key]), 0, rtol=0, atol=1e-6)
    mean = dict(zip(['a', 'b'], MEAN_POSTERIOR if posterior else
                    MEAN_LIKELIHOOD))
    for param in ['a', 'b']:
        assert np.isclose(samples[np.argmax(samples[key])][param], mean[param],
                          rtol=0, atol=1e-6)

    sd = dict(zip(['a', 'b'], SD_POSTERIOR if posterior else SD_LIKELIHOOD))
    use = (~samples.get_flag('optimize', 'a') &
           samples.get_flag('optimize', 'b'))
    assert np.sum(use) == 4
    assert np.allclose(
        -0.5 * ((samples['a'] - mean['a'])**2 / sd['a']**2)[use],
        samples[key][use], rtol=0, atol=1e-6)

    # Check the interpolation works.
    interp = samples.profile_interpolator('a', posterior=posterior)
    assert len(interp.x) == 5  # 1 (global) + 1 (single) + 3 (grid)
    a = np.linspace(-1, +1, 100)
    assert np.allclose(-0.5 * ((a - mean['a'])**2 / sd['a']**2), interp(a),
                       rtol=0, atol=1e-6)

    for threshold, sigma in zip([-0.5, -2, -4.5], [1, 2, 3]):
        bounds = samples.interval('a', threshold, posterior=posterior)
        assert len(bounds) == 1
        assert np.isclose(bounds[0][0], mean['a'] - sigma * sd['a'], rtol=0,
                          atol=1e-6)
        assert np.isclose(bounds[0][1], mean['a'] + sigma * sd['a'], rtol=0,
                          atol=1e-6)

    interp = samples.profile_interpolator(['a', 'b'], posterior=posterior)
    np.random.seed(42)
    ab = np.column_stack((np.random.uniform(-1, +1, 100),
                          np.random.uniform(-1, +1, 100)))
    mean = MEAN_POSTERIOR if posterior else MEAN_LIKELIHOOD
    cov = COV_POSTERIOR if posterior else COV_LIKELIHOOD
    assert np.allclose(-0.5 * np.einsum('...i,...i', np.einsum(
        'ij,jk', ab - mean, np.linalg.inv(cov)), ab - mean), interp(ab),
        rtol=0, atol=1e-6)


@pytest.mark.mpi_skip
def test_rng(likelihood):
    # Test that the profiler is deterministic.

    optimize_kwargs = dict(maxiter=1, no_local_search=True)

    profiler_1 = Profiler(likelihood, rng=42)
    profiler_1.add_manual_grid(dict(a=np.linspace(0, 1, 20)))
    samples_1 = profiler_1.run(optimize_kwargs=optimize_kwargs)

    profiler_2 = Profiler(likelihood, rng=42)
    profiler_2.add_manual_grid(dict(a=np.linspace(0, 1, 20)))
    samples_2 = profiler_2.run(optimize_kwargs=optimize_kwargs)

    assert len(samples_1) == len(samples_2)

    assert np.allclose(samples_1['log_posterior'], samples_2['log_posterior'])


@pytest.mark.mpi_skip
def test_remove_duplicates(likelihood):
    # Test that duplicates are successfully removed.

    profiler = Profiler(likelihood, rng=42)
    profiler.add_optimize_all()
    profiler.add_optimize_all()  # shouldn't be added
    profiler.add_manual_grid(dict(a=np.linspace(0, 1, 3)))
    profiler.add_manual_grid(
        dict(a=np.linspace(0, 1, 3)))  # shouldn't be added
    profiler.add_manual_grid(dict(b=np.linspace(0, 1, 5)))
    assert len(profiler.samples) == 1 + 3 + 5


@pytest.mark.mpi_skip
def test_write(likelihood, tmp_path):
    # Test that the profiler results are written correctly.

    optimize_kwargs = dict(maxiter=1, no_local_search=True)

    profiler_1 = Profiler(likelihood, rng=42, directory=tmp_path)
    profiler_1.add_manual_grid(dict(a=np.linspace(0, 1, 20)))
    samples_1 = profiler_1.run(optimize_kwargs=optimize_kwargs)

    profiler_2 = Profiler(likelihood, directory=tmp_path)
    samples_2 = profiler_2.run(n_per_iter=0)

    assert len(samples_1) == len(samples_2)
    assert np.allclose(samples_1['log_posterior'], samples_2['log_posterior'])
