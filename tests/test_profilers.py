import numpy as np
import pytest
from jax import numpy as jnp

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.profilers import Profiler

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
def test_accuracy(likelihood, posterior):
    # Test that the profiler returns the correct result.

    profiler = Profiler(likelihood, rng=42, posterior=posterior)
    profiler.add_optimize_all()
    profiler.add_grid_manual(dict(a=np.linspace(-1, +1, 5)))
    samples = profiler.run(optimizer_kwargs=dict(maxiter=10))

    if posterior:
        key = 'log_posterior'
    else:
        key = 'log_likelihood'

    if posterior:
        samples[key] -= likelihood(dict(
            a=MEAN_POSTERIOR[0], b=MEAN_POSTERIOR[1]))  # correct normalization

    assert np.isclose(np.amax(samples[key]), 0, rtol=0, atol=1e-6)

    if posterior:
        truth = -0.5 * ((samples['a'] - MEAN_POSTERIOR[0])**2 /
                        SD_POSTERIOR[0]**2)
    else:
        truth = -0.5 * ((samples['a'] - MEAN_LIKELIHOOD[0])**2 /
                        SD_LIKELIHOOD[0]**2)

    use = ~samples.get_flag('optimize', 'a')
    assert np.sum(use) == 5
    assert np.allclose(truth[use], samples[key][use], rtol=0, atol=1e-6)


@pytest.mark.mpi_skip
def test_rng(likelihood):
    # Test that the profiler is deterministic.

    optimizer_kwargs = dict(maxiter=1, no_local_search=True)

    profiler_1 = Profiler(likelihood, rng=42)
    profiler_1.add_grid_manual(dict(a=np.linspace(0, 1, 20)))
    samples_1 = profiler_1.run(optimizer_kwargs=optimizer_kwargs)

    profiler_2 = Profiler(likelihood, rng=42)
    profiler_2.add_grid_manual(dict(a=np.linspace(0, 1, 20)))
    samples_2 = profiler_2.run(optimizer_kwargs=optimizer_kwargs)

    assert len(samples_1) == len(samples_2)

    assert np.allclose(samples_1['log_posterior'], samples_2['log_posterior'])


@pytest.mark.mpi_skip
def test_remove_duplicates(likelihood):
    # Test that duplicates are successfully removed.

    profiler = Profiler(likelihood, rng=42)
    profiler.add_optimize_all()
    profiler.add_optimize_all()  # shouldn't be added
    profiler.add_grid_manual(dict(a=np.linspace(0, 1, 3)))
    profiler.add_grid_manual(
        dict(a=np.linspace(0, 1, 3)))  # shouldn't be added
    profiler.add_grid_manual(dict(b=np.linspace(0, 1, 5)))
    assert len(profiler.samples) == 1 + 3 + 5


@pytest.mark.mpi_skip
def test_write(likelihood, tmp_path):
    # Test that the profiler results are written correctly.

    optimizer_kwargs = dict(maxiter=1, no_local_search=True)

    profiler_1 = Profiler(likelihood, rng=42, directory=tmp_path)
    profiler_1.add_grid_manual(dict(a=np.linspace(0, 1, 20)))
    samples_1 = profiler_1.run(optimizer_kwargs=optimizer_kwargs)

    profiler_2 = Profiler(likelihood, directory=tmp_path)
    samples_2 = profiler_2.run(n_per_iter=0)

    assert len(samples_1) == len(samples_2)
    assert np.allclose(samples_1['log_posterior'], samples_2['log_posterior'])
