import numpy as np
import pytest
from emcee.autocorr import integrated_time

from desilike.samplers.mhmcmc import FastSlowProposer
from desilike.samplers.mhmcmc import StandAloneMetropolisHastingsSampler


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n_fast", [0, 1, 2, 3, 4, 5])
def test_proposer(n_fast):
    # Test that the proposals work correctly.
    n_dim = 5
    n_samples = 10000
    rng = np.random.default_rng(42)
    cov = np.cov(rng.normal(size=(10, 5)), rowvar=False)
    fast = rng.choice(np.arange(n_dim), size=n_fast, replace=False)
    prop = FastSlowProposer(cov, fast=fast, rng=rng)

    steps_fast = np.vstack([prop.propose_fast() for i in range(n_samples)])
    steps_slow = np.vstack([prop.propose_slow() for i in range(n_samples)])

    # Slow parameters should not be changed for fast proposals.
    assert np.allclose(steps_fast[:, ~np.isin(range(n_dim), fast)], 0)

    if len(steps_slow) == 0:
        steps_slow = np.zeros((1, n_dim))
    if len(steps_fast) == 0:
        steps_fast = np.zeros((1, n_dim))

    steps_fast = steps_fast[rng.choice(len(steps_fast), n_samples * n_dim)]
    steps_slow = steps_slow[rng.choice(len(steps_slow), n_samples * n_dim)]
    cov_err = np.sqrt(
        (cov**2 + np.outer(np.diag(cov), np.diag(cov))) / n_samples)

    # The proposal distribution should match the input.
    assert np.allclose(cov, np.cov(steps_fast + steps_slow, rowvar=False),
                       atol=5 * cov_err)


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "n_fast, f_fast, f_drag", [[0, 1, 0], [1, 1, 0], [1, 3, 0], [1, 1, 3]])
def test_rosenbrock(n_fast, f_fast, f_drag):
    """Test that the sampler works correctly on a 2-D Rosenbrock likelihood.

    The true mean and covariance can be computed as follows.

    >>> x = np.linspace(-10, +10, 10000)
    >>> y = np.linspace(-10, +10, 10000)
    >>> xx, yy = np.meshgrid(x, y)
    >>> x = xx.ravel()
    >>> y = yy.ravel()
    >>> w = np.exp(posterior(np.column_stack((x, y))))
    >>> cov = np.cov(np.column_stack((x, y)), aweights=w, rowvar=False)
    >>> mean = np.average(np.column_stack((x, y)), weights=w, axis=0)

    """

    def posterior(x):
        return -((1 - x[..., 0])**2 + (x[..., 1] - x[..., 0]**2)**2)

    rng = np.random.default_rng(42)

    n_dim = 2
    cov = np.array([[0.49379419, 0.9720092], [0.9720092, 2.87807935]])
    mean = np.array([0.99705773, 1.48744377])

    fast = rng.choice(np.arange(n_dim), size=n_fast, replace=False)
    sampler = StandAloneMetropolisHastingsSampler(
        posterior, f_fast=f_fast, f_drag=f_drag, fast=fast, rng=rng)
    sampler.update(pos=np.zeros((10, 2)), cov=cov)

    chains, log_p = sampler.make_n_steps(30000)
    chains = chains[:, 1000:]  # burn-in

    tau = np.amax(integrated_time(np.transpose(chains, (1, 0, 2))))
    chains = np.concatenate(chains)
    n_eff = len(chains) / tau
    assert n_eff > 1000
    mean_err = np.sqrt(np.diag(cov) / n_eff)
    cov_err = np.sqrt((cov**2 + np.outer(np.diag(cov), np.diag(cov))) / n_eff)

    assert np.allclose(np.mean(chains, axis=0), mean, atol=5 * mean_err)
    assert np.allclose(np.cov(chains, rowvar=False), cov, atol=10 * cov_err)
