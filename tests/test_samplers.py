import numpy as np
import pytest
from jax import numpy as jnp

from desilike import Samples
import desilike.samplers as samplers
from desilike.likelihoods import BaseGaussianLikelihood


SAMPLER_CLS = dict(
    dynesty=samplers.DynestySampler,
    emcee=samplers.EmceeSampler,
    grid=samplers.GridSampler,
    hmc=samplers.HMCSampler,
    importance=samplers.ImportanceSampler,
    mclmc=samplers.MCLMCSampler,
    mhmcmc=samplers.MetropolisHastingsSampler,
    nautilus=samplers.NautilusSampler,
    nuts=samplers.NoUTurnSampler,
    pocomc=samplers.PocoMCSampler,
    qmc=samplers.QMCSampler,
    zeus=samplers.ZeusSampler)
KWARGS_INIT = dict(
    dynesty=dict(dynamic=True, nlive=100),
    nautilus=dict(n_networks=1, n_live=300),
    pocomc=dict(n_effective=200, n_active=100))
KWARGS_INIT_FAST = dict(
    dynesty=dict(dynamic=True, nlive=30),
    nautilus=dict(n_networks=1, n_live=100),
    pocomc=dict(n_effective=10, n_active=5, flow='nsf3'))
KWARGS_RUN = dict(
    dynesty=dict(n_effective=0),
    grid=dict(grid=np.linspace(0, 1, 50)),
    importance=dict(samples=Samples(
        a=np.repeat(np.linspace(0, 1, 50), 50),
        b=np.tile(np.linspace(0, 1, 50), 50),
        log_posterior=np.zeros(50**2))),
    nautilus=dict(n_eff=100),
    pocomc=dict(n_total=100, n_evidence=100),
    qmc=dict(size=10000))
KWARGS_RUN_FAST = dict(
    dynesty=dict(maxiter=10),
    importance=dict(samples=Samples(
        a=np.repeat(np.linspace(0, 1, 11), 11),
        b=np.tile(np.linspace(0, 1, 11), 11),
        log_posterior=np.zeros(11**2))),
    emcee=dict(max_steps=10),
    grid=dict(grid=np.linspace(0, 1, 11)),
    hmc=dict(max_steps=10),
    mclmc=dict(max_steps=10),
    mhmcmc=dict(max_steps=10),
    nautilus=dict(n_eff=0, n_like_max=100),
    nuts=dict(max_steps=10),
    pocomc=dict(n_total=10, n_evidence=0),
    qmc=dict(size=100),
    zeus=dict(max_steps=10))


@pytest.fixture
def likelihood():

    class Likelihood(BaseGaussianLikelihood):

        def calculate(self, **kwargs):
            self.flattheory = jnp.array([kwargs[name] for name in ['a', 'b']])
            self.c = kwargs['a'] + kwargs['b']
            self.d = np.arange(3) * self.c
            super().calculate()

    likelihood = Likelihood(np.array([0.4, 0.6]), covariance=np.eye(2) * 0.01)
    likelihood.init.params = dict(
        a=dict(prior=dict(dist='norm', limits=[0, 1], loc=0.4, scale=0.1)),
        b=dict(prior=dict(dist='uniform', limits=[0, 1])),
        c=dict(derived=True), d=dict(derived=True, shape=(3, )))

    return likelihood


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_accuracy(likelihood, key):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    sampler = SAMPLER_CLS[key](likelihood, rng=42, **KWARGS_INIT.get(key, {}))
    results = sampler.run(**KWARGS_RUN.get(key, {}))

    # The mean should match.
    keys = likelihood.varied_params.names()
    assert np.allclose(results.mean(keys), likelihood.flatdata, atol=0.05,
                       rtol=0)
    # The covariance should match.
    cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
    cov_err = np.sqrt(
        (cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
    assert np.allclose(results.covariance(keys), cov, atol=3 * cov_err)


@pytest.mark.mpi
def test_importance_combine(likelihood):
    # Test that importance sampling can combine two likelihood without
    # double counting the prior. We use Gaussian quadrature integration to
    # speed up results.

    deg = 30
    x, weight = np.polynomial.legendre.leggauss(deg)
    x = (x + 1) / 2  # shift from [-1, +1] to [0, 1]
    weight /= 2
    log_weight = np.log(np.outer(weight, weight).flatten())

    sampler = samplers.GridSampler(likelihood)
    results = sampler.run(grid=x)
    results['log_weight'] += log_weight

    sampler = samplers.ImportanceSampler(likelihood)
    results = sampler.run(samples=results, resample=False)

    cov = np.linalg.inv(2 * likelihood.precision +
                        np.array([[100, 0], [0, 0]]))
    keys = likelihood.varied_params.names()
    assert np.allclose(results.mean(keys), likelihood.flatdata, atol=1e-6)
    assert np.allclose(results.covariance(keys), cov, atol=1e-3)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_derived(likelihood, key):
    # Test that derived parameters are correctly tracked.

    sampler = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results = sampler.run(**KWARGS_RUN_FAST.get(key, {}))
    assert np.allclose(results['a'] + results['b'], results['c'])
    for i in range(3):
        assert np.allclose((results['a'] + results['b']) * i,
                           results['d'][:, i])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_write(likelihood, key, tmp_path):
    # Check that the sampler correctly saves results and state, if applicable.

    sampler_1 = SAMPLER_CLS[key](
        likelihood, rng=42, directory=tmp_path,
        **KWARGS_INIT_FAST.get(key, {}))
    results_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    # The second sampler should not create any new samples if old results
    # are read correctly.
    sampler_2 = SAMPLER_CLS[key](
        likelihood, rng=43, directory=tmp_path,
        **KWARGS_INIT_FAST.get(key, {}))
    results_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    assert len(results_1) == len(results_2)
    statistic = ('log_posterior' if 'log_posterior' in results_1.keys else
                 'log_likelihood')
    assert np.allclose(results_1[statistic], results_2[statistic], atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_rng(likelihood, key):
    # Test that specifying the random seed leads to reproducible results.

    if key == 'zeus':
        pytest.skip("Zeus does not support specifying a random seed.")

    sampler_1 = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    assert len(results_1) == len(results_2)

    statistic = ('log_posterior' if 'log_posterior' in results_1.keys else
                 'log_likelihood')
    assert np.allclose(results_1[statistic], results_2[statistic], atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee', 'hmc', 'mhmcmc', 'zeus'])
def test_continue_chain(likelihood, key):
    # Test that we can continue a chain.

    sampler = SAMPLER_CLS[key](likelihood, rng=42)
    chains_10 = sampler.run(
        burn_in=0, min_steps=10, max_steps=10, flatten_chains=False)
    sampler = samplers.MetropolisHastingsSampler(
        likelihood, rng=43, chains=[c.copy() for c in chains_10])
    chains_20 = sampler.run(
        burn_in=0, min_steps=20, max_steps=20, flatten_chains=False)

    for chain_10, chain_20 in zip(chains_10, chains_20):
        assert len(chain_10) == 10
        assert len(chain_20) == 20
        assert np.allclose(chain_10['a'], chain_20['a'][:10])


@pytest.mark.mpi_skip
def test_metropolis_hastings_fast(likelihood):
    # Test we can pass fast parameters to the Metropolis-Hastings sampler.

    sampler = samplers.MetropolisHastingsSampler(
        likelihood, rng=42, fast=['a'], f_fast=1)
    sampler.run(max_steps=100)
