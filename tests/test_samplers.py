import numpy as np
import pytest
from jax import numpy as jnp

import desilike.samplers as samplers
from desilike.samples import Chain
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
    nuts=samplers.NUTSSampler,
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
    pocomc=dict(n_effective=100, n_active=50))
KWARGS_RUN = dict(
    dynesty=dict(n_effective=0),
    grid=dict(grid=np.linspace(0, 1, 101)),
    importance=dict(chain=Chain(dict(
        a=np.repeat(np.linspace(0, 1, 101), 101),
        b=np.tile(np.linspace(0, 1, 101), 101)))),
    nautilus=dict(n_eff=100),
    pocomc=dict(n_total=100, n_evidence=100),
    qmc=dict(size=10000))
KWARGS_RUN_FAST = dict(
    dynesty=dict(n_effective=0),
    importance=dict(chain=Chain(dict(
        a=np.repeat(np.linspace(0, 1, 11), 101),
        b=np.tile(np.linspace(0, 1, 11), 101)))),
    emcee=dict(max_iterations=100),
    grid=dict(grid=np.linspace(0, 1, 101)),
    nautilus=dict(n_eff=0, n_like_max=100),
    pocomc=dict(n_total=10, n_evidence=10),
    zeus=dict(max_iterations=100))


@pytest.fixture
def likelihood():

    class Likelihood(BaseGaussianLikelihood):

        def calculate(self, **kwargs):
            self.flattheory = jnp.array([kwargs[name] for name in
                                         self.params.names()])
            super().calculate()

    likelihood = Likelihood(np.array([0.4, 0.6]), covariance=np.eye(2) * 0.01)
    likelihood.init.params = dict(
        a=dict(prior=dict(dist='norm', limits=[0, 1], loc=0.4, scale=0.1)),
        b=dict(prior=dict(dist='uniform', limits=[0, 1])))

    return likelihood


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_accuracy(likelihood, key):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    sampler = SAMPLER_CLS[key](likelihood, rng=42, **KWARGS_INIT.get(key, {}))
    chain = sampler.run(**KWARGS_RUN.get(key, {}))

    # The mean should match.
    assert np.allclose(chain.mean(likelihood.varied_params),
                       likelihood.flatdata, atol=0.05, rtol=0)
    # The covariance should match.
    cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
    cov_err = np.sqrt(
        (cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
    assert np.allclose(chain.covariance(likelihood.varied_params), cov,
                       atol=3 * cov_err)


@pytest.mark.mpi_skip
def test_importance_combine(likelihood):
    # Test that importance sampling can combine two likelihood without
    # double counting the prior.

    sampler = samplers.GridSampler(likelihood)
    chain = sampler.run(grid=np.linspace(0, 1, 101))

    sampler = samplers.ImportanceSampler(likelihood)
    chain = sampler.run(chain=chain, mode='combine')

    cov = np.linalg.inv(2 * likelihood.precision +
                        np.array([[100, 0], [0, 0]]))
    assert np.allclose(chain.mean(likelihood.varied_params),
                       likelihood.flatdata, atol=1e-3, rtol=0)
    assert np.allclose(chain.covariance(likelihood.varied_params), cov,
                       atol=1e-3)


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_write(likelihood, key, tmp_path):
    # Check that the sampler correctly saves chains and state, if applicable.

    sampler_1 = SAMPLER_CLS[key](
        likelihood, rng=42, directory=tmp_path,
        **KWARGS_INIT_FAST.get(key, {}))
    chain_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    # The second sampler should not create any new samples if old chains
    # are read correctly.
    sampler_2 = SAMPLER_CLS[key](
        likelihood, rng=43, directory=tmp_path,
        **KWARGS_INIT_FAST.get(key, {}))
    chain_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_rng(likelihood, key):
    # Test that specifying the random seed leads to reproducible results.

    if key == 'zeus':
        pytest.skip("Zeus does not support specifying a random seed.")

    sampler_1 = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    chain_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    chain_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)
