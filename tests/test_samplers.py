import numpy as np
import pytest
from jax import numpy as jnp

import desilike.samplers as samplers
from desilike.likelihoods import BaseGaussianLikelihood


SAMPLER_CLS = dict(
    dynesty=samplers.DynestySampler,
    emcee=samplers.EmceeSampler,
    grid=samplers.GridSampler,
    hmc=samplers.HMCSampler,
    nautilus=samplers.NautilusSampler,
    pocomc=samplers.PocoMCSampler,
    zeus=samplers.ZeusSampler)
ARGS_INIT = dict(
    emcee=(10, ),
    hmc=(10, ),
    nuts=(10, ),
    zeus=(10, ))
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
    grid=dict(size=100),
    nautilus=dict(n_eff=100),
    pocomc=dict(n_total=100, n_evidence=100))
KWARGS_RUN_FAST = dict(
    dynesty=dict(n_effective=0),
    emcee=dict(max_iterations=100),
    grid=dict(size=100),
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
        a=dict(prior=dict(dist='uniform', limits=[0, 1])),
        b=dict(prior=dict(dist='uniform', limits=[0, 1])))

    return likelihood


@pytest.mark.parametrize("key", [
    'dynesty', 'emcee', 'grid', 'hmc', 'nautilus', 'nuts', 'pocomc', 'zeus'])
def test_accuracy(likelihood, key):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    sampler = SAMPLER_CLS[key](likelihood, *ARGS_INIT.get(key, ()), rng=42,
                               **KWARGS_INIT.get(key, {}))
    chain = sampler.run(**KWARGS_RUN_FAST.get(key, {}))

    if isinstance(sampler, samplers.GridSampler):
        chain.aweight = np.exp(chain.logposterior)

    # The mean should match.
    assert np.allclose(chain.mean(likelihood.varied_params),
                       likelihood.flatdata, atol=0.05, rtol=0)
    # The covariance should match.
    assert np.allclose(chain.covariance(likelihood.varied_params),
                       np.linalg.inv(likelihood.precision), atol=0.01,
                       rtol=0.1)


@pytest.mark.parametrize("key", [
    'dynesty', 'emcee', 'nautilus', 'pocomc', 'zeus'])
def test_save_fn(likelihood, key, tmp_path):
    # Check that the sampler correctly saves chains and state, if applicable.

    sampler_1 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT.get(key, ()), rng=42,
        save_fn=str(tmp_path / 'checkpoint_*.npz'),
        **KWARGS_INIT_FAST.get(key, {}))
    if key != 'pocomc':
        chain_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))
    else:
        chain_1 = sampler_1.run(save_every=1, **KWARGS_RUN_FAST.get(key, {}))
    # The second sampler should not create any new samples if old chains
    # are read correctly.
    sampler_2 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT.get(key, ()), rng=43,
        save_fn=str(tmp_path / 'checkpoint_*.npz'),
        **KWARGS_INIT_FAST.get(key, {}))
    if key != 'pocomc':
        chain_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))
    else:
        chain_2 = sampler_2.run(
            resume_state_path=str(sampler_2.path('sampler_final', 'state')),
            **KWARGS_RUN_FAST.get(key, {}))

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)


@pytest.mark.parametrize("key", [
    'dynesty', 'emcee', 'nautilus', 'pocomc', 'zeus'])
def test_rng(likelihood, key):
    # Test that specifying the random seed leads to reproducible results.

    if key == 'zeus':
        pytest.skip("Zeus does not support specifying a random seed.")

    sampler_1 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT.get(key, ()), rng=42,
        **KWARGS_INIT_FAST.get(key, {}))
    chain_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT.get(key, ()), rng=42,
        **KWARGS_INIT_FAST.get(key, {}))
    chain_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)
