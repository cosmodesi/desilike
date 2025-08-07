import numpy as np
import pytest

import desilike.samplers as samplers
from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood


SAMPLER_CLS = dict(
    dynesty=samplers.DynestySampler,
    emcee=samplers.EmceeSampler,
    grid=samplers.GridSampler,
    nautilus=samplers.NautilusSampler,
    pocomc=samplers.PocoMCSampler,
    zeus=samplers.ZeusSampler)
ARGS_INIT = dict(
    dynesty=(),
    emcee=(10, ),
    grid=(),
    nautilus=(),
    pocomc=(),
    zeus=(10, ))
KWARGS_INIT = dict(
    dynesty=dict(dynamic=True, nlive=100),
    emcee=dict(),
    grid=dict(),
    nautilus=dict(n_networks=1, n_live=300),
    pocomc=dict(n_effective=200, n_active=100),
    zeus=dict())
KWARGS_RUN = dict(
    dynesty=dict(n_effective=0),
    emcee=dict(),
    grid=dict(size=100),
    nautilus=dict(n_eff=100),
    pocomc=dict(n_total=100, n_evidence=100),
    zeus=dict())
KWARGS_RUN_FAST = dict(
    dynesty=dict(n_effective=0),
    emcee=dict(max_iterations=100),
    grid=dict(size=100),
    nautilus=dict(n_eff=0, n_like_max=100),
    pocomc=dict(n_total=100, n_evidence=100),
    zeus=dict(max_iterations=100))


@pytest.fixture
def likelihood():

    class Model(BaseCalculator):

        _params = dict(a=dict(prior=dict(dist='uniform', limits=[0, 1]),
                              value=0.5, proposal=0.5),
                       b=dict(prior=dict(dist='uniform', limits=[0, 1]),
                              value=0.5, proposal=0.5))

        def initialize(self, x=None):
            pass

        def calculate(self, a=0., b=0.):
            self.y = np.array([a, b])

    class Likelihood(BaseGaussianLikelihood):

        def initialize(self):
            super(Likelihood, self).initialize(
                np.array([0.4, 0.6]), covariance=np.eye(2) * 0.01)
            self.theory = Model()

        @property
        def flattheory(self):
            return self.theory.y

    return Likelihood()


@pytest.mark.parametrize("key", [
    'dynesty', 'emcee', 'grid', 'nautilus', 'pocomc', 'zeus'])
def test_accuracy(likelihood, key):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    sampler = SAMPLER_CLS[key](likelihood, *ARGS_INIT[key], rng=42,
                               **KWARGS_INIT[key])
    chain = sampler.run(**KWARGS_RUN_FAST[key])

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
    'emcee', 'zeus'])
def test_save_fn(likelihood, key, tmp_path):
    # Check that the sampler correctly saves chains and state, if applicable.

    sampler_1 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT[key], rng=42,
        save_fn=str(tmp_path / 'checkpoint_*.npz'), **KWARGS_INIT[key])
    chain_1 = sampler_1.run(**KWARGS_RUN_FAST[key])
    # The second sampler should not create any new samples if old chains
    # are read correctly.
    sampler_2 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT[key], rng=43,
        save_fn=str(tmp_path / 'checkpoint_*.npz'), **KWARGS_INIT[key])
    chain_2 = sampler_2.run(**KWARGS_RUN_FAST[key])

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
        likelihood, *ARGS_INIT[key], rng=42, **KWARGS_INIT[key])
    chain_1 = sampler_1.run(**KWARGS_RUN_FAST[key])

    sampler_2 = SAMPLER_CLS[key](
        likelihood, *ARGS_INIT[key], rng=42, **KWARGS_INIT[key])
    chain_2 = sampler_2.run(**KWARGS_RUN_FAST[key])

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)
