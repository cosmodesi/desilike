import numpy as np
import pytest

import desilike.samplers as samplers
from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood


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


@pytest.mark.parametrize("Sampler, kwargs_init, kwargs_run", [
    (samplers.DynestySampler, dict(dynamic=False, nlive=100), dict()),
    (samplers.DynestySampler, dict(dynamic=True, nlive=100),
     dict(n_effective=0)),
    (samplers.EmceeSampler, dict(), dict()),
    (samplers.GridSampler, dict(), dict(size=100)),
    (samplers.NautilusSampler, dict(n_networks=1, n_live=500), dict(n_eff=0)),
    (samplers.PocoMCSampler, dict(n_effective=200, n_active=100),
     dict(n_total=100, n_evidence=100)),
    (samplers.ZeusSampler, dict(), dict())])
def test_accuracy(likelihood, tmp_path, Sampler, kwargs_init, kwargs_run):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    if issubclass(Sampler, samplers.base.MarkovChainSampler):
        sampler = Sampler(likelihood, 4, rng=42, **kwargs_init)
    else:
        sampler = Sampler(likelihood, rng=42, **kwargs_init)
    chain = sampler.run(**kwargs_run)

    if isinstance(sampler, samplers.GridSampler):
        chain.aweight = np.exp(chain.logposterior)

    # The mean should match.
    assert np.allclose(chain.mean(likelihood.varied_params),
                       likelihood.flatdata, atol=0.03, rtol=0)
    # The covariance should match.
    assert np.allclose(chain.covariance(likelihood.varied_params),
                       np.linalg.inv(likelihood.precision), atol=0.01,
                       rtol=0.1)


@pytest.mark.parametrize("Sampler, kwargs_init, kwargs_run", [
    (samplers.EmceeSampler, dict(), dict(max_iterations=100))])
def test_save_fn(likelihood, tmp_path, Sampler, kwargs_init, kwargs_run):
    # Check that the sampler correctly saves chains and state, if applicable.

    kwargs_init['save_fn'] = str(tmp_path / 'checkpoint_*.npz')

    if issubclass(Sampler, samplers.base.MarkovChainSampler):
        args_init = (likelihood, 4)
    else:
        args_init = (likelihood, )

    sampler_1 = Sampler(*args_init, rng=42, **kwargs_init)
    chain_1 = sampler_1.run(**kwargs_run)
    # The second sampler should not create any new samples if old chains
    # are read correctly.
    sampler_2 = Sampler(*args_init, rng=43, **kwargs_init)
    chain_2 = sampler_2.run(**kwargs_run)

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)


@pytest.mark.parametrize("Sampler, kwargs_init, kwargs_run", [
    (samplers.EmceeSampler, dict(), dict(max_iterations=100))])
def test_rng(likelihood, tmp_path, Sampler, kwargs_init, kwargs_run):
    # Test that specifying the random seed leads to reproducible results.

    if issubclass(Sampler, samplers.base.MarkovChainSampler):
        args_init = (likelihood, 4)
    else:
        args_init = (likelihood, )

    sampler_1 = Sampler(*args_init, rng=42, **kwargs_init)
    sampler_2 = Sampler(*args_init, rng=42, **kwargs_init)
    chain_1 = sampler_1.run(**kwargs_run)
    chain_2 = sampler_2.run(**kwargs_run)

    assert len(chain_1) == len(chain_2)
    assert np.allclose(chain_1.logposterior.value,
                       chain_2.logposterior.value, atol=1e-6)
