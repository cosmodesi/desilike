import numpy as np
import pytest

import desilike.samplers as samplers
from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood


@pytest.fixture
def simple_likelihood():

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
def test_basic(simple_likelihood, tmp_path, Sampler, kwargs_init, kwargs_run):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    if 'save_fn' in kwargs_init:
        kwargs_init['save_fn'] = str(tmp_path / kwargs_init['save_fn'])

    if issubclass(Sampler, samplers.base.MarkovChainSampler):
        sampler = Sampler(simple_likelihood, 4, rng=42, **kwargs_init)
    else:
        sampler = Sampler(simple_likelihood, rng=42, **kwargs_init)
    chain = sampler.run(**kwargs_run)

    if isinstance(sampler, samplers.GridSampler):
        chain.aweight = np.exp(chain.logposterior)

    assert np.allclose(
        chain.mean(simple_likelihood.varied_params),
        simple_likelihood.flatdata,
        atol=0.03, rtol=0)
    assert np.allclose(
        chain.covariance(simple_likelihood.varied_params),
        np.linalg.inv(simple_likelihood.precision), atol=0.01, rtol=0.1)
