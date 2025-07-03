import numpy as np
import pytest

from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood
from desilike.samplers import DynestySampler, NautilusSampler


@pytest.fixture
def simple_likelihood():

    class Model(BaseCalculator):

        _params = dict(a=dict(prior=dict(dist='uniform', limits=[0, 1])),
                       b=dict(prior=dict(dist='uniform', limits=[0, 1])))

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


@pytest.mark.parametrize("Sampler, kwargs", [
    (DynestySampler, dict(dynamic=True)),
    (DynestySampler, dict(dynamic=False)),
    (DynestySampler, dict(save_fn='checkpoint.pkl')),
    (NautilusSampler, dict(save_fn='checkpoint.h5'))])
def test_basic(simple_likelihood, Sampler, kwargs):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    sampler = Sampler(simple_likelihood, rng=42, **kwargs)
    chain = sampler.run()

    assert np.allclose(
        chain.mean(simple_likelihood.varied_params),
        simple_likelihood.flatdata,
        atol=0.03, rtol=0)
    assert np.allclose(
        chain.covariance(simple_likelihood.varied_params),
        np.linalg.inv(simple_likelihood.precision), atol=0.01, rtol=0.1)
