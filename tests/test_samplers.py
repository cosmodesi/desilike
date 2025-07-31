import numpy as np
import pytest

from desilike import samplers
from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood


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
    (samplers.DynestySampler, dict(dynamic=True)),
    (samplers.DynestySampler, dict(dynamic=False)),
    (samplers.DynestySampler, dict(save_fn='checkpoint.pkl')),
    (samplers.NautilusSampler, dict(save_fn='checkpoint.h5')),
    (samplers.PocoMCSampler, dict()),
    (samplers.EmceeSampler, dict(nchains=5))])
def test_basic(simple_likelihood, tmp_path, Sampler, kwargs):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.

    if 'save_fn' in kwargs:
        kwargs['save_fn'] = str(tmp_path / kwargs['save_fn'])

    sampler = Sampler(simple_likelihood, rng=42, **kwargs)
    chain = sampler.run()

    assert np.allclose(
        chain.mean(simple_likelihood.varied_params),
        simple_likelihood.flatdata,
        atol=0.03, rtol=0)
    assert np.allclose(
        chain.covariance(simple_likelihood.varied_params),
        np.linalg.inv(simple_likelihood.precision), atol=0.01, rtol=0.1)
