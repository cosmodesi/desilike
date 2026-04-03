import numpy as np
import pytest
from jax import numpy as jnp

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.profilers import Profiler


@pytest.fixture
def likelihood():

    class Likelihood(BaseGaussianLikelihood):

        def calculate(self, **kwargs):
            self.flattheory = jnp.array([kwargs[name] for name in ['a', 'b']])
            super().calculate()

    likelihood = Likelihood(np.array([0.4, 0.6]), covariance=np.eye(2) * 0.01)
    likelihood.init.params = dict(
        a=dict(prior=dict(dist='norm', limits=[0, 1], loc=0.4, scale=0.1)),
        b=dict(prior=dict(dist='uniform', limits=[0, 1])))

    return likelihood


def test_accuracy(likelihood):
    # Test that the profiler returns the correct result.

    profiler = Profiler(likelihood, rng=42, posterior=False)
    profiler.add_grid(dict(a=np.linspace(0, 1, 31)))
    samples = profiler.run()

    assert np.allclose(samples['log_likelihood'],
                       -(samples['a'] - 0.4)**2 * 100, rtol=0, atol=1e-3)
