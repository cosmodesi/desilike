import numpy as np

from .base import GaussianLikelihood


class TestLikelihood(object):

    def __init__(self):
        self.params = {}
        self.params['a'] = {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}
        self.params['b'] = {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}
        self.x = np.linspace(0., 1., 10)
        mean = np.zeros_like(self.x)
        self.covariance = np.eye(len(self.x))
        self.precision = np.linalg.inv(self.covariance)
        rng = np.random.RandomState(seed=42)
        self.y = rng.multivariate_normal(mean, self.covariance)

    def __call__(self, a=0., b=0.):
        theory = a * self.x + b
        flatdiff = self.y - theory
        return -0.5 * flatdiff.dot(self.precision).dot(flatdiff)
