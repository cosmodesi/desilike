import os

import numpy as np

from desilike.likelihoods import BaseLikelihood


class TestSimpleLikelihood(BaseLikelihood):

    params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
              'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

    def initialize(self):
        self.x = np.linspace(0., 1., 10)
        mean = np.zeros_like(self.x)
        self.covariance = 100. * np.eye(len(self.x))
        self.precision = np.linalg.inv(self.covariance)
        rng = np.random.RandomState(seed=42)
        self.y = rng.multivariate_normal(mean, self.covariance)

    def calculate(self, a=0., b=0.):
        theory = a * self.x + b
        flatdiff = self.y - theory
        self.loglikelihood = -0.5 * flatdiff.dot(self.precision).dot(flatdiff)


def TestShapeFitKaiserLikelihood():

    import desilike
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrum
    from desilike.likelihoods import GaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.4))
    dirname = os.path.join(os.path.dirname(desilike.__file__), 'tests', '_pk')
    observable = ObservedTracerPowerSpectrum(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                             data=os.path.join(dirname, 'data.npy'),
                                             mocks=os.path.join(dirname, 'mock_*.npy'),
                                             wmatrix=os.path.join(dirname, 'window.npy'),
                                             theory=theory)
    return GaussianLikelihood(observables=[observable])


def TestFullKaiserLikelihood(cosmo='external'):

    import desilike
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, FullPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrum
    from desilike.likelihoods import GaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=FullPowerSpectrumTemplate(z=1.4, cosmo=cosmo))
    dirname = os.path.join(os.path.dirname(desilike.__file__), 'tests', '_pk')
    observable = ObservedTracerPowerSpectrum(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                             data=os.path.join(dirname, 'data.npy'),
                                             mocks=os.path.join(dirname, 'mock_*.npy'),
                                             wmatrix=os.path.join(dirname, 'window.npy'),
                                             theory=theory)
    return GaussianLikelihood(observables=[observable])
