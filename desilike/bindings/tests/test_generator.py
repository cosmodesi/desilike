import os

import numpy as np

from desilike.likelihoods import BaseLikelihood


class TestSimpleLikelihood(BaseLikelihood):

    _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
               'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

    def initialize(self):
        super(TestSimpleLikelihood, self).initialize()
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
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.4))
    dirname = os.path.join(os.path.dirname(desilike.__file__), 'tests', '_pk')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data=os.path.join(dirname, 'data.npy'),
                                                         covariance=os.path.join(dirname, 'mock_*.npy'),
                                                         wmatrix=os.path.join(dirname, 'window.npy'),
                                                         theory=theory)
    return ObservablesGaussianLikelihood(observables=[observable])


def TestDirectKaiserLikelihood():

    import desilike
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=1.4, cosmo='external'))
    for param in theory.params:
        param.update(namespace='LRG')
    dirname = os.path.join(os.path.dirname(desilike.__file__), 'tests', '_pk')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data=os.path.join(dirname, 'data.npy'),
                                                         covariance=os.path.join(dirname, 'mock_*.npy'),
                                                         wmatrix=os.path.join(dirname, 'window.npy'),
                                                         theory=theory)
    return ObservablesGaussianLikelihood(observables=[observable])


def TestEmulatedDirectKaiserLikelihood():

    import desilike
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=1.4))
    for name in ['b1', 'sn0']:
        theory.init.params[name].update(namespace='ELG')
    dirname = os.path.join(os.path.dirname(desilike.__file__), 'tests', '_pk')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data=os.path.join(dirname, 'data.npy'),
                                                         covariance=os.path.join(dirname, 'mock_*.npy'),
                                                         wmatrix=os.path.join(dirname, 'window.npy'),
                                                         theory=theory)
    observable()
    from desilike.emulators import Emulator, TaylorEmulatorEngine
    emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples()
    emulator.fit()
    theory.init.update(pt=emulator.to_calculator())
    return ObservablesGaussianLikelihood(observables=[observable])


if __name__ == '__main__':

    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator

    Likelihoods = [TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestDirectKaiserLikelihood, TestEmulatedDirectKaiserLikelihood]

    setup_logging('info')
    CobayaLikelihoodGenerator()(Likelihoods, kw_cobaya={})
    CosmoSISLikelihoodGenerator()(Likelihoods)
    MontePythonLikelihoodGenerator()(Likelihoods)
