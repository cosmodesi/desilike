import numpy as np

from desilike import setup_logging
from desilike.likelihoods import ObservablesGaussianLikelihood


def test_power_spectrum():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)

    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='../../tests/_pk/data.npy',
                                                       mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy',
                                                       theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    theory()
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='../../tests/_pk/data.npy',
                                                       mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy', shotnoise=1e4,
                                                       theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood.params['pk.loglikelihood'] = {}
    likelihood.params['pk.logprior'] = {}
    #observable()
    #observable.wmatrix.plot(show=True)
    theory.template.update(z=1.)
    #observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.template.z == 1.
    likelihood()
    assert np.allclose((likelihood + likelihood)(), 2. * likelihood() - likelihood.logprior)


def test_correlation_function():

    from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerCorrelationFunctionMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    observable = ObservedTracerCorrelationFunctionMultipoles(slim={0: [20., 150.], 2: [20., 150.]}, sstep=5.,
                                                             data='../../tests/_xi/data.npy',
                                                             mocks='../../tests/_xi/mock_*.npy',
                                                             theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    theory()
    observable = ObservedTracerCorrelationFunctionMultipoles(slim={0: [20., 150.], 2: [20., 150.]}, sstep=5.,
                                                             data={}, #'../../tests/_xi/data.npy',
                                                             mocks='../../tests/_xi/mock_*.npy',
                                                             theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    theory.power.template.update(z=1.)
    observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.power.template.z == 1.


def test_covariance_matrix():
    """
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2], 4: [0.05, 0.2]}, kstep=0.01,
                                                       data={}, #'../../tests/_xi/data.npy',
                                                       theory=theory)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov())
    print(likelihood())
    #observable.plot(show=True)
    observable.plot_covariance_matrix(show=True, corrcoef=True)
    """
    """
    from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerCorrelationFunctionMultipoles, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    observable = ObservedTracerCorrelationFunctionMultipoles(slim={0: [20., 150.], 2: [20., 150.]}, sstep=5.,
                                                             data={}, #'../../tests/_xi/data.npy',
                                                             theory=theory)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov())
    print(likelihood())
    #observable.plot(show=True)
    observable.plot_covariance_matrix(show=True, corrcoef=True)
    """
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles, ObservedTracerCorrelationFunctionMultipoles, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable1 = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                        data={}, #'../../tests/_xi/data.npy',
                                                        theory=theory)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    observable2 = ObservedTracerCorrelationFunctionMultipoles(slim={0: [20., 150.], 2: [20., 150.]}, sstep=5.,
                                                              data={}, #'../../tests/_xi/data.npy',
                                                              theory=theory)
    observables = [observable1, observable2]
    cov = ObservablesCovarianceMatrix(observables, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=cov)
    print(likelihood())
    #observable.plot(show=True)
    likelihood.plot_covariance_matrix(show=True, corrcoef=True)


def test_compression():

    from desilike.observables.galaxy_clustering import BAOCompression, ShapeFitCompression
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    observable = BAOCompression(data=[1., 1.], covariance=np.diag([0.01, 0.01]), quantities=['qpar', 'qper'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = BAOCompression(data=np.array([1.]), covariance=np.diag([0.01]), quantities=['qiso'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=2))
    emulator.set_samples()
    emulator.fit()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = ShapeFitCompression(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'm', 'f_sqrt_Ap'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    print(likelihood.varied_params)

    observable = ShapeFitCompression(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'dm', 'f'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=2))
    emulator.set_samples()
    emulator.fit()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood(A_s=1.5e-9), likelihood(A_s=2.5e-9))
    print(likelihood.varied_params)


if __name__ == '__main__':

    setup_logging()
    #test_power_spectrum()
    #test_correlation_function()
    test_covariance_matrix()
    #test_compression()
