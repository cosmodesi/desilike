import numpy as np
from desilike.likelihoods import GaussianLikelihood


def test_power_spectrum():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)

    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='../../tests/_pk/data.npy',
                                                       mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    likelihood()
    theory()
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='../../tests/_pk/data.npy',
                                                       mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy', shotnoise=1e4,
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    observable()
    #observable.wmatrix.plot(show=True)
    theory.template.update(z=1.)
    observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.template.z == 1.


def test_correlation_function():

    from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerCorrelationFunctionMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    observable = ObservedTracerCorrelationFunctionMultipoles(slim={0: [20., 150.], 2: [20., 150.]}, sstep=5.,
                                                             data='../../tests/_xi/data.npy',
                                                             mocks='../../tests/_xi/mock_*.npy',
                                                             theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    likelihood()
    theory()
    observable = ObservedTracerCorrelationFunctionMultipoles(slim={0: [20., 150.], 2: [20., 150.]}, sstep=5.,
                                                             #data='../../tests/_xi/data.npy',
                                                             mocks='../../tests/_xi/mock_*.npy',
                                                             theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    likelihood()
    theory.power.template.update(z=1.)
    observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.power.template.z == 1.


def test_compression():

    from desilike.observables.galaxy_clustering import BAOCompression, ShapeFitCompression

    observable = BAOCompression(data=[1., 1.], covariance=np.diag([0.01, 0.01]), quantities=['qpar', 'qper'], z=2.)
    likelihood = GaussianLikelihood(observables=[observable])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = ShapeFitCompression(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'm', 'f_sqrt_Ap'], z=2.)
    likelihood = GaussianLikelihood(observables=[observable])
    likelihood()
    print(likelihood.varied_params)


if __name__ == '__main__':

    test_power_spectrum()
    test_correlation_function()
    test_compression()
