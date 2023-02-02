import numpy as np

from desilike import setup_logging
from desilike.likelihoods import ObservablesGaussianLikelihood


def test_power_spectrum():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TopHatFiberCollisionsPowerSpectrumMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)

    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='../../tests/_pk/data.npy',
                                                         covariance='../../tests/_pk/mock_*.npy',
                                                         wmatrix='../../tests/_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    theory()

    fiber_collisions = TopHatFiberCollisionsPowerSpectrumMultipoles(fs=0.5, Dfc=1.)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='../../tests/_pk/data.npy',
                                                         covariance='../../tests/_pk/mock_*.npy',
                                                         wmatrix='../../tests/_pk/window.npy',
                                                         shotnoise=1e4,
                                                         theory=theory,
                                                         fiber_collisions=fiber_collisions)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood.params['pk.loglikelihood'] = {}
    likelihood.params['pk.logprior'] = {}
    likelihood()
    observable.plot(show=True)
    #observable()
    #observable.wmatrix.plot(show=True)
    theory.template.init.update(z=1.)
    #observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.template.z == 1.
    likelihood()
    assert np.allclose((likelihood + likelihood)(), 2. * likelihood() - likelihood.logprior)


def test_correlation_function():

    from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable, TopHatFiberCollisionsCorrelationFunctionMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [20., 150., 5.], 2: [20., 150., 5.]},
                                                               data='../../tests/_xi/data.npy',
                                                               covariance='../../tests/_xi/mock_*.npy',
                                                               theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    theory()
    fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(fs=0.5, Dfc=1.)
    observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [20., 150., 5.], 2: [20., 150., 5.]},
                                                               data={}, #'../../tests/_xi/data.npy',
                                                               covariance='../../tests/_xi/mock_*.npy',
                                                               theory=theory,
                                                               fiber_collisions=fiber_collisions)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    theory.power.template.init.update(z=1.)
    observable()
    observable.plot(show=True)
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.power.template.z == 1.


def test_footprint():
    from desilike.observables.galaxy_clustering import BoxFootprint, CutskyFootprint
    from cosmoprimo.fiducial import DESI
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    footprint = CutskyFootprint(nbar=2500., area=14000., zrange=(0.8, 1.6), cosmo=DESI())
    print(footprint.zavg, footprint.size / 1e6, footprint.shotnoise, footprint.volume / 1e9)


def test_covariance_matrix():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01], 4: [0.05, 0.2, 0.01]},
                                                         data={}, #'../../tests/_xi/data.npy',
                                                         theory=theory)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov())
    print(likelihood())
    #observable.plot(show=True)
    observable.plot_covariance_matrix(show=True, corrcoef=True)

    from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [20., 150., 5.], 2: [20., 150., 5.]},
                                                               data={}, #'../../tests/_xi/data.npy',
                                                               theory=theory)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov())
    print(likelihood())
    #observable.plot(show=True)
    observable.plot_covariance_matrix(show=True, corrcoef=True)

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable1 = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                          data={}, #'../../tests/_xi/data.npy',
                                                          theory=theory)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    observable2 = TracerCorrelationFunctionMultipolesObservable(slim={0: [20., 150., 5.], 2: [20., 150., 5.]},
                                                                data={}, #'../../tests/_xi/data.npy',
                                                                theory=theory)
    observables = [observable1, observable2]
    cov = ObservablesCovarianceMatrix(observables, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=cov)
    print(likelihood())
    #observable.plot(show=True)
    likelihood.plot_covariance_matrix(show=True, corrcoef=True)


def test_compression():

    from desilike.observables.galaxy_clustering import BAOCompressionObservable, ShapeFitCompressionObservable
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    observable = BAOCompressionObservable(data=[1., 1.], covariance=np.diag([0.01, 0.01]), quantities=['qpar', 'qper'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = BAOCompressionObservable(data=np.array([1.]), covariance=np.diag([0.01]), quantities=['qiso'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=2))
    emulator.set_samples()
    emulator.fit()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = ShapeFitCompressionObservable(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'm', 'f_sqrt_Ap'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    print(likelihood.varied_params)

    observable = ShapeFitCompressionObservable(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'dm', 'f'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=2))
    emulator.set_samples()
    emulator.fit()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood(A_s=1.5e-9), likelihood(A_s=2.5e-9))
    print(likelihood.varied_params)


def test_fiber_collisions():

    from matplotlib import pyplot as plt
    from desilike.observables.galaxy_clustering import (FiberCollisionsPowerSpectrumMultipoles, FiberCollisionsCorrelationFunctionMultipoles,
                                                        TopHatFiberCollisionsPowerSpectrumMultipoles, TopHatFiberCollisionsCorrelationFunctionMultipoles)
    from desilike.observables.galaxy_clustering import WindowedCorrelationFunctionMultipoles, WindowedPowerSpectrumMultipoles

    fs, Dfc = 0.5, 3.
    ells = (0, 2, 4)
    fiber_collisions = TopHatFiberCollisionsPowerSpectrumMultipoles(fs=fs, Dfc=Dfc, ells=ells)
    fiber_collisions()
    ax = fiber_collisions.plot()

    #kernel = [fs, fs / 2., fs / 4.]
    #sep = [0., Dfc / 2., Dfc]
    #kernel = [fs] * 2
    #sep = [0., Dfc]
    n = 100
    kernel = np.linspace(0.5, 0.5, n)
    sep = np.linspace(0., 3., n)
    fiber_collisions = FiberCollisionsPowerSpectrumMultipoles(sep=sep, kernel=kernel, ells=ells)
    fiber_collisions()

    for ill, ell in enumerate(fiber_collisions.ells):
        color = 'C{:d}'.format(ill)
        ax.plot(fiber_collisions.k, fiber_collisions.k * fiber_collisions.power[ill], color=color, linestyle=':', label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    plt.show()
    exit()

    s = np.linspace(1., 200., 200)
    fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(s=s, fs=fs, Dfc=Dfc, ells=ells)
    fiber_collisions()
    ax = fiber_collisions.plot()
    # ax.get_legend().remove()

    kernel = fs
    sep = [0., Dfc]
    fiber_collisions = FiberCollisionsCorrelationFunctionMultipoles(s=s, sep=sep, kernel=kernel, ells=ells)
    fiber_collisions()
    for ill, ell in enumerate(fiber_collisions.ells):
        color = 'C{:d}'.format(ill)
        ax.plot(fiber_collisions.s, fiber_collisions.s**2 * fiber_collisions.corr[ill], color=color, linestyle=':', label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    plt.show()

    fs, Dfc = 0.5, 3.
    ells = (0, 2, 4)
    fiber_collisions = TopHatFiberCollisionsPowerSpectrumMultipoles(fs=fs, Dfc=Dfc, ells=ells)
    window = WindowedPowerSpectrumMultipoles(k=np.linspace(0.01, 0.2, 50), fiber_collisions=fiber_collisions)
    window()
    window.plot(show=True)

    fs, Dfc = 0.5, 3.
    ells = (0, 2, 4)
    fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(fs=fs, Dfc=Dfc, ells=ells)
    window = WindowedCorrelationFunctionMultipoles(s=np.linspace(20, 150, 50), fiber_collisions=fiber_collisions)
    window()
    window.plot(show=True)


if __name__ == '__main__':

    setup_logging()
    #test_power_spectrum()
    #test_correlation_function()
    # test_footprint()
    # test_covariance_matrix()
    # test_compression()
    test_fiber_collisions()
