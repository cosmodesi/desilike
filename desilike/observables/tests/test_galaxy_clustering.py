import numpy as np

from desilike import setup_logging
from desilike.likelihoods import ObservablesGaussianLikelihood


def test_power_spectrum():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TopHatFiberCollisionsPowerSpectrumMultipoles, BoxFootprint, ObservablesCovarianceMatrix

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
                                                         fiber_collisions=fiber_collisions,
                                                         kinlim=(0., 0.24))
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

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    params = {'al0_1': 100., 'al0_-1': 100.}
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data=params,
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)(**params)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=cov)
    print(likelihood(**params))
    observable.plot_wiggles(show=True)

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    params = {'al0_1': 100., 'al0_-1': 100.}
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data=params,
                                                         wmatrix='../../tests/_pk/window.npy',
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)(**params)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=cov)
    print(likelihood(**params))
    observable.plot_wiggles(show=True)


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

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=1.1)
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-4)
    observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [30., 150., 4.], 2: [30., 150., 4.], 4: [30., 150., 4.]},
                                                               data={}, #'../../tests/_xi/data.npy',
                                                               theory=theory)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov())
    print(likelihood())
    #observable.plot(show=True)
    observable.plot_covariance_matrix(show=True, corrcoef=True)

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01], 4: [0.05, 0.2, 0.01]},
                                                         data={},  #'../../tests/_xi/data.npy',
                                                         theory=theory)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov())
    print(likelihood())
    #observable.plot(show=True)
    observable.plot_covariance_matrix(show=True, corrcoef=True)

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable1 = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                          data={},  #'../../tests/_xi/data.npy',
                                                          theory=theory)
    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    observable2 = TracerCorrelationFunctionMultipolesObservable(slim={0: [20., 150., 5.], 2: [20., 150., 5.]},
                                                                data={},  #'../../tests/_xi/data.npy',
                                                                theory=theory)
    observables = [observable1, observable2]
    cov = ObservablesCovarianceMatrix(observables, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=cov)
    print(likelihood())
    #observable.plot(show=True)
    likelihood.plot_covariance_matrix(show=True, corrcoef=True)


def test_compression():

    from desilike.observables.galaxy_clustering import BAOCompressionObservable, ShapeFitCompressionObservable, StandardCompressionObservable
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    observable = BAOCompressionObservable(data=[1., 1.], covariance=np.diag([0.01, 0.01]), quantities=['qpar', 'qper'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = BAOCompressionObservable(data=np.array([1.]), covariance=np.diag([0.01]), quantities=['qiso'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples()
    emulator.fit()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood.varied_params)
    assert np.allclose(likelihood(), 0.)

    observable = ShapeFitCompressionObservable(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'm', 'f_sqrt_Ap'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    print(likelihood.varied_params)

    observable = ShapeFitCompressionObservable(data=[1., 1., 0., 0.8], covariance=np.diag([0.01, 0.01, 0.0001, 0.01]), quantities=['qpar', 'qper', 'dm', 'df'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples()
    emulator.fit()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood(logA=3.), likelihood(logA=3.1))
    print(likelihood.varied_params)

    observable = StandardCompressionObservable(data=[1., 1., 0.8], covariance=np.diag([0.01, 0.01, 0.01]), quantities=['qpar', 'qper', 'df'], z=2.)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    print(likelihood.varied_params)

    from desilike import ParameterCovariance
    covariance = ParameterCovariance(value=np.diag([0.01, 0.01, 0.01]), center=[1., 1., 0.8], params=['qpar', 'qper', 'df'])
    observable = StandardCompressionObservable(data=covariance, covariance=covariance, quantities=['qpar', 'qper', 'df'], z=2.)
    emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples()
    emulator.fit()
    observable = emulator.to_calculator()
    likelihood = ObservablesGaussianLikelihood(observables=[emulator.to_calculator()])
    print(likelihood())
    print(likelihood.varied_params)


def test_integral_cosn():

    from desilike.observables.galaxy_clustering.window import integral_cosn

    for n in np.arange(6):
        limits = (-0.3, 0.8)
        x = np.linspace(*limits, num=1000)
        ref = np.trapz(np.cos(x)**n, x=x)
        test = integral_cosn(n=n, range=limits)
        assert np.abs(test / ref - 1.) < 1e-6


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

    n = 10
    sep = np.linspace(0., Dfc, n)
    kernel = np.linspace(fs, 0., n)
    fiber_collisions = FiberCollisionsPowerSpectrumMultipoles(sep=sep, kernel=kernel, ells=ells)
    fiber_collisions()

    for ill, ell in enumerate(fiber_collisions.ells):
        color = 'C{:d}'.format(ill)
        ax.plot(fiber_collisions.k, fiber_collisions.k * fiber_collisions.power[ill], color=color, linestyle=':', label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    plt.show()

    s = np.linspace(1., 200., 200)
    fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(s=s, fs=fs, Dfc=Dfc, ells=ells)
    fiber_collisions()
    ax = fiber_collisions.plot()
    # ax.get_legend().remove()

    n = 10
    sep = np.linspace(0., Dfc, n)
    kernel = np.linspace(fs, 0., n)
    fiber_collisions = FiberCollisionsCorrelationFunctionMultipoles(s=s, sep=sep, kernel=kernel, ells=ells)
    fiber_collisions()
    for ill, ell in enumerate(fiber_collisions.ells):
        color = 'C{:d}'.format(ill)
        ax.plot(fiber_collisions.s, fiber_collisions.s**2 * fiber_collisions.corr[ill], color=color, linestyle=':')
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
    # test_power_spectrum()
    # test_correlation_function()
    # test_footprint()
    # test_covariance_matrix()
    # test_compression()
    # test_integral_cosn()
    # test_fiber_collisions()
