from desilike import setup_logging
from desilike.profilers import MinuitProfiler, ScipyProfiler, BOBYQAProfiler


def test_profilers():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='../../tests/_pk/data.npy', covariance='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=1.)

    profiler = MinuitProfiler(likelihood)
    profiler.maximize(niterations=2)
    print(print(profiler.profiles.to_stats()))

    """
    # for param in likelihood.varied_params:
    #     print(param, [likelihood(**{param.name: param.value + param.proposal * scale}) for scale in [-1., 1.]])
    for Profiler in [MinuitProfiler, ScipyProfiler, BOBYQAProfiler]:
        profiler = Profiler(likelihood)
        profiler.maximize(niterations=2)
        print(print(profiler.profiles.to_stats()))
        #likelihood()
        #observable.plot(show=True)
    """

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    template = FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI')
    # Here we choose b-p parameterization
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode='b-p')
    theory.params['p'].update(fixed=True)  # not fixing p biases fnl_loc posterior
    observable = TracerPowerSpectrumMultipolesObservable(data='../../tests/_pk/data.npy', covariance='../../tests/_pk/mock_*.npy',
                                                         klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]}, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    profiler = MinuitProfiler(likelihood)
    profiler.maximize(niterations=2)


def test_solve():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    #theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    #for param in theory.params.select(basename=['df', 'dm', 'qpar', 'qper']): param.update(fixed=True)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={'sn0': 1000.},
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    for param in likelihood.all_params.select(basename=['df', 'dm', 'qpar', 'qper']): param.update(fixed=True)

    #import numpy as np
    #likelihood.flatdata += 100 * np.cos(np.linspace(0., 5. * np.pi, observable.flatdata.size))
    profiler = MinuitProfiler(likelihood)
    #profiler = ScipyProfiler(likelihood, method='lsq')
    profiles = profiler.maximize(niterations=2)
    print(profiles.to_stats())
    assert profiles.bestfit.logposterior.param.derived

    #print(likelihood(**profiles.bestfit.choice(varied=True)))
    #from desilike.samples import plotting
    #plotting.plot_triangle(profiles, show=True)


if __name__ == '__main__':

    setup_logging()
    #test_profilers()
    test_solve()
