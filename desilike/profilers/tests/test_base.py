from desilike import setup_logging
from desilike.profilers import MinuitProfiler, ScipyProfiler, BOBYQAProfiler


def test_profilers():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    #for param in theory.params.select(basename=['alpha*', 'sn*']):
    #    param.derived = '.best'
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='../../tests/_pk/data.npy', mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable], scale_covariance=1.)
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
    theory.params['p'].fixed = True  # not fixing p biases fnl_loc posterior
    observable = ObservedTracerPowerSpectrumMultipoles(data='../../tests/_pk/data.npy', mocks='../../tests/_pk/mock_*.npy',
                                                       #klim={0: [0.005, 0.2]},
                                                       klim={0: [0.02, 0.2], 2: [0.02, 0.2]},
                                                       kstep=0.01, theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    profiler = MinuitProfiler(likelihood)
    profiler.maximize(niterations=2)


if __name__ == '__main__':

    setup_logging()
    test_profilers()
