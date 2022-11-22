def test_base():

    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, FullPowerSpectrumTemplate
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles

    theory = KaiserTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(A_s=2e-9, b1=1.).power
    theory = KaiserTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(A_s=2e-9, b1=1.).corr
    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    print(theory.runtime_info.pipeline.params)
    print(theory(dm=0.01, b1=1.).power)
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    print(theory.runtime_info.pipeline.params)
    print(theory(dm=0.01, b1=1.).corr)

    from desilike.theories.galaxy_clustering import PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles

    theory = PyBirdTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    print(theory(A_s=2e-9, b1=1.).power)
    theory = PyBirdTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    print(theory(A_s=2e-9, b1=1.).corr)


def test_likelihood():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(b1=2., dm=0.02))
    #observable.plot(show=False)

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']):
        param.derived = '.best'
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    print(likelihood.runtime_info.pipeline.params.select(solved=True))
    print(likelihood.varied_params)
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(dm=0.02))
    likelihood()
    observable.plot(show=True)


def test_cosmo():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, FullPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=FullPowerSpectrumTemplate(z=1.4, cosmo='external'))
    print(theory.runtime_info.pipeline.get_cosmo_requires())
    print(theory.runtime_info.pipeline.params)


if __name__ == '__main__':

    test_base()
    #test_likelihood()
    #test_cosmo()
