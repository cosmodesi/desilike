def test_base():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    theory = KaiserTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(sigma8=0.9, b1=1.).power
    theory.template(b1=1.).pk_tt
    print(theory.template.runtime_info.pipeline.params)
    theory.template = ShapeFitPowerSpectrumTemplate(k=theory.kin)
    print(theory.runtime_info.pipeline.params)
    theory(dm=0.01, qpar=0.99)


def test_likelihood():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrum
    from desilike.likelihoods import GaussianLikelihood

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.4))
    observable = ObservedTracerPowerSpectrum(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                             data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                             theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    print(likelihood(dm=0.), likelihood(dm=0.01))


def test_cosmo():
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, FullPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=FullPowerSpectrumTemplate(z=1.4, cosmo='external'))
    print(theory.runtime_info.pipeline.get_cosmo_requires())
    print(theory.runtime_info.pipeline.params)


if __name__ == '__main__':

    #test_base()
    #test_likelihood()
    test_cosmo()
