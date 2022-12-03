import numpy as np


def test_galaxy_clustering():
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, FullPowerSpectrumTemplate
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    theory = KaiserTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(A_s=2e-9, b1=1.).shape
    theory = KaiserTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(A_s=2e-9, b1=1.).shape

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    print(theory.runtime_info.pipeline.params)
    theory(dm=0.01, b1=1.).shape
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    print(theory.runtime_info.pipeline.params)
    theory(dm=0.01, b1=1.).shape

    from desilike.theories.galaxy_clustering import PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles

    theory = PyBirdTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(A_s=2e-9, b1=1.).shape
    theory = PyBirdTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(A_s=2e-9, b1=1.).shape

    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles

    theory = PNGTracerPowerSpectrumMultipoles(method='prim')
    print(theory.runtime_info.pipeline.params)
    params = dict(fnl_loc=100., b1=2.)
    theory2 = PNGTracerPowerSpectrumMultipoles(method='matter')
    assert np.allclose(theory2(**params), theory(**params), rtol=2e-3)
    assert not np.allclose(theory2(fnl_loc=0.), theory(), rtol=2e-3)

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmapar=3.)
    theory = ResummedBAOWigglesTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmas=3.)
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmapar=3.)
    theory = ResummedBAOWigglesTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmas=3.)


def test_observable():
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                       theory=theory)
    observable()
    #observable.wmatrix.plot(show=True)
    theory.template.update(z=1.)
    del theory.template.params['dm']
    observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.template.z == 1.


def test_likelihood():

    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate

    template = BAOPowerSpectrumTemplate(z=1.)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['sigma*', 'al*_-3', 'al*_-2']):
        param.value = 0.
        param.fixed = True
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.08, 0.2]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    likelihood()
    #observable.plot(show=True)
    print(theory.pt.params)
    print(likelihood.varied_params)
    template = BAOPowerSpectrumTemplate(z=0.5, apmode='qiso')
    theory.update(template=template)
    likelihood()
    print(likelihood.varied_params)

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    print(likelihood.runtime_info.pipeline.params)
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(b1=2., dm=0.02))
    theory.template.update(z=1.)
    #del theory.template.params['dm']
    print(likelihood.runtime_info.pipeline.varied_params)
    likelihood()
    #observable.plot(show=False)

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']):
        param.derived = '.best'
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable], scale_covariance=False)
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


def test_install():

    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles

    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']):
        param.derived = '.best'
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                       data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable], scale_covariance=False)
    from desilike import Installer
    Installer()(likelihood)
    from desilike.samplers import EmceeSampler
    Installer(EmceeSampler)


if __name__ == '__main__':

    test_galaxy_clustering()
    test_observable()
    test_likelihood()
    test_cosmo()
    test_install()
