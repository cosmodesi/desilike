import numpy as np

from desilike import setup_logging


def test_integ():
    from desilike.theories.galaxy_clustering.base import BaseTheoryPowerSpectrumMultipolesFromWedges
    from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles

    topoles = BaseTheoryPowerSpectrumMultipolesFromWedges(mu=8)
    mu, wmu = topoles.mu, topoles.wmu
    assert np.isclose(np.sum(wmu), 1.)
    template = StandardPowerSpectrumTemplate()
    pk_trapz = KaiserTracerPowerSpectrumMultipoles(template=template, mu=100, method='trapz')()
    pk_leggauss = KaiserTracerPowerSpectrumMultipoles(template=template, mu=20, method='leggauss')()
    print(pk_trapz)
    print(pk_leggauss)


def test_bao():

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

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    theory(qpar=1.1, sigmapar=3.)

    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate
    template = BAOPowerSpectrumTemplate(z=0.1, fiducial='DESI', apmode='qiso')
    theory.init.update(template=template)


def test_full_shape():

    def test_emulator(theory):
        print('Emulating', theory)
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        #theory()
        calculator = theory.pt
        bak = theory()
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=0))
        emulator.set_samples()
        emulator.fit()
        pt = emulator.to_calculator()
        theory.init.update(pt=pt)
        assert np.allclose(theory(), bak)

    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    theory = KaiserTracerPowerSpectrumMultipoles()
    theory(logA=3.04, b1=1.).shape
    theory = KaiserTracerCorrelationFunctionMultipoles()
    theory(logA=3.04, b1=1.).shape

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    #test_emulator(theory)
    theory(dm=0.01, b1=1.).shape
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator(theory)
    theory(dm=0.01, b1=1.).shape
    theory = EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator(theory)
    theory(dm=0.01, b1=1.).shape
    theory = LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles

    theory = PyBirdTracerPowerSpectrumMultipoles()
    test_emulator(theory)
    theory(logA=3.04, b1=1.).shape
    theory = PyBirdTracerCorrelationFunctionMultipoles()
    test_emulator(theory)
    theory(logA=3.04, b1=1.).shape


def test_png():

    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles

    theory = PNGTracerPowerSpectrumMultipoles(method='prim')
    params = dict(fnl_loc=100., b1=2.)
    theory2 = PNGTracerPowerSpectrumMultipoles(method='matter')
    assert np.allclose(theory2(**params), theory(**params), rtol=2e-3)
    assert not np.allclose(theory2(fnl_loc=0.), theory(), rtol=2e-3)



if __name__ == '__main__':

    setup_logging()
    test_integ()
    #test_bao()
    test_full_shape()
    #test_png()