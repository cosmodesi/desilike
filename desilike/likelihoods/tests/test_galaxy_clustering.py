import numpy as np

from desilike import setup_logging


def test_fisher():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint
    from desilike.likelihoods.galaxy_clustering import SNWeightedPowerSpectrumLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    from desilike.utils import Monitor
    
    likelihood = SNWeightedPowerSpectrumLikelihood(theories=theory, footprints=footprint, klim=(0.01, 0.5))
    likelihood()
    from desilike import Fisher
    fisher = Fisher(likelihood)
    fisher()


if __name__ == '__main__':

    setup_logging()
    test_fisher()
