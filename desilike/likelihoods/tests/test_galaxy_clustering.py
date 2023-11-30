import numpy as np

from desilike import setup_logging


def test_precision():
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    theory = KaiserTracerPowerSpectrumMultipoles()
    observable = TracerPowerSpectrumMultipolesObservable(k=np.linspace(0.01, 0.3, 30),
                                                         ells=(0, 2),
                                                         data={},
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable, precision=np.eye(60))
    assert np.allclose(likelihood(), 0.)


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
    test_precision()
    test_fisher()
