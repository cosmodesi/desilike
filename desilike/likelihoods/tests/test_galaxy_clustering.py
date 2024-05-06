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


def test_hartlap():
    from desilike.theories import Cosmoprimo
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    cosmo = Cosmoprimo()
    cosmo.init.params['sigma8_m'] = {'derived': True, 'fixed': False, 'latex': r'\sigma_8'}
    cosmo.init.params['omega_cdm'].update(derived='0.26 * {h}**2')
    template = DirectPowerSpectrumTemplate(cosmo=cosmo)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    size = 30
    ells = (0, 2)
    rng = np.random.RandomState(seed=42)
    covariance = [rng.uniform(0., 1., size * len(ells)) for i in range(10 * size)]

    observable = TracerPowerSpectrumMultipolesObservable(k=np.linspace(0.01, 0.3, size),
                                                         ells=ells,
                                                         data={},
                                                         covariance=covariance,
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable)
    likelihood()
    percival2014_factor = likelihood.percival2014_factor

    cosmo = Cosmoprimo()
    cosmo.init.params['sigma8_m'] = {'derived': True, 'fixed': False, 'latex': r'\sigma_8'}
    template = DirectPowerSpectrumTemplate(cosmo=cosmo)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(k=np.linspace(0.01, 0.3, size),
                                                         ells=ells,
                                                         data={},
                                                         covariance=covariance,
                                                         theory=theory)

    likelihood = ObservablesGaussianLikelihood(observable)
    likelihood.all_params['omega_cdm'].update(derived='0.26 * {h}**2')
    likelihood()
    assert np.allclose(likelihood.percival2014_factor, percival2014_factor)

    likelihood = ObservablesGaussianLikelihood(observable)
    likelihood = likelihood + likelihood
    likelihood.all_params['omega_cdm'].update(derived='0.26 * {h}**2')
    likelihood()
    for like in likelihood.likelihoods:
        assert np.allclose(like.percival2014_factor, percival2014_factor)


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
    #test_precision()
    test_hartlap()
    #test_fisher()
