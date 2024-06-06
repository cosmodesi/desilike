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


def test_observable_covariance():

    from cosmoprimo.fiducial import DESI
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable
    from desilike.observables import ObservableArray, ObservableCovariance
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5, fiducial=DESI())
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)

    edges = np.linspace(0., 0.4, 81)
    data1 = ObservableArray(edges=[edges] * 3, value=[edges[:-1]] * 3, projs=[0, 2, 4])
    observable1 = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.1, 0.02], 2: [0.05, 0.1, 0.01]},
                                                         data=data1,
                                                         covariance=ObservableCovariance(np.eye(data1.flatx.size), observables=[data1]),
                                                         #covariance=np.eye(8, dtype='f8'),
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable1])
    likelihood()
    print(likelihood.covariance.shape)

    theory = KaiserTracerCorrelationFunctionMultipoles(template=template)
    edges = np.linspace(0., 200, 201)
    data2 = ObservableArray(edges=[edges] * 3, value=[edges[:-1]] * 3, projs=[0, 2, 4])
    observable2 = TracerCorrelationFunctionMultipolesObservable(slim={0: [20, 150, 4], 2: [30, 150, 5]},
                                                               data=data2,
                                                               theory=theory)

    covariance = ObservableCovariance(np.eye(data1.flatx.size + data2.flatx.size), observables=[data1, data2])
    likelihood = ObservablesGaussianLikelihood(observables=[observable1, observable2], covariance=covariance, scale_covariance=1 / 5.)
    likelihood()


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
    #test_hartlap()
    #test_fisher()
    test_observable_covariance()
