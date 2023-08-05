import numpy as np

from desilike import setup_logging
from desilike.samplers import (EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler,
                               StaticDynestySampler, DynamicDynestySampler, PolychordSampler, GridSampler, QMCSampler, ImportanceSampler)


def test_samplers():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.marg')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    likelihood.params['LRG.loglikelihood'] = likelihood.params['LRG.logprior'] = {}

    for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler][4:5]:
        kwargs = {}
        if Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]:
            kwargs.update(nwalkers=20)
        save_fn = './_tests/chain_*.npy'
        sampler = Sampler(likelihood, save_fn=save_fn, **kwargs)
        chains = sampler.run(max_iterations=20, check=True, check_every=10)
        if sampler.mpicomm.rank == 0:
            assert chains[0].concatenate(chains)._loglikelihood == 'LRG.loglikelihood'
            assert chains[0]['LRG.loglikelihood'].derivs is not None
            assert chains[0].sample_solved()['LRG.loglikelihood'].derivs is None
        chains = sampler.run(max_iterations=20, check=True, check_every=20)
        sampler = Sampler(likelihood, chains=chains, save_fn=save_fn)
        chains = sampler.run(max_iterations=20, check=True, check_every=10)


def test_fixed():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    for Sampler in [GridSampler, QMCSampler]:
        sampler = Sampler(theory)
        sampler.run()


def test_importance():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.marg')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    likelihood()
    sampler = EmceeSampler(likelihood, save_fn='./_tests/chain_*.npy')
    chains = sampler.run(max_iterations=40, check=False)
    sampler = ImportanceSampler(likelihood, chains)
    chains = sampler.run()
    if sampler.mpicomm.rank == 0:
        assert np.all(chains[0].aweight <= 1.)


if __name__ == '__main__':

    setup_logging()
    test_samplers()
    #test_fixed()
    #test_importance()
