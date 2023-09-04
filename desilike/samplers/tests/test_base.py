import pytest
import numpy as np

from desilike import PipelineError, setup_logging
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

    for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler][:-1]:
        kwargs = {}
        if Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]:
            kwargs.update(nwalkers=20)
        save_fn = './_tests/chain_*.npy'
        sampler = Sampler(likelihood, save_fn=save_fn, **kwargs)
        chains = sampler.run(max_iterations=20, check=True, check_every=10)
        if sampler.mpicomm.rank == 0:
            assert 'f_sqrt_Ap' in chains[0]
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


def test_error():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = DirectPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.marg')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)

    likelihood.varied_params['Omega_m'].update(prior=dict(), ref=dict(limits=(-10., -9.)))

    sampler = EmceeSampler(likelihood, seed=42, max_tries=10)
    with pytest.raises(ValueError):
        sampler.run(max_iterations=100)

    likelihood.init.update(catch_errors=[])
    likelihood.varied_params['Omega_m'].update(prior=dict(), ref=dict(limits=(-10., -9.)))

    sampler = EmceeSampler(likelihood, seed=42, max_tries=10)
    with pytest.raises(PipelineError):
        sampler.run(max_iterations=100)

    likelihood.init.update(catch_errors=None)
    likelihood.varied_params['Omega_m'].update(prior=dict(), ref=dict(limits=(-0.1, 0.3)))

    sampler = EmceeSampler(likelihood, seed=42, max_tries=100)
    chain = sampler.run(max_iterations=4)[0]
    if sampler.mpicomm.rank == 0:
        assert np.all(np.isfinite(chain.logposterior))




from desilike.base import BaseCalculator


class AffineModel(BaseCalculator):  # all calculators should inherit from BaseCalculator

    # Model parameters; those can also be declared in a yaml file
    _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.5}},
               'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.5}}}

    def initialize(self, x=None):
        # Actual, non-trivial initialization must happen in initialize(); this is to be able to do AffineModel(x=...)
        # without doing any actual work
        self.x = x

    def calculate(self, a=0., b=0.):
        self.y = a * self.x + b  # simple, affine model

    # Not mandatory, this is to return something in particular after calculate (else this will just be the instance)
    def get(self):
        return self.y

    # This is only needed for emulation
    def __getstate__(self):
        return {'x': self.x, 'y': self.y}  # dictionary of Python base types and numpy arrays


from desilike.likelihoods import BaseGaussianLikelihood


class Likelihood(BaseGaussianLikelihood):

    def initialize(self, theory=None):
        # Let us generate some fake data
        self.xdata = np.linspace(0., 1., 10)
        mean = np.zeros_like(self.xdata)
        self.covariance = np.eye(len(self.xdata))
        rng = np.random.RandomState(seed=42)
        y = rng.multivariate_normal(mean, self.covariance)
        super(Likelihood, self).initialize(y, covariance=self.covariance)
        # Requirements
        # AffineModel will be instantied with AffineModel(x=self.xdata)
        if theory is None:
            theory = AffineModel()
        self.theory = theory
        self.theory.init.update(x=self.xdata)  # we set x-coordinates, they will be passed to AffineModel's initialize

    @property
    def flattheory(self):
        # Requirements (theory, requested in __init__) are accessed through .name
        # The pipeline will make sure theory.run(a=..., b=...) has been called
        return self.theory.y  # data - model


def test_nested():

    from desilike.samples import plotting

    likelihood = Likelihood()
    likelihood.varied_params['a'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 0.2})
    likelihood.varied_params['b'].update(fixed=True)

    chains = {}
    #for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler]:
    for Sampler in [EmceeSampler, StaticDynestySampler, DynamicDynestySampler]:
        kwargs, check = {}, True
        mcmc = Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]
        if mcmc:
            kwargs.update(nwalkers=10)
            check = {'max_eigen_gr': 0.02}
        sampler = Sampler(likelihood, save_fn='./_tests/chain_*.npy', **kwargs)
        chain = sampler.run(check=check, min_iterations=1000, check_every=50)[0]
        if mcmc:
            chain = chain.remove_burnin(0.5)[::10]
        chains[Sampler.__name__] = chain

    print(chains)
    plotting.plot_triangle(list(chains.values()), labels=list(chains.keys()), show=True)


if __name__ == '__main__':

    setup_logging()
    test_samplers()
    #test_fixed()
    #test_importance()
    #test_error()
    #test_nested()
