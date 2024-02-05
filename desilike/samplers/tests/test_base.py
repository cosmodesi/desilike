import os
import time

import pytest
import numpy as np

from desilike import PipelineError, setup_logging
from desilike.samplers import (EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler,
                               StaticDynestySampler, DynamicDynestySampler, PolychordSampler, NautilusSampler,
                               NUTSSampler, MCLMCSampler, GridSampler, QMCSampler, ImportanceSampler)


def test_samplers():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = BAOPowerSpectrumTemplate(z=0.5)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['al*']): param.update(derived='.marg')
    for param in theory.params.select(basename=['al0_*']): param.update(derived='.marg_not_derived')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov, name='LRG')

    for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, NautilusSampler, PolychordSampler][:1]:
        kwargs = {}
        if Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]:
            kwargs.update(nwalkers=20)
        if Sampler in [StaticDynestySampler, DynamicDynestySampler, PolychordSampler, NautilusSampler]:
            kwargs.update(nlive=100)
        save_fn = ['./_tests/chain_{:d}.npz'.format(i) for i in range(min(likelihood.mpicomm.size, 1))]
        sampler = Sampler(likelihood, save_fn=save_fn, **kwargs)
        kwargs = {}
        if Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]:
            kwargs.update(thin_by=2)
        chains = sampler.run(max_iterations=100, check=True, check_every=50, **kwargs)
        if sampler.mpicomm.rank == 0:
            assert chains[0]['LRG.loglikelihood'].param.latex() == 'L_{\mathrm{LRG}}'
            assert chains[0]['LRG.loglikelihood'].param.derived
            assert chains[0].logposterior.param.latex() == '\mathcal{L}'
            assert chains[0].logposterior.param.derived
            chains[0]['LRG.loglikelihood']['al2_0', 'al2_0']
            with pytest.raises(KeyError):
                chains[0]['LRG.loglikelihood']['al0_0', 'al0_0']
            chains[0].sample_solved()
            assert np.allclose(chains[0].logposterior, chains[0]['LRG.loglikelihood'][()] + chains[0]['LRG.logprior'][()])
        size1 = sampler.mpicomm.bcast(chains[0].size if sampler.mpicomm.rank == 0 else None, root=0)
        chains = sampler.run(max_iterations=0, check=True, check_every=10)
        size2 = sampler.mpicomm.bcast(chains[0].size if sampler.mpicomm.rank == 0 else None, root=0)
        assert size2 == size1
        if sampler.mpicomm.rank == 0:
            assert 'DV_over_rd' in chains[0]
            assert chains[0].concatenate(chains)._loglikelihood == 'LRG.loglikelihood'
            assert chains[0]['LRG.loglikelihood'].derivs is not None
            assert chains[0].sample_solved()['LRG.loglikelihood'].derivs is None
        chains = sampler.run(max_iterations=20, check=True, check_every=20)
        sampler = Sampler(likelihood, chains=save_fn, save_fn=save_fn)
        chains = sampler.run(max_iterations=20, check=True, check_every=10)


def test_nautilus(test=2):

    if test == 1:
        from scipy.stats import norm
        from nautilus import Prior

        prior = Prior()
        prior.add_parameter('a', dist=(-5, +5))
        prior.add_parameter('b', dist=(-5, +5))
        prior.add_parameter('c', dist=norm(loc=0, scale=2.0))

        import numpy as np
        from scipy.stats import multivariate_normal

        def likelihood(param_dict):
            x = np.array([param_dict['a'], param_dict['b'], param_dict['c']])
            return multivariate_normal.logpdf(
                x, mean=np.zeros(3), cov=[[1, 0, 0.90], [0, 1, 0], [0.90, 0, 1]])


        from nautilus import Sampler

        sampler = Sampler(prior, likelihood, n_live=1000)
        sampler.run(verbose=True)

    if test == 2:

        import numpy as np
        from scipy.stats import uniform, multivariate_normal

        def prior(values):
            toret = uniform(-5., 5.).ppf(values)
            print(values, toret)
            return toret

        def likelihood(values):
            return multivariate_normal.logpdf(values, mean=np.zeros(3), cov=[[1, 0, 0.90], [0, 1, 0], [0.90, 0, 1]])

        from nautilus import Sampler

        sampler = Sampler(prior, likelihood, n_dim=3, n_live=1000)
        sampler.run(verbose=True)


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
        c = a + b
        d = a - b
        self.y = c * self.x + d  # simple, affine model

    # Not mandatory, this is to return something in particular after calculate (else this will just be the instance)
    def get(self):
        return self.y

    # This is only needed for emulation
    def __getstate__(self):
        return {'x': self.x, 'y': self.y}  # dictionary of Python base types and numpy arrays


class PolyModel(BaseCalculator):  # all calculators should inherit from BaseCalculator

    # Model parameters; those can also be declared in a yaml file
    _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}},
               'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}},
               'c': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}}}

    def initialize(self, x=None):
        # Actual, non-trivial initialization must happen in initialize(); this is to be able to do PolyModel(x=...)
        # without doing any actual work
        self.x = x

    def calculate(self, a=0., b=0., c=0.):
        self.y = a * self.x**2 + b * c * self.x + c

    # Not mandatory, this is to return something in particular after calculate (else this will just be the instance)
    def get(self):
        return self.y

    # This is only needed for emulation
    def __getstate__(self):
        return {'x': self.x, 'y': self.y}  # dictionary of Python base types and numpy arrays


from desilike.likelihoods import BaseGaussianLikelihood


class Likelihood(BaseGaussianLikelihood):

    def initialize(self, theory=None, scalecov=1.):
        # Let us generate some fake data
        self.xdata = np.linspace(0., 2., 10)
        mean = self.xdata**2 + self.xdata
        self.covariance = np.eye(len(self.xdata)) * scalecov
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


class MyGaussianLikelihood(BaseGaussianLikelihood):

    _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}},
               'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}}}

    def initialize(self):
        self.covariance = np.eye(len(self.params))
        y = np.zeros(len(self.params))
        super(MyGaussianLikelihood, self).initialize(y, covariance=self.covariance)

    def calculate(self, a=0., b=0.):
        self.flattheory = np.array([a, b])
        super(MyGaussianLikelihood, self).calculate()


def test_mcmc():

    from desilike.samples import plotting

    if 0:
        from desilike import LikelihoodFisher
        params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}},
                'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 1.}}}
        fisher = LikelihoodFisher(center=[0., 0.], params=params, hessian=-np.eye(2))
    else:
        likelihood = Likelihood(theory=PolyModel())
        from desilike import Fisher
        fisher = Fisher(likelihood, method='auto')
        fisher = fisher()

    likelihood = fisher.to_likelihood()  # MyGaussianLikelihood()
    #likelihood.varied_params['a'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 0.2})
    #likelihood.varied_params['b'].update(fixed=True)
    #for param in likelihood.varied_params:
    #    param.update(prior=None)

    chains, timing = {}, {}
    for Sampler in [EmceeSampler, NUTSSampler, MCLMCSampler, MCMCSampler][:3]:
        kwargs, check = {}, True
        ensemble = Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]
        if ensemble:
            kwargs.update(nwalkers=10)
            check = {'max_eigen_gr': 0.01, 'stable_over': 3}
        if Sampler is NUTSSampler:
            likelihood.varied_params['a'].update(derived='.best')
            check = {'max_eigen_gr': 0.01, 'stable_over': 3}
        save_fn = './_tests/chain_{}_0.npy'.format(Sampler.__name__.replace('Sampler', '').lower())
        nchains = save_fn if os.path.isfile(save_fn) else 1
        nchains = 1
        #os.remove(save_fn)
        sampler = Sampler(likelihood, chains=nchains, save_fn=save_fn, **kwargs)
        t0 = time.time()
        chain = sampler.run(check=check, min_iterations=100)[0]
        timing[Sampler.__name__] = time.time() - t0
        chain = chain.remove_burnin(0.5)
        if ensemble: chain = chain[::10]
        chains[Sampler.__name__] = chain

    print(timing)
    plotting.plot_triangle(list(chains.values()) + [fisher], labels=list(chains.keys()) + ['fisher'], show=True)


def test_nested():

    from desilike.samples import plotting

    likelihood = Likelihood()
    likelihood.varied_params['a'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 0.2})
    likelihood.varied_params['b'].update(fixed=True)

    chains = {}
    #for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler]:
    for Sampler in [EmceeSampler, StaticDynestySampler, DynamicDynestySampler]:
        kwargs, check = {}, True
        ensemble = Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]
        if ensemble:
            kwargs.update(nwalkers=10)
            check = {'max_eigen_gr': 0.02}
        sampler = Sampler(likelihood, save_fn='./_tests/chain_*.npy', **kwargs)
        chain = sampler.run(check=check, min_iterations=1000, check_every=50)[0]
        if ensemble:
            chain = chain.remove_burnin(0.5)[::10]
        chains[Sampler.__name__] = chain

    print(chains)
    plotting.plot_triangle(list(chains.values()), labels=list(chains.keys()), show=True)


def test_hmc():

    likelihood = Likelihood()
    #likelihood.varied_params['a'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 0.2, 'limits': [-0.5, 0.5]})
    #likelihood.all_params['a'].update(derived='.marg')
    likelihood()

    from desilike.samples import plotting
    for Sampler in [NUTSSampler, MCLMCSampler][1:]:
        sampler = Sampler(likelihood, save_fn='./_tests/chain_*.npy')
        chain = sampler.run(min_iterations=100, max_iterations=2000, check_every=200)[0]
        plotting.plot_triangle(chain.remove_burnin(0.2), show=True)


def test_marg():

    from desilike.samples import Chain, plotting

    likelihood = Likelihood(theory=PolyModel(), scalecov=0.1)
    sampler_kwargs = {'nwalkers': 20, 'seed': 42}
    run_kwargs = {'check': {'max_eigen_gr': 0.02}, 'min_iterations': 100, 'check_every': 100}
    save_fn_full = './_tests/chain_full_0.npy'
    save_fn_marg = './_tests/chain_marg_0.npy'
    save_fn_bf = './_tests/chain_bf_0.npy'
    todo = ['full', 'marg', 'bf'][:1]
    if 'full' in todo:
        sampler = ZeusSampler(likelihood, chains=save_fn_full if os.path.isfile(save_fn_full) else 1, save_fn=save_fn_full, **sampler_kwargs)
        sampler.run(**run_kwargs)[0]
    if 'marg' in todo:
        for param in likelihood.all_params.select(basename=['a', 'b']):
            param.update(derived='.auto')
        sampler = ZeusSampler(likelihood, chains=save_fn_marg if os.path.isfile(save_fn_marg) else 1, save_fn=save_fn_marg, **sampler_kwargs)
        sampler.run(**run_kwargs)[0]
    if 'bf' in todo:
        for param in likelihood.all_params.select(basename=['a', 'b']):
            param.update(derived='.best')
        sampler = ZeusSampler(likelihood, chains=save_fn_bf if os.path.isfile(save_fn_bf) else 1, save_fn=save_fn_bf, **sampler_kwargs)
        sampler.run(**run_kwargs)[0]
    chain_full = Chain.load(save_fn_full).remove_burnin(0.5)[::5].sample_solved(size=5)
    chain_marg = Chain.load(save_fn_marg).remove_burnin(0.5)[::5].sample_solved(size=5)
    chain_bf = Chain.load(save_fn_bf).remove_burnin(0.5)[::5].sample_solved(size=5)
    #plotting.plot_triangle([chain_bf], labels=['ref', 'marg', 'bestfit'], show=True, fn='./_tests/fig.png')
    plotting.plot_triangle([chain_full, chain_marg, chain_bf], labels=['ref', 'marg', 'bestfit'], show=True, fn='./_tests/fig.png')


def test_bao_hmc():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.samples import plotting


    template = BAOPowerSpectrumTemplate(z=0.5)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['al*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.3, 0.005], 2: [0.05, 0.3, 0.005]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov, name='LRG')

    for param in likelihood.varied_params.select(basename=['q*', 'dbeta']):
        param.update(prior={'dist': 'norm', 'loc': 1., 'scale': 1.})

    for param in likelihood.varied_params.select(basename='sigma*'):
        param.update(prior={'dist': 'norm', 'loc': param.prior.center(), 'scale': 2.})

    chains = {}
    for Sampler in [NUTSSampler, MCLMCSampler][:1]:
        save_fn = ['./_tests/chain_{:d}.npz'.format(i) for i in range(min(likelihood.mpicomm.size, 1))]
        kwargs = {}
        if Sampler is MCLMCSampler:
            #kwargs['adaptation'] = {'niterations': 1000, 'num_effective_samples': 200}
            kwargs['adaptation'] = False
            kwargs['L'] = 1.
        sampler = Sampler(likelihood, seed=12, save_fn=save_fn, **kwargs)
        chains[Sampler.__name__] = sampler.run(max_iterations=10000, check={'max_eigen_gr': 1., 'min_ess': 50}, check_every=200)[0]

    if likelihood.mpicomm.rank == 0:
        plotting.plot_triangle(list(chains.values()), labels=list(chains.keys()), show=True)


def test_folpsax_hmc():

    import time
    import jax
    from jax import numpy as jnp
    from desilike.theories.galaxy_clustering import FOLPSAXTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.samples import plotting

    def get_theory(z):
        template = ShapeFitPowerSpectrumTemplate(z=z)
        theory = FOLPSAXTracerPowerSpectrumMultipoles(template=template)
        for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
        return theory

    def get_theory(z):
        template = BAOPowerSpectrumTemplate(z=z)
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
        for param in theory.params.select(basename=['al*']): param.update(derived='.best')
        return theory

    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.3, 0.005], 2: [0.05, 0.3, 0.005]},
                                                         data={},
                                                         theory=get_theory(z=0.5))
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=1)()

    observable2 = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.3, 0.005], 2: [0.05, 0.3, 0.005]},
                                                         data={},
                                                         theory=get_theory(z=1.))

    #likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov, name='LRG')
    #likelihood()
    cov = np.eye(cov.shape[0] * 2)
    likelihood = ObservablesGaussianLikelihood(observables=[observable, observable2], covariance=cov, name='LRG')
    likelihood()
    t0 = time.time()
    start = jnp.array([param.prior.sample() if param.prior.is_proper() else param.value for param in likelihood.varied_params])

    def fn(values):
        return likelihood(dict(zip(likelihood.varied_params.names(), values)))
    fn(start - 1e-6)

    grad = jax.value_and_grad(fn)
    grad = jax.jit(grad)
    grad(start)
    print(time.time() - t0)
    exit()

    chains = {}
    for Sampler in [NUTSSampler, MCLMCSampler][:1]:
        save_fn = ['./_tests/chain_{:d}.npz'.format(i) for i in range(min(likelihood.mpicomm.size, 1))]
        kwargs = {}
        if Sampler is MCLMCSampler:
            #kwargs['adaptation'] = {'niterations': 1000, 'num_effective_samples': 200}
            kwargs['adaptation'] = False
            kwargs['L'] = 1.
        sampler = Sampler(likelihood, seed=12, save_fn=save_fn, **kwargs)
        chains[Sampler.__name__] = sampler.run(max_iterations=10000, check={'max_eigen_gr': 1., 'min_ess': 50}, check_every=200)[0]

    if likelihood.mpicomm.rank == 0:
        plotting.plot_triangle(list(chains.values()), labels=list(chains.keys()), show=True)


if __name__ == '__main__':

    setup_logging()
    #test_nautilus()
    #test_samplers()
    #test_fixed()
    #test_importance()
    #test_error()
    #test_mcmc()
    #test_hmc()
    #test_nested()
    #test_marg()
    #test_bao_hmc()
    test_folpsax_hmc()
