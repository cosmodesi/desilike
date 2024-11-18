import os
import time

import pytest
import numpy as np

from desilike import PipelineError, setup_logging
from desilike.samplers import (EmceeSampler, ZeusSampler, PocoMCSampler,
                               StaticDynestySampler, DynamicDynestySampler, PolychordSampler, NautilusSampler,
                               NUTSSampler, HMCSampler, MCLMCSampler, GridSampler, QMCSampler, ImportanceSampler)


def test_samplers():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = BAOPowerSpectrumTemplate(z=0.5)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['al*']): param.update(derived='.marg')
    for param in theory.params.select(basename=['al0_*']): param.update(derived='.marg_not_derived')
    #for param in theory.params.select(basename=['al*']): param.update(derived='.marg_not_derived')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov, name='LRG')

    for Sampler in [EmceeSampler, NUTSSampler, ZeusSampler, PocoMCSampler, StaticDynestySampler, DynamicDynestySampler, NautilusSampler, PolychordSampler][:3]:
        kwargs = {'seed': 42}
        if Sampler in [EmceeSampler, ZeusSampler]:
            kwargs.update(nwalkers=20)
        elif Sampler in [PocoMCSampler]:
            kwargs.update(n_active=20)
        elif Sampler in [StaticDynestySampler, DynamicDynestySampler, PolychordSampler, NautilusSampler]:
            kwargs.update(nlive=100)
        save_fn = ['./_tests/chain_{:d}.npz'.format(i) for i in range(min(likelihood.mpicomm.size, 2))]
        sampler = Sampler(likelihood, save_fn=save_fn, **kwargs)
        kwargs = {}
        if Sampler in [EmceeSampler, ZeusSampler]:
            kwargs.update(thin_by=2)
        chains = sampler.run(max_iterations=100, check=True, check_every=20, **kwargs)
        if sampler.mpicomm.rank == 0:
            assert chains[0].attrs['ndof']
            assert chains[0].attrs['hartlap2007_factor'] is None
            assert chains[0]['LRG.loglikelihood'].param.latex() == 'L_{\mathrm{LRG}}'
            assert chains[0]['LRG.loglikelihood'].param.derived
            assert chains[0].logposterior.param.latex() == '\mathcal{L}'
            assert chains[0].logposterior.param.derived
            #print(chains[0]['LRG.loglikelihood'].derivs)
            chains[0]['LRG.loglikelihood']['al2_0', 'al2_0']
            with pytest.raises(KeyError):
                chains[0]['LRG.loglikelihood']['al0_0', 'al0_0']
            chains[0].sample_solved()
            assert np.allclose(chains[0].logposterior, chains[0]['LRG.loglikelihood'][()] + chains[0]['LRG.logprior'][()])
        size1 = sampler.mpicomm.bcast(chains[0].size if sampler.mpicomm.rank == 0 else None, root=0)
        chains = sampler.run(max_iterations=0, check=True, check_every=10)
        size2 = sampler.mpicomm.bcast(chains[0].size if sampler.mpicomm.rank == 0 else None, root=0)
        assert size2 == size1, (size2, size1)
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

    likelihood.varied_params['omega_cdm'].update(prior=dict(), ref=dict(limits=(-10., -9.)))

    sampler = EmceeSampler(likelihood, seed=42, max_tries=10)
    with pytest.raises(ValueError):
        sampler.run(max_iterations=10)

    likelihood.init.update(catch_errors=[])
    likelihood.varied_params['omega_cdm'].update(prior=dict(), ref=dict(limits=(-10., -9.)))

    sampler = EmceeSampler(likelihood, seed=42, max_tries=10)
    with pytest.raises(PipelineError):
        sampler.run(max_iterations=10)

    likelihood.init.update(catch_errors=None)
    likelihood.varied_params['omega_cdm'].update(prior=dict(), ref=dict(limits=(-0.1, 0.3)))

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


def test_ensemble():

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
    from desilike.samplers import MCMCSampler

    chains, timing = {}, {}
    for Sampler in [MCMCSampler, EmceeSampler][:1]:
        kwargs, check = {}, True
        ensemble = Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler]
        check = {'max_eigen_gr': 0.01, 'stable_over': 3}
        if ensemble:
            kwargs.update(nwalkers=10)
            check = {'max_eigen_gr': 0.01, 'stable_over': 3}
        if Sampler is MCMCSampler:
            kwargs['blocks'] = [[1, ['a']], [2, ['b', 'c']]]
        save_fn = './_tests/chain_{}_0.npy'
        nchains = save_fn if os.path.isfile(save_fn) else 1
        nchains = 4
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
    #for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler]:
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

    plotting.plot_triangle(list(chains.values()), labels=list(chains.keys()), show=True)


def test_hmc():

    likelihood = Likelihood()
    #likelihood.varied_params['a'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 0.2, 'limits': [-0.5, 0.5]})
    #likelihood.all_params['a'].update(derived='.marg')
    likelihood()

    from desilike.samples import plotting
    chains = {}
    for Sampler in [NUTSSampler, HMCSampler, MCLMCSampler][-1:]:
        sampler = Sampler(likelihood, adaptation=False, save_fn='./_tests/chain_*.npy')
        chain = sampler.run(check={'max_eigen_gr': 0.01}, min_iterations=100, check_every=200)[0]
        chains[Sampler.__name__] = chain.remove_burnin(0.2)

    from desilike import Fisher
    fisher = Fisher(likelihood, method='auto')
    fisher = fisher()
    plotting.plot_triangle(list(chains.values()) + [fisher], labels=list(chains.keys()) + ['fisher'], show=True)


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
        template = BAOPowerSpectrumTemplate(z=z)
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
        for param in theory.params.select(basename=['al*']): param.update(derived='.best')
        return theory

    def get_theory(z):
        template = ShapeFitPowerSpectrumTemplate(z=z)
        theory = FOLPSAXTracerPowerSpectrumMultipoles(template=template)
        for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
        return theory

    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.3, 0.005], 2: [0.05, 0.3, 0.005]},
                                                         data={},
                                                         theory=get_theory(z=0.5))
    footprint = BoxFootprint(volume=1e10, nbar=1e-3)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=1)()

    observable2 = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.3, 0.005], 2: [0.05, 0.3, 0.005]},
                                                         data={},
                                                         theory=get_theory(z=1.))

    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov, name='LRG')
    likelihood()
    #cov = np.eye(cov.shape[0] * 2)
    #likelihood = ObservablesGaussianLikelihood(observables=[observable, observable2], covariance=cov, name='LRG')
    #likelihood()
    import jax
    jax.config.update("jax_log_compiles", True)

    chains = {}
    for Sampler in [NUTSSampler, MCLMCSampler][:1]:
        save_fn = ['./_tests/chain_{:d}.npz'.format(i) for i in range(min(likelihood.mpicomm.size, 1))]
        kwargs = {'adaptation': False}
        if Sampler is MCLMCSampler:
            #kwargs['adaptation'] = {'niterations': 1000, 'num_effective_samples': 200}
            kwargs['adaptation'] = False
            kwargs['L'] = 1.
        sampler = Sampler(likelihood, seed=12, save_fn=save_fn, **kwargs)
        chains[Sampler.__name__] = sampler.run(max_iterations=10000, check={'max_eigen_gr': 1., 'min_ess': 50}, check_every=200)[0]

    if likelihood.mpicomm.rank == 0:
        plotting.plot_triangle(list(chains.values()), labels=list(chains.keys()), show=True)


def test_cobaya_mcmc():

    class MyGaussianLikelihood(BaseGaussianLikelihood):

        def initialize(self, mean=None, covariance=None):
            self.mean = np.array(mean)
            self.covariance = np.array(covariance)
            y = np.zeros(len(self.params))
            super(MyGaussianLikelihood, self).initialize(y, covariance=self.covariance)

        def calculate(self, **kwargs):
            from jax import numpy as jnp
            self.flattheory = jnp.array([kwargs[name] for name in self.params.names()])
            super(MyGaussianLikelihood, self).calculate()


    likelihood = MyGaussianLikelihood(mean=[0., 0.], covariance=[[0.1, 0.05], [0.05, 0.2]])
    likelihood.init.params = {'a': {'prior': {'limits': [-0.5, 3]}, 'proposal': 0.4}, 'b': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'proposal': 0.5}}
    #likelihood.init.params = {'a': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'proposal': 0.5}, 'b': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'proposal': 0.5}}
    from desilike.samplers import MCMCSampler
    from desilike.samples import plotting

    from desilike import Fisher
    fisher = Fisher(likelihood, method='auto')
    fisher = fisher()

    sampler = MCMCSampler(likelihood, proposal_scale=2.4)
    niterations = 10000
    gr = 0.03
    learn_every = 80
    chain_desilike = sampler.run(check_every=learn_every, max_iterations=niterations, check={'max_eigen_gr': gr, 'stable_over': 2})[0].remove_burnin(0.3)

    #likelihood = {'gaussian_mixture': {'means': [0.2, 0], 'covs': [[0.1, 0.05], [0.05, 0.2]]}}
    #params = {'a': {'prior': {'min': -0.5, 'max': 3}, 'proposal': 0.4}, 'b': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'proposal': 0.5}}
    from desilike.bindings.cobaya import CobayaLikelihoodFactory

    likelihood_cobaya = CobayaLikelihoodFactory(lambda: likelihood, params=True)
    sampler = {'mcmc': {'proposal_scale': 2.4, 'max_samples': niterations, 'Rminus1_stop': gr, 'Rminus1_cl_stop': 1., 'learn_every': learn_every, 'output_every': learn_every}}
    info = {'likelihood': {'my_likelihood': likelihood_cobaya}, 'sampler': sampler, 'output': None}
    from cobaya.run import run
    updated_info, sampler = run(info)
    from desilike.samples import Chain
    samples = sampler.products()['sample']
    samples = {name: np.asarray(samples[name]) for name in ['a', 'b', 'weight']}
    samples['fweight'] = samples.pop('weight')
    print(samples['fweight'].min(), samples['fweight'].max())
    chain_cobaya = Chain(samples).remove_burnin(0.3)
    for param in chain_cobaya.params(): param.update(fixed=False)

    print(len(chain_desilike), chain_desilike['a'].sum(), len(chain_cobaya))
    from desilike.samples import diagnostics
    from matplotlib import pyplot as plt

    def gelman_rubin_cobaya(chains, params=None, nsplits=None, statistic='mean', method='eigen', return_matrices=False, check_valid='raise'):
        from desilike import utils
        if not utils.is_sequence(chains):
            chains = [chains]
        nchains = len(chains)
        if nchains < 2:
            if nsplits is None or nchains * nsplits < 2:
                raise ValueError('Provide a list of at least 2 chains to estimate Gelman-Rubin, or specify nsplits >= {:d}'.format(int(2. / nchains + 0.5)))
            chains = [chain[islab * len(chain) // nsplits:(islab + 1) * len(chain) // nsplits] for islab in range(nsplits) for chain in chains]
        sizes = [chain.size for chain in chains]
        if any(size < 2 for size in sizes):
            raise ValueError('Not enough samples ({}) to estimate Gelman-Rubin'.format(sizes))
        if params is None: params = chains[0].params(varied=True)
        nchains = len(chains)

        if statistic == 'mean':

            def statistic(chain, params):
                return [chain.mean(param) for param in params]

        means = np.asarray([statistic(chain, params) for chain in chains])
        covs = np.asarray([chain.covariance(params) for chain in chains])
        wsums = np.asarray([chain.weight.sum() for chain in chains])
        mean_of_covs = np.average(covs, weights=wsums, axis=0)
        cov_of_means = np.atleast_2d(np.cov(means.T))
        d = np.sqrt(np.diag(cov_of_means))
        corr_of_means = (cov_of_means / d).T / d
        norm_mean_of_covs = (mean_of_covs / d).T / d
        # Cholesky of (normalized) mean of covs and eigvals of Linv*cov_of_means*L
        L = np.linalg.cholesky(norm_mean_of_covs)
        Linv = np.linalg.inv(L)
        eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
        return np.abs(eigvals) + 1.

    def plot_gelman_rubin(chain, label, func=diagnostics.gelman_rubin):
        nsteps = np.arange(100, chain.size + 1, 100)
        gr = np.array([func(chain[:end], nsplits=4, statistic='mean', method='eigen', return_matrices=False, check_valid='raise').max() for end in nsteps])
        plt.plot(nsteps, gr, label=label)
        return gr

    gr = plot_gelman_rubin(chain_desilike, label='desilike')
    gr = plot_gelman_rubin(chain_cobaya, label='cobaya')
    gr_c = plot_gelman_rubin(chain_cobaya, label='cobaya gr', func=gelman_rubin_cobaya)
    print((gr_c - 1.) / (gr - 1.))
    plt.legend()
    plt.show()

    plotting.plot_triangle([fisher, chain_desilike, chain_cobaya], labels=['fisher', 'desilike', 'cobaya'], show=True)


def test_sample_solved():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = BAOPowerSpectrumTemplate(z=0.5)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.init.params.select(basename=['al*']): param.update(derived='.marg', prior=dict(dist='norm', loc=0., scale=0.3))
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov, name='LRG')

    sampler = EmceeSampler(likelihood, seed=42)
    samples = sampler.run(max_iterations=1)[0].ravel()

    samples_solved = samples.sample_solved()
    for param in theory.init.params.select(basename=['al*']): param.update(derived=False)
    for i in range(samples_solved.size):
        params = {param.name: samples_solved[param][i] for param in samples_solved.params(input=True)}
        print(likelihood(params) - samples_solved.logposterior[i])


def test_hard_prior():

    class MyGaussianLikelihood(BaseGaussianLikelihood):

        def initialize(self, mean=None, covariance=None):
            self.mean = np.array(mean)
            self.covariance = np.array(covariance)
            y = np.zeros(len(self.params))
            super(MyGaussianLikelihood, self).initialize(y, covariance=self.covariance)

        def calculate(self, **kwargs):
            from jax import numpy as jnp
            self.flattheory = jnp.array([kwargs[name] for name in self.params.names()])
            super(MyGaussianLikelihood, self).calculate()

    likelihood = MyGaussianLikelihood(mean=[0.], covariance=[[0.05]])
    likelihood.init.params = {'a': {'prior': {'limits': [0., 3]}, 'proposal': 0.2}}
    #likelihood.init.params = {'a': {'prior': {}, 'ref':  {'dist': 'norm', 'loc': 0., 'scale': 0.1}, 'proposal': 0.2}}
    #likelihood.init.params = {'a': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'proposal': 0.5}, 'b': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'proposal': 0.5}}
    from desilike.samplers import MCMCSampler
    from desilike.samples import plotting

    sampler = MCMCSampler(likelihood, proposal_scale=2.4)
    gr = 0.01
    learn_every = 1000
    chain = sampler.run(check_every=learn_every, min_iterations=50000, check={'max_eigen_gr': gr, 'stable_over': 2})[0].remove_burnin(0.3)

    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.hist(chain['a'], bins=50)
    plt.show()



if __name__ == '__main__':

    setup_logging()
    #test_samplers()
    #test_nautilus()
    #test_fixed()
    #test_importance()
    #test_error()
    #test_ensemble()
    test_hmc()
    #test_nested()
    #test_marg()
    #test_bao_hmc()
    #test_cobaya_mcmc()
    #test_sample_solved()
    #test_hard_prior()
