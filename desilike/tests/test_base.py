import numpy as np

from desilike import setup_logging


def test_init():

    from desilike.base import BaseConfig, InitConfig
    from desilike import ParameterCollection
    params = ParameterCollection({'a': {'ref': {'limits': [0., 1.]}}})
    init = InitConfig(params=params)


def test_observable():
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    observable()
    #observable.wmatrix.plot(show=True)
    theory.template.init.update(z=1.)
    del theory.template.params['dm']
    observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.template.z == 1.


def test_likelihood():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    template = BAOPowerSpectrumTemplate(z=1.)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['sigma*', 'al*_-3', 'al*_-2']):
        param.update(value=0., fixed=True)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.08, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    if likelihood.mpicomm.rank == 0:
        likelihood(b1=2.)

    #observable.plot(show=True)
    print(theory.pt.params)
    print(likelihood.varied_params)
    template = BAOPowerSpectrumTemplate(z=0.5, apmode='qiso')
    theory.init.update(template=template)
    likelihood()
    print(likelihood.varied_params)

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    print(likelihood.runtime_info.pipeline.params)
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(b1=2., dm=0.02))
    theory.template.init.update(z=1.)
    #del theory.template.params['dm']
    print(likelihood.runtime_info.pipeline.varied_params)
    likelihood()
    #observable.plot(show=False)

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    print(likelihood.runtime_info.pipeline.params.select(solved=True))
    print(likelihood.varied_params)
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(dm=0.02))
    likelihood()
    observable.plot(show=True)


def test_combined_likelihood():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    theory.params['sn0'].update(namespace='LRG')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood1 = ObservablesGaussianLikelihood(observables=[observable])
    likelihood1.all_params['LRG.sn0'].update(derived='.auto')
    print(likelihood1.varied_params)
    #print(theory.runtime_info.params['LRG.sn0'].derived)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    theory.params['sn0'].update(namespace='ELG')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood2 = ObservablesGaussianLikelihood(observables=[observable])
    likelihood2.all_params['ELG.sn0'].update(derived='.auto')

    likelihood = likelihood1 + likelihood2
    print(likelihood.varied_params)


def test_params():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    likelihood.observables[0].wmatrix.theory.params['b1'].update(value=3.)
    print(likelihood(), likelihood.runtime_info.pipeline.input_values)

    print(likelihood.runtime_info.pipeline.params)
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(b1=2., dm=0.02))
    print(likelihood.varied_params)
    likelihood.all_params = {'dm': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1}}}
    print(likelihood.varied_params)
    assert likelihood.varied_params['dm'].prior.scale == 1.
    import pytest
    from desilike.base import PipelineError
    with pytest.raises(PipelineError):
        likelihood.all_params = {'a': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}}}
    likelihood.all_params = 'test_params.yaml'
    assert likelihood.varied_params['dm'].prior.scale == 2.
    likelihood.all_params['dm'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 100.})
    assert likelihood.varied_params['dm'].prior.scale == 100.
    likelihood.all_params = {'*': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}}}
    assert likelihood.varied_params['dm'].prior.scale == 1.

    theory = KaiserTracerPowerSpectrumMultipoles()
    theory.params['b1'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 1.})
    theory.params = {'b1': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}}, 'sn0': {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1e4}}}
    theory.all_params['Omega_m'].update(prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.5})
    theory.all_params = {'*mega_m': {'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.5}}}
    assert theory.template.cosmo.params['Omega_m'].ref.scale == 0.5
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood.all_params = {'sn0': {'derived': '.marg'}}
    likelihood(b1=1.5)
    bak = likelihood.loglikelihood
    print(likelihood.varied_params)
    likelihood.all_params['b1'].update(derived='{b}**2', prior=None)
    likelihood.all_params['b'] = {'prior': {'limits': [0., 2.]}}
    print(likelihood.varied_params)
    likelihood(b=1.5**0.5)
    assert np.allclose(likelihood.loglikelihood, bak)


def test_copy():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    #likelihood()

    likelihood.all_params['sn0'].update(derived=False)
    #print(likelihood.varied_params)
    likelihood2 = likelihood.deepcopy()
    #for param in likelihood.all_params.select(basename=['sn*']): param.update(derived=False)
    likelihood.all_params['sn0'].update(derived=False)
    print(likelihood.varied_params)
    print(likelihood2.varied_params)
    assert np.allclose(likelihood2(), SumLikelihood(likelihoods=likelihood2)())

    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood


    template = BAOPowerSpectrumTemplate(z=0.5, fiducial='DESI')
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    # Set damping sigmas to zero, as data follows linear pk
    for param in theory.params.select(basename='sigma*'):
        param.update(value=0., fixed=True)
    # Fix some broadband parameters (those with k^{-3} and k^{-2}) to speed up calculation in this notebook
    for param in theory.params.select(basename=['al*_-3', 'al*_-2']):
        param.update(value=0., fixed=True)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    template = BAOPowerSpectrumTemplate(z=0.5, fiducial='DESI', apmode='qiso', only_now=False)
    theory.init.update(template=template)
    assert 'qiso' in likelihood.all_params


def test_cosmo():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=1.4, cosmo='external'))
    print(theory.runtime_info.pipeline.get_cosmo_requires())
    print(theory.runtime_info.pipeline.params)

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=1.4))
    print(theory.runtime_info.pipeline.get_cosmo_requires())
    print(theory.runtime_info.pipeline.params)


def test_install():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles

    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']):
        param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    from desilike import Installer
    Installer()(likelihood)
    from desilike.samplers import EmceeSampler
    Installer()(EmceeSampler)


def test_vmap():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles
    from desilike import vmap
    import jax

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['sn*']):
        param.update(derived='.marg')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()

    size = (3,)
    params = {param.name: param.ref.sample(size=size, random_state=42) if param.ref.is_proper() else np.full(param.value, size) for param in likelihood.varied_params}
    if True:
        vlikelihood = vmap(likelihood, errors='return')
        toret = vlikelihood(params, return_derived=True)
        if likelihood.mpicomm.rank == 0:
            print(toret)
    if True:
        vlikelihood = vmap(likelihood, backend='jax', errors='return')
        toret = vlikelihood(params, return_derived=True)
        if likelihood.mpicomm.rank == 0:
            print(toret[0][1]['loglikelihood'].derivs)
            print(toret)
        toret = jax.jit(vlikelihood, static_argnames=['return_derived'])(params, return_derived=True)
        if likelihood.mpicomm.rank == 0:
            print(toret)

    if True:
        vlikelihood = vmap(likelihood, backend='mpi', errors='return', return_derived=True)
        toret = vlikelihood(params)
        if likelihood.mpicomm.rank == 0:
            print(toret)
    if True:
        vlikelihood = vmap(jax.jit(likelihood, static_argnames=['return_derived']), backend='mpi', errors='return')
        toret = vlikelihood(params, return_derived=True)
        if likelihood.mpicomm.rank == 0:
            print(toret)
    if True:
        vlikelihood = vmap(jax.jit(vmap(likelihood, backend='jax', errors='return'), static_argnames=['return_derived']), backend='mpi', errors='return')
        toret = vlikelihood(params, return_derived=True)
        if likelihood.mpicomm.rank == 0:
            print(toret)


def test_cosmo():

    from desilike.theories import Cosmoprimo

    # Provides primordial cosmology computation. You can also give default, fixed, parameters here... (like m_ncdm, or class precision parameters)
    cosmo = Cosmoprimo(engine='class', m_ncdm=[0.10])
    cosmo.init.params['w0_fld'].update(derived='({w1} + {w2}) / 2.')  # w0_fld expressed as derived parameter
    cosmo.init.params['wa_fld'].update(derived='({w1} - {w2}) / 2.')  # wa_fld expressed as derived parameter
    # w1, w2 as sampled "varied" parameters
    cosmo.init.params['w1'] = dict(value=-1., prior=dict(dist='uniform', limits=[-5., 0.]))
    cosmo.init.params['w2'] = dict(value=0., prior=dict(dist='norm', loc=0., scale=1.))
    print(cosmo.varied_params)  # varied parameters (also cosmo.all_params(varied=True)): ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s', 'tau_reio', 'w1', 'w2']

    # Example: interaction cosmology / theory
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles
    cosmo.init.params['tau_reio'].update(fixed=True)  # tau is useless
    template = DirectPowerSpectrumTemplate(cosmo=cosmo, z=1.4)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    print(theory.varied_params)  # varied parameters (including that of Kaiser): ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s', 'tau_reio', 'w1', 'w2', 'b1', 'sn0']
    poles = theory(w2=0.5, b1=2.)  # theory prediction

    if False:
        # For a galaxy power spectrum likelihood with the previous theory model
        from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
        from desilike.likelihoods import ObservablesGaussianLikelihood
        observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                            data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                            theory=theory)
        likelihood = ObservablesGaussianLikelihood(observables=[observable])
    if True:
        # With Planck likelihood
        from desilike.likelihoods.cmb import TTTEEEHighlPlanck2018LiteLikelihood
        # to install the likelihood
        #from desilike import Installer
        #Installer(user=True)(TTTEEEHighlPlanck2018LiteLikelihood)
        cosmo.init.params['tau_reio'].update(fixed=False)
        # Planck likelihood has its own data / covariance
        likelihood = TTTEEEHighlPlanck2018LiteLikelihood(cosmo=cosmo)  # ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s', 'tau_reio', 'w1', 'w2', 'A_planck']
        print(likelihood.varied_params)


    # Posterior profiling
    from desilike.profilers import MinuitProfiler
    profiler = MinuitProfiler(likelihood, seed=7)
    #profiles = profiler.maximize(niterations=2)  # profiles is an object which contains the result of the fit

    # Posterior sampling
    from desilike.samplers import MCMCSampler
    sampler = MCMCSampler(likelihood, chains=8, seed=7)
    samples = sampler.run(check={'max_eigen_gr': 0.01, 'stable_over': 3}, check_every=10, max_iterations=100)  # profiles is an object which contains samples




if __name__ == '__main__':

    setup_logging()
    #test_init()
    #test_observable()
    #test_likelihood()
    #test_combined_likelihood()
    #test_params()
    #test_copy()
    #test_cosmo()
    #test_install()
    #test_vmap()
    test_cosmo()
