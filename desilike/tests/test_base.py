import numpy as np

from desilike import setup_logging


def test_init():

    from desilike.base import BaseConfig, InitConfig
    from desilike import ParameterCollection
    params = ParameterCollection({'a': {'ref': {'limits': [0., 1.]}}})
    init = InitConfig(params=params)


def test_galaxy_clustering():
    """
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, DirectPowerSpectrumTemplate
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    theory = KaiserTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(logA=3.04, b1=1.).shape
    theory = KaiserTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(logA=3.04, b1=1.).shape

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    print(theory.runtime_info.pipeline.params)
    theory(dm=0.01, b1=1.).shape
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    print(theory.runtime_info.pipeline.params)
    theory(dm=0.01, b1=1.).shape

    from desilike.theories.galaxy_clustering import PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles

    theory = PyBirdTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(logA=3.04, b1=1.).shape
    theory = PyBirdTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(logA=3.04, b1=1.).shape

    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles

    theory = PNGTracerPowerSpectrumMultipoles(method='prim')
    print(theory.runtime_info.pipeline.params)
    params = dict(fnl_loc=100., b1=2.)
    theory2 = PNGTracerPowerSpectrumMultipoles(method='matter')
    assert np.allclose(theory2(**params), theory(**params), rtol=2e-3)
    assert not np.allclose(theory2(fnl_loc=0.), theory(), rtol=2e-3)

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
    """
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    theory(qpar=1.1, sigmapar=3.)

    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate
    template = BAOPowerSpectrumTemplate(z=0.1, fiducial='DESI', apmode='qiso')
    theory.init.update(template=template)


def test_observable():
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    observable()
    #observable.wmatrix.plot(show=True)
    theory.template.init.update(z=1.)
    del theory.template.params['dm']
    observable()
    print(observable.runtime_info.pipeline.varied_params)
    assert theory.template.z == 1.


def test_likelihood():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipoles
    from desilike.likelihoods import ObservablesGaussianLikelihood

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, BAOPowerSpectrumTemplate
    template = BAOPowerSpectrumTemplate(z=1.)
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['sigma*', 'al*_-3', 'al*_-2']):
        param.update(value=0., fixed=True)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.08, 0.2]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
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
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
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
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    print(likelihood.runtime_info.pipeline.params.select(solved=True))
    print(likelihood.varied_params)
    print(likelihood(dm=0.), likelihood(dm=0.01), likelihood(dm=0.02))
    likelihood()
    observable.plot(show=True)


def test_params():

    import copy
    from desilike.parameter import Parameter, ParameterCollection
    from scipy import stats
    params = ParameterCollection([Parameter('test', prior=dict(dist=stats.uniform, limits=(0., 1.)))])
    copy.deepcopy(params)

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
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
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
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
    assert np.allclose(likelihood.loglikelihood, bak)# XXX:


def test_cosmo():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=1.4, cosmo='external'))
    print(theory.runtime_info.pipeline.get_cosmo_requires())
    print(theory.runtime_info.pipeline.params)

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=1.4))
    print(theory.runtime_info.pipeline.get_cosmo_requires())
    print(theory.runtime_info.pipeline.params)


def test_differentiation():

    import timeit
    import numpy as np
    from desilike.jax import jax
    from desilike.jax import numpy as jnp

    def f(a, b):
        return jnp.sum(a * b)

    jac = jax.jacrev(f)
    jac(1., 3.)

    a = np.arange(10)
    number = 100000
    d = {}
    d['np-sum'] = {'stmt': "np.sum(a)", 'number': number}
    d['jnp-sum'] = {'stmt': "jnp.sum(a)", 'number': number}

    for key, value in d.items():
        dt = timeit.timeit(**value, globals={**globals(), **locals()}) #/ value['number'] * 1e3
        print('{} takes {: .3f} milliseconds'.format(key, dt))

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate

    from desilike import Differentiation
    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.4))
    theory.params['power'] = {'derived': True}
    theory(sn0=100.)
    diff = Differentiation(theory, method=None, order=2)
    diff()


def test_copy():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
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


def test_fisher():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    likelihood.all_params['logA'].update(derived='jnp.log(10 *  {A_s})', prior=None)
    likelihood.all_params['A_s'] = {'prior': {'limits': [1.9, 2.2]}, 'ref': {'dist': 'norm', 'loc': 2.083, 'scale': 0.01}}

    from desilike import Fisher
    fisher = Fisher(likelihood)
    matrix = fisher()
    print(matrix.to_covariance().to_stats())

    fisher = Fisher(likelihood)
    matrix = fisher()


def test_install():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles

    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']):
        param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    from desilike import Installer
    Installer()(likelihood)
    from desilike.samplers import EmceeSampler
    Installer()(EmceeSampler)


if __name__ == '__main__':

    setup_logging()
    #test_init()
    #test_galaxy_clustering()
    #test_observable()
    #test_likelihood()
    #test_params()
    #test_copy()
    #test_differentiation()
    #test_fisher()
    #test_cosmo()
    #test_install()
