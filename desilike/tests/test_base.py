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
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy',# wmatrix='_pk/window.npy',
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
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    from desilike import Installer
    Installer()(likelihood)
    from desilike.samplers import EmceeSampler
    Installer()(EmceeSampler)


if __name__ == '__main__':

    setup_logging()
    #test_init()
    #test_observable()
    #test_likelihood()
    #test_params()
    test_copy()
    #test_cosmo()
    #test_install()
