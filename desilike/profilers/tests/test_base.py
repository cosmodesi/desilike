import numpy as np

from desilike import setup_logging
from desilike.profilers import MinuitProfiler, ScipyProfiler, BOBYQAProfiler, OptaxProfiler


def test_profilers():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data='../../tests/_pk/data.npy', covariance='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=1., name='LRG')
    for param in likelihood.all_params.select(basename=['qpar', 'qper']):
        param.update(fixed=True)

    # for param in likelihood.varied_params:
    #     print(param, [likelihood(**{param.name: param.value + param.proposal * scale}) for scale in [-1., 1.]])
    for Profiler, kwargs in [(MinuitProfiler, {}),
                             (MinuitProfiler, {'gradient': True}),
                             #(ScipyProfiler, {'method': 'lsq'}),
                             (ScipyProfiler, {}),
                             (ScipyProfiler, {'gradient': True}),
                             (OptaxProfiler, {}),
                             (BOBYQAProfiler, {})][:1]:
        profiler = Profiler(likelihood, seed=42, **kwargs)
        profiles = profiler.maximize(niterations=2)
        assert profiles.bestfit.attrs['ndof']
        assert profiles.bestfit.attrs['hartlap2007_factor'] is not None
        assert profiles.bestfit['LRG.loglikelihood'].param.latex() == 'L_{\mathrm{LRG}}'
        assert profiles.bestfit['LRG.loglikelihood'].param.derived
        assert profiles.bestfit.logposterior.param.latex() == '\mathcal{L}'
        assert profiles.bestfit.logposterior.param.derived
        profiler.profile(params=['df'], size=4)
        profiler.grid(params=['df', 'dm'], size=(2, 2))
        if Profiler is MinuitProfiler:
            profiler.interval(params=['df'])
            profiler.contour(params=['df', 'dm'], cl=1, size=10)
            profiler.contour(params=['df', 'dm'], cl=2, size=10)
            profiler.contour(params=['df', 'b1'], cl=2, size=10)
            print(profiles.contour)
        print(profiler.profiles.to_stats())
        #likelihood()
        #observable.plot(show=True)

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    template = FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI')
    # Here we choose b-p parameterization
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode='b-p')
    theory.params['p'].update(fixed=True)  # not fixing p biases fnl_loc posterior
    observable = TracerPowerSpectrumMultipolesObservable(data='../../tests/_pk/data.npy', covariance='../../tests/_pk/mock_*.npy',
                                                         klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]}, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    profiler = MinuitProfiler(likelihood)
    profiler.maximize(niterations=2)


def test_solve():

    from desilike.theories.galaxy_clustering import EFTLikeKaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    template.init.params['f_sqrt_Ap'] = {'derived': True}
    theory = EFTLikeKaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    #for param in theory.params.select(basename=['df', 'dm', 'qpar', 'qper']): param.update(fixed=True)
    for param in theory.params.select(basename=['sn*']): param.update(prior=dict(dist='norm', loc=1., scale=0.01))
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={'b1': 2., 'ct0_2': 1., 'sn0': 0.5},
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable], name='LRG')
    #likelihood.params['LRG.loglikelihood'] = likelihood.params['LRG.logprior'] = {}
    likelihood()
    for param in likelihood.all_params.select(basename=['qpar']):
        param.update(fixed=True)
    profiler = MinuitProfiler(likelihood, rescale=False, seed=42)
    #profiler = ScipyProfiler(likelihood, method='lsq')
    profiles = profiler.maximize(niterations=2)
    print(profiles.to_stats())
    for param in likelihood.all_params.select(basename=['ct*', 'sn*']): param.update(derived='.best')
    #for param in theory.params.select(basename=['ct*', 'sn*']): param.update(fixed=True)
    for param in likelihood.all_params.select(basename=['sn*']): param.update(derived='.prec')
    #import numpy as np
    #likelihood.flatdata += 100 * np.cos(np.linspace(0., 5. * np.pi, observable.flatdata.size))
    profiler = MinuitProfiler(likelihood, rescale=True, seed=42)
    #profiler = ScipyProfiler(likelihood, method='lsq')
    profiles = profiler.maximize(niterations=2)
    print(profiles.to_stats())
    for param in likelihood.all_params.select(basename=['sn*']): param.update(derived='.best_not_derived')
    print('GRADIENT')
    profiler = MinuitProfiler(likelihood, rescale=False, gradient=True, seed=42)
    profiles = profiler.maximize(niterations=2)
    print(profiles.to_stats())
    profiles.bestfit['LRG.loglikelihood']['ct2_2', 'ct2_2']
    try: profiles.bestfit['LRG.loglikelihood']['sn0', 'sn0']
    except KeyError: pass
    else: raise ValueError
    for param in likelihood.all_params.select(basename=['sn*']): param.update(derived='.prec')
    profiler = MinuitProfiler(likelihood, rescale=False, seed=42)
    profiles = profiler.maximize(niterations=2)
    print(profiles.to_stats())
    profiler.interval(params=['df', 'b1'])
    #print(profiles.bestfit.attrs, profiles.error.attrs, profiles.covariance.attrs, profiles.interval.attrs)
    assert profiles.bestfit._loglikelihood == 'LRG.loglikelihood'
    #profiles = profiler.interval(params=['df'])
    #profiler.grid(params=['df', 'qpar'], size=2)
    print(profiles.bestfit['LRG.loglikelihood'], profiles.bestfit['f_sqrt_Ap'])
    likelihood(**profiles.bestfit.choice(input=True))
    #observable.plot(show=True)
    #print(likelihood(**profiles.bestfit.choice(varied=True)))
    #from desilike.samples import plotting
    #plotting.plot_triangle(profiles, show=True)


def test_bao():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    #for param in theory.params.select(basename=['df', 'dm', 'qpar', 'qper']): param.update(fixed=True)
    for param in theory.params.select(basename=['al*']): param.update(derived='.auto')
    for param in theory.params.select(basename=['al*']): param.update(fixed=True, derived='.auto')
    for param in theory.params.select(basename=['al0_0']): param.update(fixed=False, derived='.auto')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={'b1': 1.5},
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    #import numpy as np
    #likelihood.flatdata += 100 * np.cos(np.linspace(0., 5. * np.pi, observable.flatdata.size))
    profiler = MinuitProfiler(likelihood, rescale=False, seed=42)
    #profiler = ScipyProfiler(likelihood, method='lsq')
    profiles = profiler.maximize(niterations=1)
    profiles = profiler.profile(params=['qpar'], grid=np.linspace(0.8, 1.2, 3))
    assert profiles.profile['qpar'].shape == (3, 2)
    likelihood(**profiles.bestfit.choice(input=True))
    if likelihood.mpicomm.rank == 0:
        print(profiles.bestfit.choice(input=True))
        observable.plot(show=True)
    #print(likelihood(**profiles.bestfit.choice(varied=True)))
    #from desilike.samples import plotting
    #plotting.plot_triangle(profiles, show=True)


if __name__ == '__main__':

    setup_logging()
    #test_profilers()
    test_solve()
    #test_bao()
