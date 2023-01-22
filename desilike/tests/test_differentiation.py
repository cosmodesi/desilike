import numpy as np

from desilike import setup_logging


def test_misc():
    from desilike.differentiation import deriv_nd, deriv_grid

    X = np.linspace(0., 1., 11)[..., None]
    Y = np.linspace(0., 1., 11)[..., None]
    center = X[0]
    print(deriv_nd(X, Y, orders=[(0, 1, 2)], center=center))

    deriv = deriv_grid([(np.array([0]), np.array([0]), 0)] * 3)
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))

    deriv = deriv_grid([(np.linspace(-1., 1., 3), [1, 0, 1], 2)] * 3)
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))

    deriv = deriv_grid([(np.linspace(-1., 1., 3), [1, 0, 1], 2), (np.linspace(-1., 1., 5), [1, 1, 0, 1, 1], 1)])
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))

    deriv = deriv_grid([(np.linspace(-1., 1., 3), [1, 0, 1], 2)] * 20)
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))


def test_jax():
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


def test_differentiation():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate

    from desilike import Differentiation
    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.4))
    theory.params['power'] = {'derived': True}
    theory(sn0=100.)
    diff = Differentiation(theory, method=None, order=2)
    diff()
    diff(sn0=50.)


def test_fisher_galaxy():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=0.5))
    #for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2], 2: [0.05, 0.18]}, kstep=0.01,
                                                         data='_pk/data.npy', mocks='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=False)
    likelihood.all_params['logA'].update(derived='jnp.log(10 *  {A_s})', prior=None)
    likelihood.all_params['A_s'] = {'prior': {'limits': [1.9, 2.2]}, 'ref': {'dist': 'norm', 'loc': 2.083, 'scale': 0.01}}
    for param in likelihood.all_params.select(name=['m_ncdm', 'w0_fld', 'wa_fld', 'Omega_k']):
        param.update(fixed=False)

    #print(likelihood(w0_fld=-1), likelihood(w0_fld=-1.1))
    #print(likelihood(wa_fld=0), likelihood(wa_fld=0.1))
    from desilike import Fisher
    fisher = Fisher(likelihood)
    matrix = fisher()
    print(matrix.to_covariance().to_stats())

    fisher = Fisher(likelihood)
    matrix = fisher()


def test_fisher_cmb():
    from desilike import Fisher
    from desilike.likelihoods.cmb import BasePlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood,\
                                         EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood
    from desilike.likelihoods import SumLikelihood
    from desilike.theories.primordial_cosmology import Cosmoprimo
    # Now let's turn to Planck (lite) clik likelihoods
    cosmo = Cosmoprimo(fiducial='DESI')
    likelihoods = [Likelihood(cosmo=cosmo) for Likelihood in [TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood,\
                                                              EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood]]
    likelihood_clik = SumLikelihood(likelihoods=likelihoods)
    #for param in likelihood_clik.all_params:
    #    param.update(fixed=True)
    likelihood_clik.all_params['m_ncdm'].update(fixed=False)
    fisher_clik = Fisher(likelihood_clik)
    # Planck covariance matrix used above should roughly correspond to Fisher at the Planck posterior bestfit
    # at which logA ~= 3.044 (instead of logA = ln(1e10 2.0830e-9) = 3.036 assumed in the DESI fiducial cosmology)
    precision_clik = fisher_clik()
    print(precision_clik.to_covariance().to_stats(tablefmt='pretty'))


if __name__ == '__main__':

    setup_logging()
    #test_misc()
    #test_differentiation()
    #test_fisher_galaxy()
    test_fisher_cmb()
