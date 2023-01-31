import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.likelihoods.cmb import (BasePlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
                                      TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood,
                                      LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood)


def test_install():
    for Likelihood in (BasePlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
                       TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood,
                       LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood)[:1]:
        if 'Unbinned' in Likelihood.__name__: continue
        if 'Lite' in Likelihood.__name__: continue
        print(Likelihood.__name__)
        likelihood = Likelihood()
        likelihood.params['planck.loglikelihood'] = {}
        likelihood.params['planck.logprior'] = {}
        installer = Installer(user=True)
        installer(likelihood)
        assert np.allclose((likelihood + likelihood)(), 2. * likelihood() - likelihood.logprior)


def test_clik():
    likelihood = TTTEEEHighlPlanck2018PlikLiteLikelihood()
    likelihood()
    TTTEEEHighlPlanck2018PlikLiteLikelihood()()


def test_sum():
    from desilike.likelihoods import SumLikelihood
    likelihood = SumLikelihood([Likelihood() for Likelihood in [TTTEEEHighlPlanck2018PlikLiteLikelihood, LensingPlanck2018ClikLikelihood]])
    print(likelihood())


def test_gaussian_likelihood():

    likelihood = BasePlanck2018GaussianLikelihood(source='covmat')
    likelihood()
    covmat = likelihood.covariance

    likelihood = BasePlanck2018GaussianLikelihood(source='chains')
    likelihood()
    chains = likelihood.covariance

    print(covmat.to_stats(tablefmt='pretty'))
    print(chains.to_stats(tablefmt='pretty'))
    print(np.abs((chains._value - covmat._value) / covmat._value))
    from desilike.samples import plotting
    plotting.plot_triangle([chains, covmat], labels=['chains', 'covmat'], show=True)


def test_params():
    from desilike.likelihoods.cmb import TTTEEEHighlPlanck2018PlikLiteLikelihood
    from desilike.theories.primordial_cosmology import Cosmoprimo

    planck_avg = {'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237, 'logA': 3.044, 'n_s': 0.9649, 'tau_reio': 0.0544}

    cosmo = Cosmoprimo()
    for key, val in planck_avg.items(): cosmo.all_params[key].update(value=val)
    testL = TTTEEEHighlPlanck2018PlikLiteLikelihood(cosmo=cosmo)
    testL()
    print(cosmo.varied_params['logA'].value)
    print(testL.varied_params['logA'].value)
    import time
    t0 = time.time()
    testL()
    print('in desilike', time.time() - t0)
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    t0 = time.time()
    cosmo.get_harmonic()
    print('in cosmoprimo', time.time() - t0)


def test_help():
    help(TTHighlPlanck2018PlikLikelihood)


def test_copy():
    from desilike import Fisher, setup_logging
    from desilike.likelihoods.cmb import (BasePlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood,
                                          TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood)
    from desilike.likelihoods import SumLikelihood
    from desilike.theories.primordial_cosmology import Cosmoprimo

    setup_logging()

    planck_avg = {'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237, 'logA': 3.044, 'n_s': 0.9649, 'tau_reio': 0.0544}
    planck_best = {'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237, 'logA': 3.044, 'n_s': 0.9649, 'tau_reio': 0.0544}
    cosmodefault = Cosmoprimo()
    cosmo = cosmodefault.copy()
    cosmoother = cosmodefault.copy()
    cosmo(**planck_best)
    cosmoother(**planck_avg)

    likelihoods = [Likelihood(cosmo=cosmo) for Likelihood in [TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood,
                                                              EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood]]
    likelihood_clik = SumLikelihood(likelihoods=likelihoods)
    likelihood_clik()


if __name__ == '__main__':

    setup_logging()
    #test_install()
    #test_clik()
    #test_sum()
    #test_gaussian_likelihood()
    test_params()
    #test_help()
    #test_copy()
