import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.theories import Cosmoprimo
from desilike.likelihoods.supernovae import PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood, Union3SNLikelihood, DESY5SNLikelihood


def test_install():

    cosmo = Cosmoprimo(fiducial='DESI')
    for Likelihood, ref in [(PantheonSNLikelihood, -5649.588564438464),
                            (PantheonPlusSNLikelihood, -7526.797408441289),
                            (PantheonPlusSHOESSNLikelihood, -7600.468586494155),
                            (Union3SNLikelihood, -18.973744770948603),
                            (DESY5SNLikelihood, -1034.1324141181244)]:
        if Likelihood is Union3SNLikelihood:
            params = {'dM': -9.}
        elif Likelihood is DESY5SNLikelihood:
            params = {'Mb': 0.}
        else:
            params = {'Mb': -19.}
        likelihood = Likelihood(cosmo=cosmo)
        installer = Installer(user=True)
        installer(likelihood)
        likelihood(**params)
        print(likelihood.loglikelihood, ref)
        assert np.allclose(likelihood.loglikelihood, ref)
        assert np.allclose((likelihood + likelihood)(**params), 2. * likelihood(**params) - likelihood.logprior)
        for param in params:
            likelihood.all_params[param].update(prior=None, derived='.prec')
        likelihood()
        print(likelihood.varied_params)


def test_profile():
    from desilike.profilers import MinuitProfiler
    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = {'Omega_m': {'prior': {'limits': [0.1, 0.4]}}}
    likelihood = DESY5SNLikelihood(cosmo=cosmo)
    likelihood.init.params['Mb'].update(prior=None, derived='.prec')
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize(niterations=2)
    print(profiles.to_stats(tablefmt='pretty'))


def test_union3(plot=False):
    cosmo = Cosmoprimo(fiducial='DESI', engine='eisenstein_hu')
    if True:
        cosmo.init.params = {'Omega_m': {'prior': {'limits': [0.01, 0.9]}, 'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.002}, 'latex': '\Omega_m'},
                            'w0_fld': {'prior': {'limits': [-3, 1.]}, 'ref': {'dist': 'norm', 'loc': -1, 'scale': 0.05}, 'latex': 'w_0'},
                            'wa_fld': {'prior': {'limits': [-20., 2.]}, 'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.05}, 'latex': 'w_a'}}
    likelihood = Union3SNLikelihood(cosmo=cosmo)

    from desilike.samplers import NUTSSampler, MCMCSampler
    resume = False
    nchains = 4

    def samples_fn(correct_prior=False, i=0):
        #return '_tests/chain{}_{:d}.npy'.format('_corrected2' if correct_prior else '', i)
        return '_tests/chain{}_{:d}.npy'.format('_prior', i)

    if plot:

        from desilike.samples import Chain, plotting
        chains = []
        for correct_prior in [False, True]:
            chains.append(Chain.concatenate([Chain.load(samples_fn(correct_prior=correct_prior, i=i)).remove_burnin(0.5) for i in range(nchains)]))
        plotting.plot_triangle(chains, labels=['standard', 'corrected'], params=['Omega_m', 'w0_fld', 'wa_fld'], show=True)

    else:

        for correct_prior in [False, True][1:]:
            likelihood.init.update(correct_prior=correct_prior)
            likelihood.all_params['dM'].update(fixed=True)
            #print(likelihood({param.name: param.ref.sample() for param in likelihood.varied_params}))
            save_fn = [samples_fn(correct_prior=correct_prior, i=i) for i in range(nchains)]
            chains = save_fn if resume else nchains
            #sampler = NUTSSampler(likelihood, chains=chains, adaptation=False, save_fn=save_fn, seed=42)
            sampler = MCMCSampler(likelihood, chains=chains, save_fn=save_fn, seed=42)
            sampler.run(max_iterations=100000, check={'max_eigen_gr': 0.01, 'min_ess': 50}, check_every=200)


def test_desy5():
    import os
    from desilike.theories import Cosmoprimo
    from desilike.likelihoods.supernovae import DESY5SNLikelihood

    cosmo = Cosmoprimo(engine='class', fiducial='DESI')
    cosmo.init.params['Omega_m'] = dict(prior=dict(limits=[0., 1.]))
    cosmo.init.params.pop('omega_cdm')
    for name in ['w0_fld', 'wa_fld']: cosmo.init.params[name].update(fixed=False)
    likelihood = DESY5SNLikelihood(cosmo=cosmo)

    def cov_log_likelihood(mu_model, mu, inv_cov):
        # From https://github.com/des-science/DES-SN5YR/blob/main/5_COSMOLOGY/SN_only_cosmosis_likelihood.py
        """
        Computes modified likelihood computation to marginalize offset M
        from https://arxiv.org/abs/astro-ph/0104009v1, see Equation A9-12
        """
        delta = np.array([mu_model - mu])
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))
        return -0.5*chi2

    likelihood.init.params['Mb'].update(prior=None, derived='.best')
    chi2 = -2. * likelihood()
    chi2_ref = -2. * cov_log_likelihood(likelihood.flattheory, likelihood.flatdata, likelihood.precision)
    print(chi2 - chi2_ref)
    fn = '../../../../DES-SN5YR/5_COSMOLOGY/chains/fw0wacdm/fw0wacdm_SN_emcee.txt'
    #fn = '../../../../DES-SN5YR/5_COSMOLOGY/chains/flcdm/flcdm_SN_emcee.txt'
    if os.path.exists(fn):
        samples = np.loadtxt(fn, unpack=True)
        #logprior = - np.log(0.9 - 0.1)
        #samples = {name: samples[idx] for name, idx in zip(['Omega_m', 'logprior', 'logposterior'], [0, 1, 2])}
        samples = {name: samples[idx] for name, idx in zip(['Omega_m', 'w0_fld', 'wa_fld', 'logprior', 'logposterior'], [0, 1, 2, 3, 4])}
        samples['logposterior'] -= samples['logprior']  # logprior is for Omega_m, w0_fld, wa_fld
        print('min chi2', -2. * samples['logposterior'].max())
        for isample in range(10):
            values = {name: samples[name][isample] for name in samples if name not in ['logposterior', 'logprior']}
            chi2 = -2. * likelihood(values)
            chi2_ref = -2. * samples['logposterior'][isample]
            print(chi2, chi2_ref, chi2 - chi2_ref)



if __name__ == '__main__':

    setup_logging()
    #test_install()
    #test_profile()
    #test_union3(plot=True)
    test_desy5()
