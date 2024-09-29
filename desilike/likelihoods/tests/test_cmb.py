import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.likelihoods.cmb import (BasePlanck2018GaussianLikelihood, FullGridPlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
                                      TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood,
                                      LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood, TTLowlPlanck2018Likelihood, EELowlPlanck2018Likelihood, TTTEEEHighlPlanck2018LiteLikelihood, TTHighlPlanck2018LiteLikelihood,
                                      HighlTTTEEEPlanck2020HillipopLikelihood, HighlTTPlanck2020HillipopLikelihood,
                                      LowlEEPlanck2020LollipopLikelihood, LowlEBPlanck2020LollipopLikelihood, LowlBBPlanck2020LollipopLikelihood, ACTDR6LensingLikelihood, read_planck2018_chain)
from desilike.theories import Cosmoprimo


def test_install():
    for Likelihood in (BasePlanck2018GaussianLikelihood, FullGridPlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
                       TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood,
                       LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood):
        if 'Unbinned' in Likelihood.__name__: continue
        if 'Lite' in Likelihood.__name__: continue
        print(Likelihood.__name__)
        likelihood = Likelihood()
        likelihood.init.params['planck.loglikelihood'] = {}
        likelihood.init.params['planck.logprior'] = {}
        installer = Installer(user=True)
        installer(likelihood)
        assert np.allclose((likelihood + likelihood)(), 2. * likelihood() - likelihood.logprior)


def test_clik():
    likelihood = TTTEEEHighlPlanck2018PlikLikelihood()
    likelihood()
    for param in likelihood.all_params.select(basename=['loglikelihood', 'logprior']):
        assert param.namespace
    likelihood = TTTEEEHighlPlanck2018PlikLikelihood()
    #print(likelihood(xi_sz_cib=0.), likelihood(xi_sz_cib=0.5))


def test_sum():
    from desilike.likelihoods import SumLikelihood
    likelihood = SumLikelihood([Likelihood() for Likelihood in [TTTEEEHighlPlanck2018PlikLiteLikelihood, LensingPlanck2018ClikLikelihood]])
    print(likelihood())


def test_gaussian_likelihood():

    params = ['Omega_m', 'Omega_cdm', 'A_s', 'H0']
    chain = read_planck2018_chain('base_w_wa_plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18', weights='cmb_only', params=params).sort(params)
    print(chain.names())
    assert chain.names()[:len(params)] == params
    print(chain.weight)

    likelihood = BasePlanck2018GaussianLikelihood(basename='base_plikHM_TE_lowE_BAO', weights='cmb_only')
    likelihood()
    chains = likelihood.fisher
    print(chains.to_stats(tablefmt='pretty'))

    likelihood = BasePlanck2018GaussianLikelihood(source='chains')
    likelihood()
    chains = likelihood.fisher
    print(chains.to_stats(tablefmt='pretty'))

    likelihood = BasePlanck2018GaussianLikelihood(source='covmat')
    likelihood()
    covmat = likelihood.fisher
    print(covmat.to_stats(tablefmt='pretty'))

    print(np.abs((chains._hessian - covmat._hessian) / covmat._hessian))
    from desilike.samples import plotting
    plotting.plot_triangle([chains, covmat], labels=['chains', 'covmat'], show=True)


def test_params():
    from desilike.likelihoods.cmb import TTTEEEHighlPlanck2018PlikLiteLikelihood

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


def test_error():
    from desilike import setup_logging
    from desilike.likelihoods.cmb import TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood
    from desilike.likelihoods import SumLikelihood

    setup_logging()
    cosmo = Cosmoprimo(engine='camb')
    likelihoods = [Likelihood(cosmo=cosmo) for Likelihood in [TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood]]
    likelihood_clik = SumLikelihood(likelihoods=likelihoods)

    assert not np.isfinite(likelihood_clik(h=0.71845617, omega_cdm=0.11316231, omega_b=0.02500884, logA=3.25690416, n_s=0.97226037, tau_reio=0.17722994, A_planck=0.9907607))
    assert likelihood_clik(**{param.name: param.value for param in likelihood_clik.varied_params}) > -np.inf


def test_emulator_direct():
    from desilike.likelihoods.cmb import TTHighlPlanck2018PlikLiteLikelihood
    from desilike.emulators import Emulator, TaylorEmulatorEngine
    from matplotlib import pyplot as plt

    size = 3
    #param_values = {'h': np.linspace(0.6, 0.8, size)}  #np.linspace(0.69, 0.7, size)}  # np.linspace(0.6, 0.8, size)}
    #param_values = {'omega_cdm': np.linspace(0.08, 0.2, size)}
    #param_values = {'omega_b': np.linspace(0.022, 0.025, size)} #np.linspace(0.01, 0.03, size)}
    #param_values = {'Omega_k': np.linspace(-0.05, 0.05, size)}  #np.linspace(-0.3, 0.3, size)}
    param_values = {'m_ncdm': np.linspace(0., 0.8, size)}
    #param_values = {'tau_reio': np.linspace(0.03, 0.08, size)}
    #param_values = {'logA': np.linspace(2.5, 3.5, size)}
    #param_values = {'n_s': np.linspace(0.8, 1.1, size)}
    #param_values = {'N_eff': np.linspace(2., 4., size)}
    #param_values = {'w0_fld': np.linspace(-1.4, -0.6, size)}  #np.linspace(-1.5, -0.5, size)}
    #param_values = {'wa_fld': np.linspace(-0.8, 0.8, size)}
    todo = ['lowe']

    Likelihoods = {'tt': TTHighlPlanck2018PlikLiteLikelihood, 'lowl': TTLowlPlanck2018ClikLikelihood, 'lowe': EELowlPlanck2018ClikLikelihood}

    for likelihood, Likelihood in Likelihoods.items():
        if likelihood in todo:
            cosmo = Cosmoprimo()
            for param in cosmo.init.params.select(fixed=False):
                param.update(fixed=True)
            for param in param_values:
                cosmo.init.params[param].update(fixed=False)
            likelihood = Likelihood(cosmo=cosmo)
            likelihood()
            theory = likelihood.theory

            emulator = Emulator(theory, engine=TaylorEmulatorEngine(order=4))
            emulator.set_samples()
            emulator.fit()
            emulated_theory = emulator.to_calculator()

            for param, values in param_values.items():
                cmap = plt.get_cmap('jet', len(values))
                ax = plt.gca()
                param = theory.all_params[param]
                ax.set_title(param.latex(inline=True))
                center = param.value
                for ivalue, value in enumerate(values):
                    theory(**{param.name: value})
                    emulated_theory(**{param.name: value})
                    color = cmap(ivalue / len(values))
                    print({cl: np.abs(emulated_theory.cls[cl][1:] / theory.cls[cl][1:] - 1).max() for cl in theory.cls})
                    for cl in theory.cls:
                        ells = np.arange(theory.cls[cl].size)
                        ax.plot(ells, ells * (ells + 1) * theory.cls[cl], color=color, linestyle='-')
                        ax.plot(ells, ells * (ells + 1) * emulated_theory.cls[cl], color=color, linestyle='--')
                theory(**{param.name: center})
                emulated_theory(**{param.name: center})
                plt.show()


def test_cmb():
    from desilike.likelihoods.cmb import TTHighlPlanck2018PlikLikelihood

    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    likelihood = TTHighlPlanck2018PlikLikelihood(cosmo=cosmo)
    likelihood()


def test_profile():
    from desilike.likelihoods.cmb import (TTTEEEHighlPlanck2018PlikLiteLikelihood,
                                          TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood)

    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    likelihood = TTTEEEHighlPlanck2018PlikLiteLikelihood(cosmo=cosmo) + TTLowlPlanck2018ClikLikelihood(cosmo=cosmo) + EELowlPlanck2018ClikLikelihood(cosmo=cosmo)
    likelihood()
    from desilike.profilers import MinuitProfiler
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize(niterations=5)
    print(profiles.bestfit)


def test_act_lensing():
    installer = Installer(user=True)
    installer(ACTDR6LensingLikelihood)

    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    likelihood = ACTDR6LensingLikelihood(cosmo=cosmo)
    likelihood()


def test_planck_python():

    installer = Installer(user=True)

    for Likelihood in [TTLowlPlanck2018Likelihood, EELowlPlanck2018Likelihood, TTTEEEHighlPlanck2018LiteLikelihood]:
        installer(Likelihood)
        cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
        likelihood = Likelihood(cosmo=cosmo)
        likelihood()

    for cl in ['TT', 'EE'][1:]:
        theory = {'camb': {'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'dark_energy_model': 'ppf', 'num_massive_neutrinos': 1}}}
        #theory = {'cosmoprimo.bindings.cobaya.cosmoprimo': {'engine': 'camb', 'stop_at_error': True, 'extra_args': {'N_ncdm': 1}}}
        params = {'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001}, 'proposal': 0.001, 'latex': '\\ln(10^{10} A_\\mathrm{s})', 'drop': True}, 'As': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'}, 'ns': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'}, 'H0': {'prior': {'min': 20, 'max': 100}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01}, 'latex': 'H_0'}, 'ombh2': {'prior': {'min': 0.005, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001}, 'proposal': 0.0001, 'latex': '\\Omega_\\mathrm{b} h^2'}, 'omch2': {'prior': {'min': 0.001, 'max': 0.99}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}, 'proposal': 0.0005, 'latex': '\\Omega_\\mathrm{c} h^2'}, 'tau': {'latex': '\\tau_\\mathrm{reio}', 'value': 0.0544}, 'mnu': {'latex': '\\sum m_\\nu', 'value': 0.06}, 'nnu': {'latex': 'N_\\mathrm{eff}', 'value': 3.044}}
        info = {'theory': theory, 'likelihood': {'planck_2018_lowl.{}'.format(cl): None}, 'params': params}
        from cobaya.model import get_model
        model = get_model(info)
        logpost = model.logposterior({'logA': 3.057147, 'ns': 0.9657119, 'H0': 70., 'ombh2': 0.02246306, 'omch2': 0.1184775, 'nnu': 3.044})
        print(logpost.loglike)

        cosmo = Cosmoprimo(engine='camb')
        likelihood = globals()[cl + 'LowlPlanck2018Likelihood'](cosmo=cosmo)
        params = {'logA': 3.057147, 'n_s': 0.9657119, 'h': 0.7, 'omega_b': 0.02246306, 'omega_cdm': 0.1184775, 'N_eff': 3.044}
        likelihood(params)
        print(likelihood.loglikelihood)

        likelihood_ref = globals()[cl + 'LowlPlanck2018ClikLikelihood'](cosmo=cosmo)
        likelihood_ref(params)
        assert np.allclose(likelihood.loglikelihood, likelihood_ref.loglikelihood, rtol=4e-5 if cl == 'EE' else 1e-6, atol=0.)

    theory = {'camb': {'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'dark_energy_model': 'ppf', 'num_massive_neutrinos': 1}}}
    params = {'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001}, 'proposal': 0.001, 'latex': '\\ln(10^{10} A_\\mathrm{s})', 'drop': True}, 'As': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'}, 'ns': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'}, 'H0': {'prior': {'min': 20, 'max': 100}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01}, 'latex': 'H_0'}, 'ombh2': {'prior': {'min': 0.005, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001}, 'proposal': 0.0001, 'latex': '\\Omega_\\mathrm{b} h^2'}, 'omch2': {'prior': {'min': 0.001, 'max': 0.99}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}, 'proposal': 0.0005, 'latex': '\\Omega_\\mathrm{c} h^2'}, 'tau': {'latex': '\\tau_\\mathrm{reio}', 'value': 0.0544}, 'mnu': {'latex': '\\sum m_\\nu', 'value': 0.06}, 'nnu': {'latex': 'N_\\mathrm{eff}', 'value': 3.044}}
    info = {'theory': theory, 'likelihood': {'planck_2018_highl_plik.TTTEEE_lite_native': None}, 'params': params}
    from cobaya.model import get_model
    model = get_model(info)
    logpost = model.logposterior({'logA': 3.057147, 'ns': 0.9657119, 'H0': 70., 'ombh2': 0.02246306, 'omch2': 0.1184775, 'nnu': 3.044, 'A_planck': 1.})
    print(logpost.loglike)

    for cl in ['TTTEEE', 'TT'][:1]:
        cosmo = Cosmoprimo(engine='camb')
        likelihood = globals()[cl + 'HighlPlanck2018LiteLikelihood'](cosmo=cosmo)
        params = {'logA': 3.057147, 'n_s': 0.9657119, 'h': 0.7, 'omega_b': 0.02246306, 'omega_cdm': 0.1184775, 'N_eff': 3.044}
        likelihood(params)
        print(likelihood.loglikelihood)
        likelihood_ref = globals()[cl + 'HighlPlanck2018PlikLiteLikelihood'](cosmo=cosmo)
        likelihood_ref(params)
        print(likelihood_ref.loglikelihood)
        assert np.allclose(likelihood.loglikelihood, likelihood_ref.loglikelihood)


def test_hillipop():

    #installer = Installer(user=True)
    #installer(HighlTTTEEEPlanck2020HillipopLikelihood)
    """
    theory = {'camb': {'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'dark_energy_model': 'ppf', 'num_massive_neutrinos': 1}}}
    params = {'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001}, 'proposal': 0.001, 'latex': '\\ln(10^{10} A_\\mathrm{s})', 'drop': True}, 'As': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'}, 'ns': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'}, 'H0': {'prior': {'min': 20, 'max': 100}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01}, 'latex': 'H_0'}, 'ombh2': {'prior': {'min': 0.005, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001}, 'proposal': 0.0001, 'latex': '\\Omega_\\mathrm{b} h^2'}, 'omch2': {'prior': {'min': 0.001, 'max': 0.99}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}, 'proposal': 0.0005, 'latex': '\\Omega_\\mathrm{c} h^2'}, 'tau': {'latex': '\\tau_\\mathrm{reio}', 'value': 0.0544}, 'mnu': {'latex': '\\sum m_\\nu', 'value': 0.06}, 'nnu': {'latex': 'N_\\mathrm{eff}', 'value': 3.044}}
    info = {'theory': theory, 'likelihood': {'planck_2020_hillipop.TTTEEE': None}, 'params': params}
    from cobaya.model import get_model
    model = get_model(info)
    logpost = model.logposterior({'logA': 3.057147, 'ns': 0.9657119, 'H0': 70., 'ombh2': 0.02246306, 'omch2': 0.1184775, 'nnu': 3.044, 'A_planck': 1.})
    print(logpost.loglike)
    """

    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    likelihood = HighlTTTEEEPlanck2020HillipopLikelihood(cosmo=cosmo)
    print(likelihood())
    likelihood = HighlTTPlanck2020HillipopLikelihood(cosmo=cosmo)
    print(likelihood())


def test_lollipop():

    #installer = Installer(user=True)
    #installer(LowlEEPlanck2020LollipopLikelihood)
    """
    theory = {'camb': {'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'dark_energy_model': 'ppf', 'num_massive_neutrinos': 1}}}
    params = {'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001}, 'proposal': 0.001, 'latex': '\\ln(10^{10} A_\\mathrm{s})', 'drop': True}, 'As': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'}, 'ns': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'}, 'H0': {'prior': {'min': 20, 'max': 100}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01}, 'latex': 'H_0'}, 'ombh2': {'prior': {'min': 0.005, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001}, 'proposal': 0.0001, 'latex': '\\Omega_\\mathrm{b} h^2'}, 'omch2': {'prior': {'min': 0.001, 'max': 0.99}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}, 'proposal': 0.0005, 'latex': '\\Omega_\\mathrm{c} h^2'}, 'tau': {'latex': '\\tau_\\mathrm{reio}', 'value': 0.0544}, 'mnu': {'latex': '\\sum m_\\nu', 'value': 0.06}, 'nnu': {'latex': 'N_\\mathrm{eff}', 'value': 3.044}}
    info = {'theory': theory, 'likelihood': {'planck_2020_lollipop.EE': None}, 'params': params}
    from cobaya.model import get_model
    model = get_model(info)
    logpost = model.logposterior({'logA': 3.057147, 'ns': 0.9657119, 'H0': 70., 'ombh2': 0.02246306, 'omch2': 0.1184775, 'nnu': 3.044, 'A_planck': 1.})
    print(logpost.loglike)
    """

    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    likelihood = LowlEEPlanck2020LollipopLikelihood(cosmo=cosmo)
    print(likelihood())
    likelihood = LowlEBPlanck2020LollipopLikelihood(cosmo=cosmo)
    print(likelihood())
    likelihood = LowlBBPlanck2020LollipopLikelihood(cosmo=cosmo)
    print(likelihood())



def test_jax():
    import jax
    for Likelihood in [ACTDR6LensingLikelihood, TTLowlPlanck2018Likelihood, EELowlPlanck2018Likelihood, TTTEEEHighlPlanck2018LiteLikelihood, HighlTTTEEEPlanck2020HillipopLikelihood, LowlEEPlanck2020LollipopLikelihood][-1:]:
        cosmo = Cosmoprimo(engine='capse')
        likelihood = Likelihood(cosmo=cosmo)
        params = {'logA': 3.057147, 'n_s': 0.9657119, 'h': 0.7, 'omega_b': 0.02246306, 'omega_cdm': 0.1184775, 'N_eff': 3.044}
        print(likelihood(params))
        grad = jax.grad(likelihood)
        print(grad(params))



if __name__ == '__main__':

    setup_logging()
    #test_install()
    #test_clik()
    #test_sum()
    #test_gaussian_likelihood()
    #test_params()
    #test_help()
    #test_copy()
    #test_error()
    #test_emulator_direct()
    #test_cmb()
    #test_profile()
    #test_act_lensing()
    #test_planck_python()
    #test_hillipop()
    test_lollipop()
    #test_jax()