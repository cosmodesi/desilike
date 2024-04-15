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


if __name__ == '__main__':

    setup_logging()
    test_install()
    #test_profile()
