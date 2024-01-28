import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.theories import Cosmoprimo
from desilike.likelihoods.supernovae import PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood, Union3SNLikelihood


def test_install():
    ref = [-5649.588564438464, -7526.797408441289, -7600.468586494155, -18.973744770948603]
    cosmo = Cosmoprimo(fiducial='DESI')
    for ilike, Likelihood in enumerate([PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood, Union3SNLikelihood]):
        params = {'dM': -9.} if Likelihood is Union3SNLikelihood else {'Mb': -19.}
        likelihood = Likelihood(cosmo=cosmo)
        installer = Installer(user=True)
        installer(likelihood)
        likelihood(**params)
        print(likelihood.loglikelihood)
        assert np.allclose(likelihood.loglikelihood, ref[ilike])
        assert np.allclose((likelihood + likelihood)(**params), 2. * likelihood(**params) - likelihood.logprior)
        for param in params:
            likelihood.all_params[param].update(prior=None, derived='.prec')
        likelihood()
        print(likelihood.varied_params)


if __name__ == '__main__':

    setup_logging()
    test_install()
