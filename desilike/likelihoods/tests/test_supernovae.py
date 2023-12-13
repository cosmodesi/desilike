import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.theories import Cosmoprimo
from desilike.likelihoods.supernovae import PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood,Union3SNLikelihood

def test_install():
    ref = [-5649.588564438464, -7526.797408441289, -7600.468586494155]
    cosmo = Cosmoprimo(fiducial='DESI')
    Mb = -19.
    for ilike, Likelihood in enumerate([PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood]):
        likelihood = Likelihood(cosmo=cosmo)
        installer = Installer(user=True)
        installer(likelihood)
        likelihood(Mb=Mb)
        assert np.allclose(likelihood.loglikelihood, ref[ilike])
        assert np.allclose((likelihood + likelihood)(Mb=Mb), 2. * likelihood(Mb=Mb) - likelihood.logprior)

def test_install_Union3():
    cosmo = Cosmoprimo(engine='camb')
    installer = Installer(user=True)
    likelihood=Union3SNLikelihood(cosmo=cosmo)
    installer(likelihood)

if __name__ == '__main__':

    setup_logging()
    test_install()
    test_install_Union3()
