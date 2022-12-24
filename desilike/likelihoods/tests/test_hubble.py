import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.likelihoods.hubble import Riess2020H0Likelihood


def test_install():
    likelihood = Riess2020H0Likelihood()
    likelihood.params['riess2020.loglikelihood'] = {}
    likelihood.params['riess2020.logprior'] = {}
    installer = Installer(user=True)
    installer(likelihood)
    likelihood()
    assert np.allclose((likelihood + likelihood)(), 2. * likelihood() - likelihood.logprior)


if __name__ == '__main__':

    setup_logging()
    test_install()
