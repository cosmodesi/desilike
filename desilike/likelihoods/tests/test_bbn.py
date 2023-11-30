import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.likelihoods.bbn import BBNOmegaBLikelihood


def test_install():
    likelihood = BBNOmegaBLikelihood()
    likelihood.params['bbn.loglikelihood'] = {}
    likelihood.params['bbn.logprior'] = {}
    installer = Installer(user=True)
    installer(likelihood)
    likelihood()
    assert np.allclose((likelihood + likelihood)(), 2. * likelihood() - likelihood.logprior)


if __name__ == '__main__':

    setup_logging()
    test_install()
