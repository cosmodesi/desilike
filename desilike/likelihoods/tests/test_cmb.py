import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.likelihoods.cmb import (Planck2018GaussianLikelihood, TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
                                      TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood,
                                      LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood)


def test_install():
    for Likelihood in (Planck2018GaussianLikelihood, TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
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


if __name__ == '__main__':

    setup_logging()
    test_install()
