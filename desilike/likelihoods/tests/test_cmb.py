import numpy as np

from desilike import setup_logging
from desilike.install import Installer
from desilike.likelihoods.cmb import (TTHighlPlanck2018ClikLikelihood, TTTEEEHighlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood,
                                      TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood)


def test_install():
    likelihood = TTHighlPlanck2018ClikLikelihood()
    installer = Installer(user=True)
    installer(likelihood)
    likelihood()
    assert np.allclose((likelihood + likelihood)(), 2. * likelihood())


if __name__ == '__main__':

    setup_logging()
    test_install()
