import numpy as np

from desilike.samples import utils


def test_utils():
    assert np.allclose(utils.nsigmas_to_deltachi2(10.), 100.)


if __name__ == '__main__':

    test_utils()
