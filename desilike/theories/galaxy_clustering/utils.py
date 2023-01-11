import numpy as np


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    if x.size == 0:
        return np.array(1.)
    if x.size == 1:
        return np.ones(x.size)
    if x.size == 2:
        return np.ones(x.size) / 2. * (x[1] - x[0])
    return np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.
