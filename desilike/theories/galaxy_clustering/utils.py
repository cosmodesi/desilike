import numpy as np


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.
