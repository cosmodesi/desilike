"""Collection of wrappers for commonly used optimizers."""
# TODO: implement other optimizers

from scipy.optimize import dual_annealing, minimize


def scipy_minimize(f, x_0, rng, **kwargs):
    """Optimize using :func:`scipy.minimize`.

    Parameters
    ----------
    f : callable
        Objective function.
    x_0 : array-like
        Starting point.
    rng : numpy.random.Generator
        Unused. Present for API consistency.
    **kwargs
        Additional keyword arguments passed to ``scipy.minimize``.

    Returns
    -------
    x_min : numpy.ndarray
        Coordinates of the minimum.
    f_min : float
        Value of the objective function at the minimum.
    success : bool
        Whether the optimizer finished successfully.

    """
    res = minimize(f, x_0, bounds=[(0, 1)] * len(x_0), **kwargs)
    return res.x, res.fun, res.success


def scipy_dual_annealing(f, x_0, rng, **kwargs):
    """Optimize using :func:`scipy.dual_annealing`.

    Parameters
    ----------
    f : callable
        Function to optimize.
    x_0 : array-like
        Starting point.
    rng : numpy.random.Generator
        Random number generator.
    **kwargs
        Additional keyword arguments passed to ``scipy.dual_annealing``.

    Returns
    -------
    x_min : numpy.ndarray
        Coordinates of the minimum.
    f_min : float
        Value of the objective function at the minimum.
    success : bool
        Whether the optimizer finished successfully.

    """
    kwargs = kwargs | dict(maxiter=1)
    res = dual_annealing(
        f, [(0, 1)] * len(x_0), x0=x_0, rng=rng, **kwargs)
    return res.x, res.fun, res.success
