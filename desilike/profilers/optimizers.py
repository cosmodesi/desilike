"""Collection of wrappers for commonly used optimizers."""
# TODO: implement other optimizers
try:
    import pybobyqa
    BOBYQA_INSTALLED = True
except ModuleNotFoundError:
    BOBYQA_INSTALLED = False
try:
    from iminuit import Minuit
    MINUIT_INSTALLED = True
except ModuleNotFoundError:
    MINUIT_INSTALLED = False
try:
    import jax
    JAX_INSTALLED = True
except ModuleNotFoundError:
    JAX_INSTALLED = False
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import dual_annealing as scipy_dual_annealing


def bobyqa(f, x_0, rng, **kwargs):
    """Optimize using :meth:`pybobyqa.solve`.

    .. rubric:: References

    - `pybobyqa repo <https://github.com/numericalalgorithmsgroup/pybobyqa>`_
    - `pybobyqa docs <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_
    - `bobyqa paper A <https://doi.org/10.1145/3338517>`_
    - `bobyqa paper B <https://doi.org/10.1080/02331934.2021.1883015>`_

    Parameters
    ----------
    f : callable
        Objective function.
    x_0 : array-like
        Starting point.
    rng : numpy.random.Generator
        Unused. Present for API consistency.
    **kwargs
        Additional keyword arguments passed to ``pybobyqa.solve``.

    Returns
    -------
    x_min : numpy.ndarray
        Coordinates of the minimum.
    f_min : float
        Value of the objective function at the minimum.
    success : bool
        Whether the optimizer finished successfully.

    Raises
    ------
    ImportError
        If ``Py-BOBYQA`` is not installed.

    """
    if not BOBYQA_INSTALLED:
        msg = "The 'Py-BOBYQA' package is required but not installed."
        raise ImportError(msg)

    soln = pybobyqa.solve(
        f, x_0, bounds=(np.zeros(len(x_0)), np.ones(len(x_0))), **kwargs)
    return soln.x, soln.f, soln.flag == 0


def dual_annealing(f, x_0, rng, **kwargs):
    """Optimize using :func:`scipy.dual_annealing`.

    Parameters
    ----------
    f : callable
        Objective function.
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
    res = scipy_dual_annealing(
        f, [(0, 1)] * len(x_0), x0=x_0, rng=rng, **kwargs)
    return res.x, res.fun, res.success


def minuit(f, x_0, rng):
    """Optimize using :meth:`iminuit.Minuit.migrad`.

    .. rubric:: References

    - `minuit repo <https://github.com/scikit-hep/iminuit>`_
    - `minuit docs <https://scikit-hep.org/iminuit/>`_
    - `minuit paper <https://doi.org/10.1016/0010-4655(75)90039-9>`_

    Parameters
    ----------
    f : callable
        Objective function.
    x_0 : array-like
        Starting point.
    rng : numpy.random.Generator
        Random number generator. Ignored and only used for API consistency.

    Returns
    -------
    x_min : numpy.ndarray
        Coordinates of the minimum.
    f_min : float
        Value of the objective function at the minimum.
    success : bool
        Whether the optimizer finished successfully.

    Raises
    ------
    ImportError
        If ``iminuit`` is not installed.

    """
    if not MINUIT_INSTALLED:
        msg = "The 'iminuit' package is required but not installed."
        raise ImportError(msg)

    n_dim = len(x_0)

    try:
        assert JAX_INSTALLED
        grad = jax.grad(f)
    except (AssertionError, jax.errors.TracerArrayConversionError):
        grad = None
    m = Minuit(f, x_0, grad=grad)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    return np.array([m.values[f'x{i}'] for i in range(n_dim)]), m.fval, m.valid


def scipy(f, x_0, rng, **kwargs):
    """Optimize using :func:`scipy.minimize`.

    - `scipy repo <https://github.com/scipy/scipy>`_
    - `scipy docs <https://docs.scipy.org/doc/scipy/index.html>`_
    - `scipy paper <https://doi.org/10.1038/s41592-019-0686-2>`_

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
