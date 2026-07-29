"""Collection of wrappers for commonly used optimizing functions."""
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
try:
    import optax
    OPTAX_INSTALLED = True
except ModuleNotFoundError:
    OPTAX_INSTALLED = False
import numpy as np
from scipy.optimize import dual_annealing as scipy_dual_annealing
from scipy.optimize import minimize


def optimize_bobyqa(f, x_0, rng, **kwargs):
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
    return soln.x, soln.f, soln.flag == soln.EXIT_SUCCESS


def optimize_dual_annealing(f, x_0, rng, **kwargs):
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
    kwargs['maxiter'] = 1
    res = scipy_dual_annealing(
        f, [(0, 1)] * len(x_0), x0=x_0, rng=rng, **kwargs)
    return res.x, res.fun, res.success


def optimize_minuit(f, x_0, rng):
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


def optimize_optax(f, x_0, rng, optimizer=None, n_steps=1000, **kwargs):
    """Optimize using an :mod:`optax` gradient-based optimizer.

    .. rubric:: References

    - `optax repo <https://github.com/google-deepmind/optax>`_
    - `optax docs <https://optax.readthedocs.io/>`_

    Parameters
    ----------
    f : callable
        Objective function.
    x_0 : array-like
        Starting point.
    rng : numpy.random.Generator
        Unused. Present for API consistency.
    optimizer : optax.GradientTransformation, default=None
        Optax optimizer. If ``None``, defaults to ``optax.adam(1e-3)``.
    n_steps : int, default=1000
        Number of gradient steps.
    **kwargs
        Additional keyword arguments passed to the default ``optax.adam``
        optimizer. Ignored if ``optimizer`` is provided.

    Returns
    -------
    x_min : numpy.ndarray
        Coordinates of the minimum.
    f_min : float
        Value of the objective function at the minimum.
    success : bool
        Whether the optimizer finished successfully. Always ``True`` if
        the final parameter values are finite.

    Raises
    ------
    ImportError
        If ``optax`` or ``jax`` are not installed.

    """
    for installed, name in zip([JAX_INSTALLED, OPTAX_INSTALLED],
                               ['jax', 'optax']):
        if not installed:
            msg = f"The '{name}' package is required but not installed."
            raise ImportError(msg)

    if optimizer is None:
        optimizer = optax.adam(1e-3, **kwargs)

    params = jax.numpy.array(x_0, dtype=float)
    opt_state = optimizer.init(params)
    grad_f = jax.jit(jax.grad(f))

    for _ in range(n_steps):
        grads = grad_f(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = jax.numpy.clip(params, 0.0, 1.0)

    x_min = np.array(params)
    f_min = float(f(x_min))
    success = np.all(np.isfinite(x_min))
    return x_min, f_min, success


def optimize_scipy(f, x_0, rng, **kwargs):
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
