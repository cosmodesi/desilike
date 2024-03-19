import functools
import logging

logging.getLogger('jax._src.lib.xla_bridge').addFilter(logging.Filter('No GPU/TPU found, falling back to CPU.'))


# jax array types
array_types = ()
interpax = None

try:
    # raise ImportError
    import jax, jaxlib
    from jax import config
    config.update('jax_enable_x64', True)
    from jax import numpy, scipy
    from jax.tree_util import register_pytree_node_class
    array_types = []
    for line in ['jaxlib.xla_extension.DeviceArrayBase', 'type(numpy.array(0))', 'jax.core.Tracer']:
        try:
            array_types.append(eval(line))
        except AttributeError:
            pass
    array_types = tuple(array_types)
    import interpax
except ImportError:
    jax = None
    import numpy
    import scipy
    def register_pytree_node_class(cls):
        return cls


def jit(*args, **kwargs):
    """Return :mod:`jax` just-in-time compiler."""

    def get_wrapper(func):
        if jax is None:
            return func
        return jax.jit(func, **kwargs)

    if kwargs or not args:
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))

    return get_wrapper(args[0])


def use_jax(array):
    """Whether to use jax.numpy depending on whether array is jax's object."""
    return isinstance(array, tuple(array_types))


def to_nparray(array):
    """Convert to numpy array if possible."""
    import numpy as _np
    try:
        return _np.asarray(array)
    except jax.errors.TracerArrayConversionError:
        return


def dist_name(dist):
    """
    Return distribution name, which should work with either scipy (where dist is a :class:`rv_continuous` instance)
    or jax implementation (where dist is a module).
    """
    name = getattr(dist, 'name', None)
    if name is None: name = dist.__name__.split('.')[-1]
    return name


def fallback(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        args, kwargs = func(self, *args, **kwargs)
        ofunc = getattr(self.odist, func.__name__)
        #print(any(use_jax(arg) for arg in args), func)
        if not any(use_jax(arg) for arg in args):
            return ofunc(*args, **kwargs)
        return getattr(self.dist, func.__name__, ofunc)(*args, **kwargs)

    return wrapper


class rv_frozen(object):
    """
    ``jax`` currently does not implement scipy's frozen random variate.
    Here is an ersatz.
    """
    def __init__(self, dist, *args, **kwds):
        self.dist = dist
        from scipy import stats
        self.odist = getattr(stats, dist_name(dist))
        self.args = args
        self.kwds = kwds

    @fallback
    def pdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def logpdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def cdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def logcdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def ppf(self, q):
        return (q,) + self.args, self.kwds

    @fallback
    def isf(self, q):
        return (q,) + self.args, self.kwds

    @fallback
    def rvs(self, size=None, random_state=None):
        return self.args, {**self.kwds, 'size': size, 'random_state': random_state}

    @fallback
    def sf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def logsf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def stats(self, moments='mv'):
        return self.args, {**self.kwds, 'moments': moments}

    @fallback
    def median(self):
        return self.args, self.kwds

    @fallback
    def mean(self):
        return self.args, self.kwds

    @fallback
    def var(self):
        return self.args, self.kwds

    @fallback
    def std(self):
        return self.args, self.kwds

    @fallback
    def moment(self, order=None, **kwds):
        return (order,) + self.args, {**self.kwds, **kwds}

    @fallback
    def entropy(self):
        return self.args, self.kwds

    @fallback
    def interval(self, confidence=None, **kwds):
        return (confidence,) + self.args, {**self.kwds, **kwds}

    @fallback
    def support(self):
        return self.args, self.kwds


def interp1d(xq, x, f, method='cubic'):
    """
    Interpolate a 1d function.

    Note
    ----
    Using interpax: https://github.com/f0uriest/interpax

    Parameters
    ----------
    xq : ndarray, shape(Nq,)
        query points where interpolation is desired
    x : ndarray, shape(Nx,)
        coordinates of known function values ("knots")
    f : ndarray, shape(Nx,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints

    derivative : int >= 0
        derivative order to calculate
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as a 2 element array or tuple to specify different conditions
        for xq<x[0] and x[-1]<xq
    period : float > 0, None
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. None denotes no periodicity

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points
    """
    method = {1: 'linear', 3: 'cubic'}.get(method, method)
    if interpax is not None:
        shape = xq.shape
        return interpax.interp1d(xq.reshape(-1), x, f, method=method).reshape(shape + f.shape[1:])

    from scipy import interpolate
    return interpolate.interp1d(x, f, kind=method, fill_value='extrapolate', axis=0)(xq)