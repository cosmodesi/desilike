import functools
import logging

logging.getLogger('jax._src.lib.xla_bridge').addFilter(logging.Filter('No GPU/TPU found, falling back to CPU.'))


# jax array types
array_types = ()

try:
    # raise ImportError
    import jax, jaxlib
    from jax.config import config; config.update('jax_enable_x64', True)
    from jax import numpy, scipy
    array_types = []
    for line in ['jaxlib.xla_extension.DeviceArrayBase', 'type(numpy.array(0))', 'jax.core.Tracer']:
        try:
            array_types.append(eval(line))
        except AttributeError:
            pass
    array_types = tuple(array_types)
except ImportError:
    jax = None
    import numpy
    import scipy


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
        return getattr(self.dist, func.__name__, getattr(self.odist, func.__name__))(*args, **kwargs)

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