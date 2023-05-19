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

    def pdf(self, x):
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.odist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.odist.stats(*self.args, **kwds)

    def median(self):
        return self.odist.median(*self.args, **self.kwds)

    def mean(self):
        return self.odist.mean(*self.args, **self.kwds)

    def var(self):
        return self.odist.var(*self.args, **self.kwds)

    def std(self):
        return self.odist.std(*self.args, **self.kwds)

    def moment(self, order=None, **kwds):
        return self.odist.moment(order, *self.args, **self.kwds, **kwds)

    def entropy(self):
        return self.odist.entropy(*self.args, **self.kwds)

    def interval(self, confidence=None, **kwds):
        return self.odist.interval(confidence, *self.args, **self.kwds, **kwds)

    def support(self):
        return self.odist.support(*self.args, **self.kwds)