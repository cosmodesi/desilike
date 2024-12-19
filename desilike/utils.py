"""A few utilities."""

import os
import sys
import time
import logging
import traceback
import warnings
import functools
import importlib
from collections import UserDict
import math

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from . import mpi, jax
from .jax import numpy as jnp
from .mpi import CurrentMPIComm


@CurrentMPIComm.enable
def exception_handler(exc_type, exc_value, exc_traceback, mpicomm=None):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '=' * 100
    # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')
    if mpicomm.size > 1:
        mpicomm.Abort()


def mkdir(dirname, **kwargs):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname, **kwargs)  # MPI...
    except OSError:
        return


class hybridmethod(object):
    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        # support use on abstract base classes
        self.__isabstractmethod__ = bool(
            getattr(fclass, '__isabstractmethod__', False)
        )

    def classmethod(self, fclass):
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
              # either bound to the class, or no instance method available
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)


def evaluate(value, type=None, locals=None):
    """
    Evaluate input value.

    Parameters
    ----------
    value : str, any type
        If value is string, call ``eval``, with input ``locals`` (dictionary of local objects).
        "np", "sp", "jnp", "jsp" are recognized as numpy, scipy, jax.numpy, jax.scipy (if jax is installed).

    type : type, default=None
        If not ``None``, cast output ``value`` with ``type``.

    locals : dict, default=None
        Dictionary of local objects to use when calling ``eval``.

    Returns
    -------
    value : evaluated value.
    """
    if isinstance(value, str):
        from .jax import numpy as jnp
        from .jax import scipy as jsp
        value = eval(value, {'np': np, 'sp': sp, 'jnp': jnp, 'jsp': jsp}, locals)
    if type is not None:
        value = type(value)
    return value


def rebin(array, new_shape, statistic=np.sum):
    """
    Bin an array in all axes based on the target shape, by summing or
    averaging. Number of output dimensions must match number of input dimensions and
    new axes must divide old ones.

    Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

    Example
    -------
    >>> m = np.arange(0, 100, 1).reshape((10, 10))
    >>> n = rebin(m, new_shape=(5, 5), statistic=np.sum)
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if array.ndim == 1 and np.ndim(new_shape) == 0:
        new_shape = [new_shape]
    if array.ndim != len(new_shape):
        raise ValueError('Input array dim is {}, but requested output one is {}'.format(array.ndim, len(new_shape)))

    pairs = []
    for d, c in zip(new_shape, array.shape):
        if c % d != 0:
            raise ValueError('New shape should divide current shape, but {:d} % {:d} = {:d}'.format(c, d, c % d))
        pairs.append((d, c // d))

    flattened = [ll for p in pairs for ll in p]
    array = array.reshape(flattened)

    for i in range(len(new_shape)):
        array = statistic(array, axis=-1 * (i + 1))

    return array


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Note
    ----
    You may find it useful to have different logging level depending on the process; e.g.
    ``setup_logging(level=(logging.INFO if mpicomm.rank == 0 else logging.ERROR))``
    will set INFO level on rank 0, and ERROR level on all other ranks of the MPI communicator ``mpicomm``.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    logging.getLogger("jax._src.lib.xla_bridge").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        @CurrentMPIComm.enable
        def format(self, record, mpicomm=None):
            ranksize = '[{:{dig}d}/{:d}]'.format(mpicomm.rank, mpicomm.size, dig=len(str(mpicomm.size)))
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ranksize + ' %(asctime)s %(name)-25s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler


class LoggingContext(object):
    """
    Class to locally update logging level:

    >>> with LoggingContext('warning') as log:
            ...
            # Logging level is warning
            logger = logging.getLogger('Logger')
            logger.info('This should not be printed')  # not logged
            logger.warning('This should be printed')  # logged
            ...
    """
    def __init__(self, level=None):
        self._level = logging.root.level
        if level is not None:
            if isinstance(level, str):
                level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
            logging.root.level = level
            for handler in logging.root.handlers:
                handler.setLevel(level)

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        logging.root.level = level = self._level
        for handler in logging.root.handlers:
            handler.setLevel(level)


class BaseMetaClass(type):

    """Metaclass to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            @CurrentMPIComm.enable
            def logger(cls, *args, rank=None, mpicomm=None, **kwargs):
                if rank is None or mpicomm.rank == rank:
                    getattr(cls.logger, level)(*args, **kwargs)

            return logger

        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(cls, 'log_{}'.format(level), make_logger(level))


def serialize_class(cls):
    """Serialize input class (such that it can be loaded from scratch) as module_name.ClassName."""
    clsname = '.'.join([cls.__module__, cls.__name__])
    return (clsname,)


def import_class(clsname, pythonpath=None, registry=None):
    """
    Import class from class name.

    Parameters
    ----------
    clsname : str, type
        Class name, as ``module_name.ClassName`` w.r.t. ``pythonpath``, or directly class type;
        in this case, other arguments are ignored.

    pythonpath : str, default=None
        Optionally, path where to find package/module where class is defined.

    registry : set, default=None
        Optionally, a set of class types to look into.
    """
    if isinstance(clsname, str):
        tmp = clsname.rsplit('.', 1)
        if len(tmp) == 1:
            clsname = tmp[0]
            if registry is None:
                try:
                    return globals()[clsname]
                except KeyError:
                    raise ImportError('Unknown class {}, provide e.g. pythonpath or module name as module_name.ClassName'.format(clsname))
            allcls = []
            for cls in registry:
                if cls.__name__ == clsname: allcls.append(cls)
            if len(allcls) == 1:
                cls = allcls[0]
            elif len(allcls) > 1:
                raise ImportError('Multiple classes are named {} in registry'.format(clsname))
            else:
                raise ImportError('No calculator class {} found in registry'.format(clsname))
        else:
            modname, clsname = tmp
            if pythonpath is not None:
                sys.path.insert(0, pythonpath)
            else:
                sys.path.append(os.path.dirname(__file__))
            module = importlib.import_module(modname)
            cls = getattr(module, clsname)
    else:
        cls = clsname
    return cls


class BaseClass(object, metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self, *args, **kwargs):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, *args, **kwargs):
        return self.__copy__(*args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        state = {'__class__': serialize_class(self.__class__), **self.__getstate__()}
        if getattr(self, 'mpicomm', None) is None or self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            mkdir(os.path.dirname(filename))
            np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename, fallback_class=None):
        filename = str(filename)
        state = np.load(filename, allow_pickle=True)
        if (cls is BaseClass or fallback_class is not None):
            is_npz = filename.endswith('.npz')
            if is_npz:
                cls = state['__class__'][()]
            else:
                cls = state[()].get('__class__', None)
            if cls is not None:
                try:
                    cls = import_class(*cls)
                except ImportError as exc:
                    if fallback_class is not None:
                        cls = fallback_class
                    else:
                        raise ImportError('Could not import file {} as {}'.format(filename, cls)) from exc
            if is_npz:
                return cls.load(filename)
        cls.log_info('Loading {}.'.format(filename))
        state = state[()]
        state.pop('__class__', None)
        new = cls.from_state(state)
        return new


def is_sequence(item):
    """Whether input item is a tuple or list."""
    return isinstance(item, (list, tuple, set, frozenset))


def is_path(item):
    """Whether input item is a path."""
    return isinstance(item, (str, os.PathLike))


def dict_to_yaml(d):
    """
    (Recursively) cast objects of input dictionary ``d`` to Python base types,
    such that they can be understood by the base yaml.
    """
    import numbers
    toret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_to_yaml(v)
        elif is_sequence(v):
            v = dict_to_yaml({i: vv for i, vv in enumerate(v)})
            v = [v[i] for i in range(len(v))]
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                v = v.item()
            else:
                v = v.tolist()
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif (v is not None) and not isinstance(v, (bool, numbers.Number)):
            v = str(v)
        toret[k] = v
    return toret


def deep_eq(obj1, obj2, equal_nan=True):
    from desilike.parameter import ParameterArray
    """(Recursively) test equality between ``obj1`` and ``obj2``."""
    if type(obj2) is type(obj1):
        if isinstance(obj1, dict):
            if obj2.keys() == obj1.keys():
                return all(deep_eq(obj1[name], obj2[name]) for name in obj1)
        elif isinstance(obj1, (tuple, list)):
            if len(obj2) == len(obj1):
                return all(deep_eq(o1, o2) for o1, o2 in zip(obj1, obj2))
        elif isinstance(obj1, (np.ndarray, ParameterArray) + jax.array_types):
            return np.array_equal(obj2, obj1, equal_nan=equal_nan)
        else:
            return obj2 == obj1
    return False


class NamespaceDict(BaseClass):
    """
    Dict-like type, that allows access to items as attributes, i.e. d.name is d[name].
    Should be removed soon (currently only used by :class:`ParameterConfig`.)
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], self.__class__):
                self.__dict__.update(args[0].__dict__)
            elif args[0] is not None:
                kwargs = {**args[0], **kwargs}
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        for name, value in kwargs.items():
            self[name] = value

    def get(self, *args, **kwargs):
        return getattr(self, *args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return getattr(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return setattr(self, *args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return delattr(self, *args, **kwargs)

    def __contains__(self, name):
        return name in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def setdefault(self, name, item):
        if name not in self:
            self[name] = item

    def update(self, *args, exclude=(), **kwargs):
        other = self.__class__(*args, **kwargs)
        for name, value in other.items():
            if name not in exclude:
                self[name] = value

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def __getstate__(self):
        return self.__dict__.copy()

    def pop(self, *args, **kwargs):
        return self.__dict__.pop(*args, **kwargs)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and deep_eq(other.__getstate__(), self.__getstate__())

    def __repr__(self):
        return str(self.__getstate__())


def _check_valid_inv(mat, invmat, rtol=1e-03, atol=1e-03, check_valid='raise'):
    """
    Check input array ``mat`` and ``invmat`` are matrix inverse within relative difference ``rtol`` and absolute difference ``atol``.
    If inversion is inaccurate, and ``check_valid`` is:

    - 'raise': raise a :class:`LinAlgError`
    - 'warn': issue a warning
    - 'ignore': ignore

    """
    if check_valid == 'ignore':
        return
    tmp = mat.dot(invmat)
    ref = np.eye(tmp.shape[0], dtype=tmp.dtype)
    if not np.allclose(tmp, ref, rtol=rtol, atol=atol):
        msg = 'Numerically inaccurate inverse matrix, max absolute diff {:.6f}.'.format(np.max(np.abs(tmp - ref)))
        if check_valid == 'raise':
            raise LinAlgError(msg)
        elif check_valid == 'warn':
            warnings.warn(msg)
        elif check_valid != 'ignore':
            raise ValueError('check_valid must be one of ["raise", "warn", "ignore"]')


def inv(mat, inv=np.linalg.inv, check_valid='raise'):
    """
    Return inverse of input 2D or 0D (scalar) array ``mat``.

    Parameters
    ----------
    mat : 2D array, scalar
        Input matrix to invert.

    inv : callable, default=np.linalg.inv
        Function that takes in 2D array and returns its inverse.

    check_valid : str, bool, default='raise'
        If inversion is inaccurate, and ``check_valid`` is:

        - 'raise': raise a :class:`LinAlgError`
        - 'warn': issue a warning
        - 'ignore': ignore

    Returns
    -------
    toret : 2D array, scalar
        Inverse of ``mat``.
    """
    mat = np.asarray(mat)
    if mat.ndim == 0:
        return 1. / mat
    toret = None
    try:
        toret = inv(mat)
    except LinAlgError as exc:
        if check_valid == 'raise':
            raise exc
        elif check_valid == 'warn':
            warnings.warn('Numerically inaccurate inverse matrix')
        elif check_valid != 'ignore':
            raise ValueError('check_valid must be one of ["raise", "warn", "ignore"]')

    _check_valid_inv(mat, toret, check_valid=check_valid)
    return toret


def blockinv(blocks, inv=np.linalg.inv, check_valid='raise'):
    """
    Return inverse of input ``blocks`` matrix.

    Parameters
    ----------
    blocks : list of list of arrays
        Input matrix to invert, in the form of blocks, e.g. ``[[A,B],[C,D]]``.

    inv : callable, default=np.linalg.inv
        Function that takes in 2D array and returns its inverse.

    check_valid : str, bool, default='raise'
        If inversion is inaccurate, and ``check_valid`` is:

        - 'raise': raise a :class:`LinAlgError`
        - 'warn': issue a warning
        - 'ignore': ignore

    Returns
    -------
    toret : 2D array
        Inverse of ``blocks`` matrix.
    """
    A = blocks[0][0]
    if (len(blocks), len(blocks[0])) == (1, 1):
        return inv(A)
    B = np.bmat(blocks[0][1:]).A
    C = np.bmat([b[0].T for b in blocks[1:]]).A.T
    invD = blockinv([b[1:] for b in blocks[1:]], inv=inv)

    def dot(*args):
        return np.linalg.multi_dot(args)

    invShur = inv(A - dot(B, invD, C))
    toret = np.bmat([[invShur, -dot(invShur, B, invD)], [-dot(invD, C, invShur), invD + dot(invD, C, invShur, B, invD)]]).A
    mat = np.bmat(blocks).A
    _check_valid_inv(mat, toret, check_valid=check_valid)
    return toret


def cov_to_corrcoef(cov):
    """
    Return correlation matrix corresponding to input covariance matrix ``cov``.
    If ``cov`` is scalar, return 1.
    """
    if np.ndim(cov) == 0:
        return 1.
    stddev = np.sqrt(np.diag(cov).real)
    c = cov / stddev[:, None] / stddev[None, :]
    return c


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    if x.size == 0:
        return np.array(1.)
    if x.size == 1:
        return np.ones(x.size)
    if x.size == 2:
        return np.ones(x.size) / 2. * (x[1] - x[0])
    return jnp.insert(x[2:] - x[:-2], jnp.array([0, len(x) - 1]), jnp.array([x[1] - x[0], x[-1] - x[-2]])) / 2.


def weights_leggauss(nx, sym=False):
    """Return weights for Gauss-Legendre integration."""
    x, wx = np.polynomial.legendre.leggauss((1 + sym) * nx)
    if sym:
        x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
    return x, wx


def weights_mu(mu, method='leggauss'):
    if method == 'trapz':
        if np.ndim(mu) == 0:
            mu = np.linspace(0., 1., mu)
        else:
            mu = jnp.asarray(mu)
        return mu, weights_trapz(mu) / (mu[-1] - mu[0])
    if method == 'leggauss':
        if np.ndim(mu) == 0:
            return weights_leggauss(mu, sym=True)
        raise ValueError('gauss integration does not array of mus')


def matrix_lininterp(xin, xout):
    # Matrix for linear interpolation
    toret = np.zeros((len(xin), len(xout)), dtype='f8')
    for iout, xout in enumerate(xout):
        iin = np.searchsorted(xin, xout, side='right') - 1
        if 0 <= iin < len(xin) - 1:
            frac = (xout - xin[iin]) / (xin[iin + 1] - xin[iin])
            toret[iin, iout] = 1. - frac
            toret[iin + 1, iout] = frac
        elif np.isclose(xout, xin[-1]):
            toret[iin, iout] = 1.
    return toret


def subspace(X, precision=None, npcs=None, chi2min=None, fweights=None, aweights=None):
    r"""
    Project input values ``X`` to a subspace.
    See https://arxiv.org/pdf/2009.03311.pdf

    Parameters
    ----------
    X : array
        Array of shape (number of samples, ndim).

    precision : array, default=None
        Optionally, precision matrix, to normalize ``X``.

    npcs : int, default=None
        Optionally, number (<= ndim) of principal components to keep.
        If ``None``, number of components to be kept is fixed by ``chi2min``.

    chi2min : int, default=None
        In case ``npcs`` is provided, threshold for the maximum difference in :math:`\chi^{2}`
        w.r.t. keeping all components. If ``None``, all components are kept.

    fweights : array, default=None
        Optionally, integer frequency weights, of shape (number of samples,).

    aweights : array, default=None
        Optionally, observation weights.

    Returns
    -------
    eigenvectors : array of shape (ndim, npcs)
        Eigenvectors.
    """
    X = np.asarray(X)
    X = X.reshape(X.shape[0], -1)
    if precision is None:
        L = np.array(1.)
    else:
        L = np.linalg.cholesky(precision)
    X = X.dot(L)
    cov = np.cov(X, rowvar=False, ddof=0, fweights=fweights, aweights=aweights)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if npcs is None:
        if chi2min is None:
            npcs = len(eigenvalues)
        else:
            npcs = len(eigenvalues) - np.sum(np.cumsum(eigenvalues) < chi2min)
    if npcs > len(eigenvectors):
        raise ValueError('Number of requested components is {0:d}, but dimension is {1:d} < {0:d}.'.format(npcs, len(eigenvalues)))
    return L.dot(eigenvectors)[..., -npcs:]


def txt_to_latex(txt):
    """Transform standard text into latex by replacing '_xxx' with '_{xxx}' and '^xxx' with '^{xxx}'."""
    latex = ''
    txt = list(txt)
    for c in txt:
        latex += c
        if c in ['_', '^']:
            latex += '{'
            txt += '}'
    return latex


def outputs_to_latex(name):
    """
    Turn ``name`` to latex string,
    with defaults for 'loglikelihood', 'logposterior' and 'logprior'.
    """
    toret = txt_to_latex(name)
    for full, symbol in [('loglikelihood', 'L'), ('logposterior', '\\mathcal{L}'), ('logprior', 'p')]:
        toret = toret.replace(full, symbol)
    return toret


class Monitor(BaseClass):
    """
    Class to monitor execution time (optionally, memory):

    >>> with Monitor() as mem:
            ...
            mem.stop()  # stop monitoring
            ...
            mem.start() # restart monitoring
            ...
            dt = mem.get('time')  # elapsed time
            mem.reset()  # reset, i.e. forget about previous monitoring and start

    """
    def __init__(self, quantities='time'):
        """
        Initialize monitor.

        Parameters
        ----------
        quantities : str, default='time'
            Quantities to monitor: 'time', 'mem' (requires package psutil to be installed).
        """
        if not is_sequence(quantities):
            quantities = (quantities,)
        self.quantities = list(quantities)
        self.reset()

    def time(self):
        return time.time()

    @property
    def proc(self):
        if getattr(self, '_proc', None) is None:
            import psutil
            self._proc = psutil.Process(os.getpid())
        return self._proc

    def mem(self):
        return self.proc.memory_info().rss / 1e6

    def start(self):
        """Start monitoring."""
        self._start = {quantity: getattr(self, quantity)() for quantity in self.quantities}

    def stop(self):
        """Stop monitoring."""
        stop = {quantity: getattr(self, quantity)() for quantity in self.quantities}
        self._counter += 1
        self._diffs = {quantity: stop[quantity] - self._start[quantity] + diff for quantity, diff in self._diffs.items()}
        self._start = stop

    @property
    def counter(self):
        """Number of data points."""
        return self._counter

    def get(self, quantity, average=True):
        """Return time series of quantity ('time' or 'mem'), or, if ``average`` is ``True``, its average."""
        if average:
            if self._counter == 0:
                return np.nan
            return self._diffs[quantity] / self._counter
        return self._diffs[quantity]

    def reset(self):
        """Reset, i.e. forget about previous monitoring and start."""
        self._diffs = {quantity: 0. for quantity in self.quantities}
        self._counter = 0
        self.start()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""


def expand_dict(di, names):
    """
    Expand input dictionary, taking care of wildcards, e.g.:

    >>> expand_dict({'*': 2}, ['a', 'b'])
    {'a': 2, 'b': 2}
    >>> expand_dict({'a*': 2, 'b': 1}, ['a1', 'a2', 'b'])
    {'a1': 2, 'a2': 2, 'b': 1}
    """
    toret = dict.fromkeys(names)
    if is_sequence(di):
        di = dict(zip(names, di))
    if not hasattr(di, 'items'):
        di = {'*': di}
    from .parameter import find_names
    for template, value in di.items():
        for tmpname in find_names(names, template):
            toret[tmpname] = value
    return toret


def std_notation(value, sigfigs, positive_sign=False):
    """
    Standard notation (US version).
    Return a string corresponding to value with the number of significant digits ``sigfigs``.

    >>> std_notation(5, 2)
    '5.0'
    >>> std_notation(5.36, 2)
    '5.4'
    >>> std_notation(5360, 2)
    '5400'
    >>> std_notation(0.05363, 3)
    '0.0536'

    Created by William Rusnack:
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    return ('-' if is_neg else '+' if positive_sign else '') + _place_dot(sig_digits, power)


def sci_notation(value, sigfigs, filler='e', positive_sign=False):
    """
    Scientific notation.

    Return a string corresponding to value with the number of significant digits ``sigfigs``,
    with 10s exponent filler ``filler`` placed between the decimal value and 10s exponent.

    >>> sci_notation(123, 1, 'e')
    '1e2'
    >>> sci_notation(123, 3, 'e')
    '1.23e2'
    >>> sci_notation(0.126, 2, 'e')
    '1.3e-1'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    dot_power = min(-(sigfigs - 1), 0)
    ten_power = power + sigfigs - 1
    return ('-' if is_neg else '+' if positive_sign else '') + _place_dot(sig_digits, dot_power) + filler + str(ten_power)


def _place_dot(digits, power):
    """
    Place dot in the correct spot, given by integer ``power`` (starting from the right of ``digits``)
    in the string ``digits``.
    If the dot is outside the range of the digits zeros will be added.

    >>> _place_dot('123', 2)
    '12300'
    >>> _place_dot('123', -2)
    '1.23'
    >>> _place_dot('123', 3)
    '0.123'
    >>> _place_dot('123', 5)
    '0.00123'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    if power > 0: out = digits + '0' * power

    elif power < 0:
        power = abs(power)
        sigfigs = len(digits)

        if power < sigfigs:
            out = digits[:-power] + '.' + digits[-power:]

        else:
            out = '0.' + '0' * (power - sigfigs) + digits

    else:
        out = digits + ('.' if digits[-1] == '0' else '')

    return out


def _number_profile(value, sigfigs):
    """
    Return elements to turn number into string representation.

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Parameters
    ----------
    value : float
        Number.

    sigfigs : int
        Number of significant digits.

    Returns
    -------
    sig_digits : string
        Significant digits.

    power : int
        10s exponent to get the dot to the proper location in the significant digits

    is_neg : bool
        ``True`` if value is < 0 else ``False``
    """
    if value == 0:
        sig_digits = '0' * sigfigs
        power = -(1 - sigfigs)
        is_neg = False

    else:
        is_neg = value < 0
        if is_neg: value = abs(value)

        power = -1 * math.floor(math.log10(value)) + sigfigs - 1
        sig_digits = str(int(round(abs(value) * 10.0**power)))

    return sig_digits, int(-power), is_neg


def round_measurement(x, u=0.1, v=None, sigfigs=2, positive_sign=False, notation='auto'):
    """
    Return string representation of input central value ``x`` with uncertainties ``u`` and ``v``.

    Parameters
    ----------
    x : float
        Central value.

    u : float, default=0.1
        Upper uncertainty on ``x`` (positive).

    v : float, default=None
        Lower uncertainty on ``v`` (negative).
        If ``None``, only returns string representation for ``x`` and ``u``.

    sigfigs : int, default=2
        Number of digits to keep for the uncertainties (hence fixing number of digits for ``x``).

    Returns
    -------
    xr : str
        String representation for central value ``x``.

    ur : str
        String representation for upper uncertainty ``u``.

    vr : str
        If ``v`` is not ``None``, string representation for lower uncertainty ``v``.
    """
    x, u = float(x), float(u)
    return_v = True
    if v is None:
        return_v = False
        v = -abs(u)
    else:
        v = float(v)
    if x == 0. or not np.isfinite(x): logx = 0
    else: logx = math.floor(math.log10(abs(x)))
    if u == 0. or not np.isfinite(u): logu = logx
    else: logu = math.floor(math.log10(abs(u)))
    if v == 0. or not np.isfinite(v): logv = logx
    else: logv = math.floor(math.log10(abs(v)))
    if x == 0.: logx = max(logu, logv)

    def round_notation(val, sigfigs, notation='auto', positive_sign=False):
        if not np.isfinite(val):
            return str(val)
        if notation == 'auto':
            # if 1e-3 < abs(val) < 1e3 or center and (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
            if (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
                notation = 'std'
            else:
                notation = 'sci'
        notation_dict = {'std': std_notation, 'sci': sci_notation}

        if notation in notation_dict:
            return notation_dict[notation](val, sigfigs=sigfigs, positive_sign=positive_sign)
        return notation(val, sigfigs=sigfigs, positive_sign=positive_sign)

    if logv > logu:
        sigfigs = (logx - logu + sigfigs, sigfigs, logv - logu + sigfigs)
    else:
        sigfigs = (logx - logv + sigfigs, logu - logv + sigfigs, sigfigs)

    xr = round_notation(x, sigfigs=sigfigs[0], notation=notation, positive_sign=bool(positive_sign) and positive_sign != 'u')
    ur = round_notation(u, sigfigs=sigfigs[1], notation=notation, positive_sign=bool(positive_sign))
    vr = round_notation(v, sigfigs=sigfigs[2], notation=notation, positive_sign=bool(positive_sign))

    if return_v: return xr, ur, vr
    return xr, ur


class TaskManager(BaseClass):
    """
    A MPI task manager that distributes tasks over a set of MPI processes,
    using a specified number of independent workers to compute each task.

    Given the specified number of independent workers (which compute
    tasks in parallel), the total number of available CPUs will be
    divided evenly.

    The main function is ``iterate`` which iterates through a set of tasks,
    distributing the tasks in parallel over the available ranks.

    Tasks are assigned deterministically, there is no manager process.
    """
    @CurrentMPIComm.enable
    def __init__(self, nprocs_per_task=1, use_all_nprocs=True, mpicomm=None):
        """
        Initialize :class:`TaskManager`.

        Parameters
        ----------
        nprocs_per_task : int, default=1
            The desired number of processes assigned to compute each task.

        use_all_nprocs : bool, default=True
            If `True`, use all available processes, including the remainder
            if `nprocs_per_task` does not divide the total number of processes
            evenly; default is `False`.

        mpicomm : MPI communicator, default=None
            The global communicator that will be split so each worker
            has a subset of processes available; default is COMM_WORLD.
        """
        self.basecomm = mpicomm
        self.worker = -1
        if nprocs_per_task > self.basecomm.size:
            raise ValueError('cannot attribute {:d} processes per task given {:d} total processes'.format(nprocs_per_task, self.basecomm.size))
        if use_all_nprocs:
            for isplit, split in enumerate(np.array_split(np.arange(self.basecomm.size), max(self.basecomm.size // nprocs_per_task, 1))):
                if split[0] <= self.basecomm.rank <= split[-1]:
                    self.worker = isplit
                    self.self_worker_ranks = list(split)
        else:
            for isplit in range(max(self.basecomm.size // nprocs_per_task, 1)):
                low, up = isplit * nprocs_per_task, (isplit + 1) * nprocs_per_task
                if low <= self.basecomm.rank < up:
                    self.worker = isplit
                    self.self_worker_ranks = list(range(low, up))
        self.nworkers = isplit + 1
        # split the comm between the workers
        self.mpicomm = self.basecomm.Split(self.worker, 0)
        self.rootcomm = self.basecomm.Split(int(self.basecomm.rank == self.self_worker_ranks[0]), 0)
        if self.mpicomm.rank == 0:
            self.log_debug('Entering {} with {:d} workers.'.format(self.__class__.__name__, self.nworkers))

    @property
    def size(self):
        return self.nworkers

    def __enter__(self):
        """Enter task manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit gracefully by closing and freeing the MPI-related variables."""

        if exc_value is not None:
            exception_handler(exc_type, exc_value, exc_traceback)

        # wait and exit
        self.log_debug('Rank {:d} process finished'.format(self.basecomm.rank))
        self.basecomm.Barrier()

        if self.mpicomm.rank == 0:
            self.log_debug('Root is finished; terminating')

        if self.mpicomm is not None:
            self.mpicomm.Free()

    def iterate(self, tasks):
        """
        Iterate through a series of tasks in parallel.

        Notes
        -----
        This is a collective operation and should be called by all ranks.

        Parameters
        ----------
        tasks : iterable
            An iterable of `task` items that will be yielded in parallel across all ranks.

        Returns
        -------
        tasks : list
            The individual items of `tasks`, iterated through in parallel.
        """
        tasks = list(tasks)
        size = len(tasks)
        start, stop = self.worker * size // self.nworkers, (self.worker + 1) * size // self.nworkers
        return tasks[start:stop]

    def reduce(self, li, root=0):
        """Reduce to root."""
        return self.rootcomm.reduce(li, root=root)

    def allreduce(self, li):
        """Reduce to all ranks."""
        return self.basecomm.bcast(self.reduce(li), root=0)

    def map(self, func, tasks):
        """
        Apply a function to all of the values in a list and return the list of results.

        If ``tasks`` contains tuples, the arguments are passed to
        ``function`` using the ``*args`` syntax.

        Notes
        -----
        This is a collective operation and should be called by
        all ranks.

        Parameters
        ----------
        function : callable
            The function to apply to the list.

        tasks : list
            The list of tasks.

        Returns
        -------
        results : list
            The list of the return values of ``function``.
        """
        results = map(func, self.iterate(tasks))
        return np.asarray(self.allreduce(results))