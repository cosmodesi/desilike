"""A few utilities."""

import os
import sys
import time
import logging
import traceback
import warnings
import importlib

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from .mpi import CurrentMPIComm

try:
    # raise ImportError
    import jax, jaxlib
    from jax.config import config; config.update('jax_enable_x64', True)
    import jax.numpy as jnp
except ImportError:
    jax = None
    import numpy as jnp

try:
    from collections.abc import MutableSet  # >= 3.10
except ImportError:
    from collections import MutableSet
from collections import OrderedDict


def use_jax(array):
    """Whether to use jax.numpy depending on whether array is jax object"""
    return jax and isinstance(array, (jaxlib.xla_extension.DeviceArrayBase, jax.core.Tracer))


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


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def evaluate(value, type=None, locals=None):
    if isinstance(value, str):
        value = eval(value, {'np': np, 'sp': sp}, locals)
    if type is not None:
        value = type(value)
    return value


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

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
    clsname = '.'.join([cls.__module__, cls.__name__])
    return (clsname,)


def import_class(clsname, pythonpath=None, registry=None, install=None):
    """
    Import class from class name.

    Parameters
    ----------
    clsname : string, type
        Class name, as ``module.ClassName`` w.r.t. ``pythonpath``, or directly class type;
        in this case, other arguments are ignored.

    pythonpath : string, default=None
        Optionally, path where to find package/module where class is defined.

    registry : set, default=None
        Optionally, a set of class types to look into.
    """
    from .parameter import find_names
    if isinstance(clsname, str):
        if install is not None:
            exclude = install.get('exclude', [])
            if find_names(clsname, exclude): install = None
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
    if install is not None and hasattr(cls, 'install'):
        from .install import InstallerConfig
        install = InstallerConfig(install)
        if not find_names(serialize_class(cls)[0], install.get('exclude', [])):
            cls.install(install)
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
        self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, {'__class__': serialize_class(self.__class__), **self.__getstate__()}, allow_pickle=True)

    @classmethod
    def load(cls, filename, fallback_class=None):
        state = np.load(filename, allow_pickle=True)[()]
        if (cls is BaseClass or fallback_class is not None) and '__class__' in state:
            cls = state['__class__']
            try:
                cls = import_class(*cls)
            except ImportError as exc:
                if fallback_class is not None:
                    cls = fallback_class
                else:
                    raise ImportError('Could not import file {} as {}'.format(filename, cls)) from exc
        cls.log_info('Loading {}.'.format(filename))
        state.pop('__class__', None)
        new = cls.from_state(state)
        return new


def is_sequence(item):
    """Whether input item is a tuple or list."""
    return isinstance(item, (list, tuple, set, OrderedSet))


def dict_to_yaml(d):
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
        elif not isinstance(v, (bool, numbers.Number)):
            v = str(v)
        toret[k] = v
    return toret


def deep_eq(obj1, obj2):
    if type(obj2) is type(obj1):
        if isinstance(obj1, dict):
            if obj2.keys() == obj1.keys():
                return all(deep_eq(obj1[name], obj2[name]) for name in obj1)
        elif isinstance(obj1, (tuple, list)):
            if len(obj2) == len(obj1):
                return all(deep_eq(o1, o2) for o1, o2 in zip(obj1, obj2))
        elif isinstance(obj1, (np.ndarray, jnp.ndarray)):
            return np.all(obj2 == obj1)
        else:
            return obj2 == obj1
    return False


class OrderedSet(OrderedDict, MutableSet):

    """Adapted from https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set"""

    def __init__(self, *args):
        if not args:
            return
        if len(args) > 1:
            args = [args]
        for elem in OrderedDict.fromkeys(*args):
            self.add(elem)

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)


class NamespaceDict(BaseClass):

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


def _check_valid_inv(mat, invmat, rtol=1e-04, atol=1e-05, check_valid='raise'):
    """
    Check input array ``mat`` and ``invmat`` are matrix inverse.
    Raise :class:`LinAlgError` if input product of input arrays ``mat`` and ``invmat`` is not close to identity
    within relative difference ``rtol`` and absolute difference ``atol``.
    """
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

    check_valid : bool, default=True
        If inversion inaccurate, raise a :class:`LinAlgError` (see :func:`_check_valid_inv`).

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

    check_valid : bool, default=True
        If inversion inaccurate, raise a :class:`LinAlgError` (see :func:`_check_valid_inv`).

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


def subspace(X, precision=None, npcs=None, chi2min=None, **kwargs):
    # See https://arxiv.org/pdf/2009.03311.pdf
    X = np.asarray(X)
    X = X.reshape(X.shape[0], -1)
    if precision is None:
        L = np.array(1.)
    else:
        L = np.linalg.cholesky(precision)
    X = X.dot(L)
    cov = np.cov(X, rowvar=False, ddof=0, **kwargs)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if npcs is None:
        if chi2min is None:
            npcs = len(eigenvalues)
        else:
            npcs = len(eigenvalues) - np.sum(np.cumsum(eigenvalues) < chi2min)
    if npcs > len(eigenvectors):
        raise ValueError('Number of requested components is {0:d}, but dimension is already {1:d} < {0:d}.'.format(npcs, len(eigenvalues)))
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
    """Turn outputs ``name`` to latex string."""
    toret = txt_to_latex(name)
    for full, symbol in [('loglikelihood', 'L'), ('logposterior', '\\mathcal{L}'), ('logprior', 'p')]:
        toret = toret.replace(full, symbol)
    return toret


class Monitor(BaseClass):

    def __init__(self, quantities=('time',)):
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
        self._start = {quantity: getattr(self, quantity)() for quantity in self.quantities}

    def stop(self):
        stop = {quantity: getattr(self, quantity)() for quantity in self.quantities}
        self._counter += 1
        self._diffs = {quantity: stop[quantity] - self._start[quantity] + diff for quantity, diff in self._diffs.items()}
        self._start = stop

    @property
    def counter(self):
        return self._counter

    def get(self, quantity, average=True):
        if average:
            if self._counter == 0:
                return np.nan
            return self._diffs[quantity] / self._counter
        return self._diffs[quantity]

    def reset(self):
        self._diffs = {quantity: 0. for quantity in self.quantities}
        self._counter = 0
        self.start()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()
