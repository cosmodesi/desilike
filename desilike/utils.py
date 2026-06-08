"""Generic I/O utilities for desilike."""

import logging
import math
import os
import sys
import time
import json
import shutil

import numpy as np


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """Set up logging.

    Parameters
    ----------
    level : str or int, default=logging.INFO
        Logging level.
    stream : file-like, default=sys.stdout
        Stream to write log records to (ignored when *filename* is given).
    filename : str, optional
        If provided, write log records to this file instead of *stream*.
    filemode : str, default='w'
        Open mode for the log file (only used when *filename* is given).
    **kwargs
        Extra keyword arguments forwarded to :func:`logging.basicConfig`.
    """
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG,
                 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):
        def format(self, record):
            self._style._fmt = ('[%09.2f] ' % (time.time() - t0)
                                + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s')
            return super().format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        dirn = os.path.dirname(filename)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to plain Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.complexfloating):
            return complex(obj)
        if isinstance(obj, np.ndarray) and obj.ndim == 0:
            return self.default(obj[()])
        try:
            import jax
            if isinstance(obj, jax.Array):
                return self.default(np.asarray(obj))
        except ImportError:
            pass
        return super().default(obj)


def _npy_auto_format_specifier(dtype):
    if np.issubdtype(dtype, np.bool_):
        return '%d'
    if np.issubdtype(dtype, np.integer):
        return '%d'
    if np.issubdtype(dtype, np.floating):
        return '%.18e'
    if np.issubdtype(dtype, np.complexfloating):
        return '%.18e+%.18ej'
    if np.issubdtype(dtype, np.str_):
        return '%s'
    if np.issubdtype(dtype, np.bytes_):
        return f'%{dtype.itemsize}s'
    raise TypeError(f'Unsupported dtype: {dtype}')


def _h5py_recursively_write_dict(h5file, path, dic, with_attrs=True):
    """Save a nested dict of arrays to an HDF5 file."""
    import h5py
    for key, item in dic.items():
        if with_attrs and key == 'attrs':
            h5file[path].attrs.update(item)
            continue
        path_key = f'{path}/{key}'.rstrip('/')
        if isinstance(item, dict):
            h5file.create_group(path_key, track_order=True)
            _h5py_recursively_write_dict(h5file, path_key, item, with_attrs=with_attrs)
        else:
            item = np.asarray(item)
            try:
                if item.dtype.kind in ('U', 'S') or (item.size and isinstance(item.flat[0].item(), str)):
                    dset = h5file.create_dataset(path_key, shape=item.shape, dtype=h5py.string_dtype())
                    if item.size:
                        dset[...] = item
                else:
                    h5file.create_dataset(path_key, data=item)
            except Exception as exc:
                raise ValueError(f'Failed to write {key!r} (value {item!r}) as HDF5 dataset') from exc


def _h5py_recursively_read_dict(h5file, path='/'):
    """Load a nested dict of arrays from an HDF5 file."""
    import h5py
    dic = {}
    for key, item in h5file[path].items():
        path_key = f'{path}/{key}'.rstrip('/')
        if isinstance(item, h5py.Group):
            dic[key] = _h5py_recursively_read_dict(h5file, path_key)
        elif isinstance(item, h5py.Dataset):
            dic[key] = item[...]
            if h5py.check_string_dtype(item.dtype):
                dic[key] = dic[key].astype('U')
            if not dic[key].shape:
                dic[key] = dic[key].item()
    if h5file[path].attrs:
        dic['attrs'] = {k: v for k, v in h5file[path].attrs.items()}
    return dic


def _txt_recursively_write_dict(path, dic, with_attrs=True):
    """Save a nested dict of arrays to a directory tree of text files."""
    os.makedirs(path, exist_ok=True)
    for key, item in dic.items():
        path_key = os.path.join(path, key)
        if with_attrs and key == 'attrs':
            with open(path_key + '.json', 'w') as fh:
                json.dump(item, fh, cls=NumpyEncoder)
            continue
        if isinstance(item, dict):
            os.makedirs(path_key, exist_ok=True)
            _txt_recursively_write_dict(path_key, item, with_attrs=with_attrs)
        else:
            item = np.asarray(item)
            header = f'dtype = {item.dtype}\nshape = {item.shape}'
            flat = np.ravel(item)
            try:
                np.savetxt(path_key + '.txt', flat, fmt=_npy_auto_format_specifier(flat.dtype), header=header)
            except Exception as exc:
                raise ValueError(f'Failed to write {key!r} (value {item!r}) as text dataset') from exc


def _txt_recursively_read_dict(path):
    """Load a nested dict of arrays from a directory tree of text files."""
    dic = {}
    for key in os.listdir(path):
        path_key = os.path.join(path, key)
        if os.path.isdir(path_key):
            dic[key] = _txt_recursively_read_dict(path_key)
        elif os.path.isfile(path_key):
            if path_key.endswith('.json'):
                with open(path_key, 'r') as fh:
                    dic['attrs'] = json.load(fh)
                continue
            with open(path_key, 'r') as fh:
                dtype = np.dtype(fh.readline().rstrip('\r\n').replace(' ', '').replace('#dtype=', ''))
                shape = tuple(int(s) for s in
                              fh.readline().rstrip('\r\n').replace(' ', '').replace('#shape=', '')[1:-1].split(',')
                              if s)
                stem = key[:-4] if key.endswith('.txt') else key
                if dtype.kind == 'U':
                    rows = [line.rstrip('\r\n') for line in fh]
                    dic[stem] = np.array(rows, dtype=dtype)
                else:
                    dic[stem] = np.loadtxt(fh, dtype=dtype)
            dic[stem] = dic[stem].item() if not shape else dic[stem].reshape(shape)
    return dic


def _write(filename, state, overwrite=True):
    """Write a state dict to HDF5 (.h5 / .hdf5) or text (.txt) format."""
    filename = str(filename)
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if filename.endswith(('.h5', '.hdf5')):
        import h5py
        with h5py.File(filename, 'w' if overwrite else 'a') as f:
            _h5py_recursively_write_dict(f, '/', state)
    elif filename.endswith('.txt'):
        if overwrite:
            shutil.rmtree(filename[:-4], ignore_errors=True)
        _txt_recursively_write_dict(filename[:-4], state)
    else:
        raise ValueError(f'Unknown file format for {filename!r}; expected .h5, .hdf5, or .txt')


def _read(filename):
    """Read a state dict from HDF5 (.h5 / .hdf5) or text (.txt) format."""
    filename = str(filename)
    if filename.endswith(('.h5', '.hdf5')):
        import h5py
        with h5py.File(filename, 'r') as f:
            return _h5py_recursively_read_dict(f, '/')
    if filename.endswith('.txt'):
        return _txt_recursively_read_dict(filename[:-4])
    raise ValueError(f'Unknown file format for {filename!r}; expected .h5, .hdf5, or .txt')


# ── Number formatting ─────────────────────────────────────────────────────────

def _number_profile(value, sigfigs):
    """Return ``(sig_digits, power, is_neg)`` for *value* rounded to *sigfigs* significant figures."""
    if value == 0:
        return '0' * sigfigs, -(1 - sigfigs), False
    is_neg = value < 0
    if is_neg:
        value = abs(value)
    power = -1 * math.floor(math.log10(value)) + sigfigs - 1
    sig_digits = str(int(round(abs(value) * 10.0 ** power)))
    return sig_digits, int(-power), is_neg


def _place_dot(digits, power):
    """Place the decimal point in *digits* given the exponent *power*."""
    if power > 0:
        return digits + '0' * power
    if power < 0:
        power = abs(power)
        sigfigs = len(digits)
        if power < sigfigs:
            return digits[:-power] + '.' + digits[-power:]
        return '0.' + '0' * (power - sigfigs) + digits
    return digits + ('.' if digits[-1] == '0' else '')


def std_notation(value, sigfigs, positive_sign=False):
    """Standard (fixed-point) notation rounded to *sigfigs* significant figures.

    Examples
    --------
    >>> std_notation(5.36, 2)
    '5.4'
    >>> std_notation(5360, 2)
    '5400'
    >>> std_notation(0.05363, 3)
    '0.0536'
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits):
        is_neg = False
    return ('-' if is_neg else '+' if positive_sign else '') + _place_dot(sig_digits, power)


def sci_notation(value, sigfigs, filler='e', positive_sign=False):
    """Scientific notation rounded to *sigfigs* significant figures.

    Examples
    --------
    >>> sci_notation(123, 3, 'e')
    '1.23e2'
    >>> sci_notation(0.126, 2, 'e')
    '1.3e-1'
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits):
        is_neg = False
    dot_power = min(-(sigfigs - 1), 0)
    ten_power = power + sigfigs - 1
    return ('-' if is_neg else '+' if positive_sign else '') + _place_dot(sig_digits, dot_power) + filler + str(ten_power)


def round_measurement(x, u=0.1, v=None, sigfigs=2, positive_sign=False, notation='auto'):
    """Return string representations of a central value *x* with uncertainty *u* (and optionally *v*).

    Parameters
    ----------
    x : float
        Central value.
    u : float
        Upper (positive) uncertainty.
    v : float, optional
        Lower (negative) uncertainty.  When ``None``, only ``(xr, ur)`` is returned.
    sigfigs : int
        Number of significant figures for the uncertainties; the central value
        is rounded to match.
    positive_sign : bool or 'u'
        Prefix positive numbers with ``'+'``.  ``'u'`` applies only to the
        uncertainties, not the central value.
    notation : 'auto', 'std', or 'sci'
        ``'auto'`` picks ``'std'`` for values in ``(1e-3, 1e3)``, else ``'sci'``.

    Returns
    -------
    xr, ur : str
        When *v* is ``None``.
    xr, ur, vr : str
        When *v* is provided.
    """
    x, u = float(x), float(u)
    return_v = v is not None
    v = float(v) if return_v else -abs(u)

    logx = 0 if x == 0. or not np.isfinite(x) else math.floor(math.log10(abs(x)))
    logu = logx if u == 0. or not np.isfinite(u) else math.floor(math.log10(abs(u)))
    logv = logx if v == 0. or not np.isfinite(v) else math.floor(math.log10(abs(v)))
    if x == 0.:
        logx = max(logu, logv)

    def _fmt(val, nfigs, ps):
        if not np.isfinite(val):
            return str(val)
        note = notation
        if note == 'auto':
            note = 'std' if (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)) else 'sci'
        if note == 'std':
            return std_notation(val, nfigs, positive_sign=ps)
        return sci_notation(val, nfigs, positive_sign=ps)

    if logv > logu:
        sf = (logx - logu + sigfigs, sigfigs, logv - logu + sigfigs)
    else:
        sf = (logx - logv + sigfigs, logu - logv + sigfigs, sigfigs)

    ps_center = bool(positive_sign) and positive_sign != 'u'
    ps_err    = bool(positive_sign)
    xr = _fmt(x, sf[0], ps_center)
    ur = _fmt(u, sf[1], ps_err)
    vr = _fmt(v, sf[2], ps_err)

    if return_v:
        return xr, ur, vr
    return xr, ur


# ── Type registry and top-level read / write ──────────────────────────────────

_registry = {}


def register_type(cls):
    """Register *cls* so that :func:`read` can reconstruct it automatically.

    The class must have a ``_name`` class attribute (a unique string key) and
    implement ``__getstate__`` / ``__setstate__``.

    Can be used as a decorator::

        @register_type
        class MyClass:
            _name = 'MyClass'
            ...
    """
    _registry[cls._name] = cls
    return cls


def write(filename, obj):
    """Write *obj* to *filename*.

    The object's class must be registered with :func:`register_type` and must
    implement ``__getstate__``.  The format is inferred from the file extension
    (``.h5`` / ``.hdf5`` for HDF5, ``.txt`` for plain text).

    Parameters
    ----------
    filename : str
    obj : registered object
    """
    _write(filename, obj.__getstate__(to_file=True))


def read(filename):
    """Read an object from *filename*, dispatching to the correct class automatically.

    The class is identified by the ``__class__`` key stored in the file's root
    ``attrs`` at write time.  The class must have been registered with
    :func:`register_type`.

    Parameters
    ----------
    filename : str

    Returns
    -------
    Registered object instance.
    """
    state = _read(filename)
    cls_name = state.get('attrs', {}).get('__class__')
    if cls_name is None:
        raise ValueError(
            f'Cannot determine object type from {filename!r}: '
            f'no __class__ key found in root attrs'
        )
    try:
        cls = _registry[cls_name]
    except KeyError:
        raise ValueError(
            f'Unknown class {cls_name!r} in {filename!r}; '
            f'registered types: {list(_registry)}'
        )
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj
