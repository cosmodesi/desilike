"""Samples — a VariableCollection whose Variables hold multi-sample arrays."""

import copy

import numpy as np

from ..parameter import Variable, VariableCollection
from ..utils import register_type, write as _utils_write, read as _utils_read
from ..distributed import default_mpicomm, get_mpicomm, gather as _mpi_gather, scatter as _mpi_scatter, send as _mpi_send, recv as _mpi_recv


def _vals(samples, name):
    """Return the flat value array for *name*, shape ``(size, *var.shape)``."""
    var = VariableCollection.__getitem__(samples, name)
    return np.asarray(var._value).reshape((samples.size,) + var.shape)


def _normalise_params(samples, params):
    """Normalise *params* to a ``(is_scalar, list_of_name_strings)`` pair.

    When *params* is ``None`` all variable names are returned.  A single
    string or Variable is treated as scalar (``is_scalar=True``) so the
    caller can unwrap the one-element result list.
    """
    if params is None:
        return False, [p.name for p in samples]
    if isinstance(params, VariableCollection):
        return False, [p.name for p in params]
    scalar = not isinstance(params, (list, tuple))
    if scalar:
        params = [params]
    return scalar, [p if isinstance(p, str) else p.name for p in params]


@register_type
class Samples(VariableCollection):
    """Collection of Variables holding multi-sample arrays.

    Every Variable's ``_value`` is stored as a NumPy array of shape
    ``samples.shape + variable.shape``.  ``samples.shape`` is the leading
    batch of sample dimensions (e.g. ``(n_steps,)`` for a flat chain,
    ``(n_chains, n_steps)`` for a 2-D chain).  ``variable.shape`` is the
    intrinsic per-sample shape (``()`` for scalar parameters, ``(15,)`` for a
    power-spectrum vector, etc.).

    This class handles structure, I/O, and export.
    Statistical methods (mean, variance, quantiles, …) live in the subclass
    :class:`~desilike.samples.chain.Chain`.

    Examples
    --------
    Building from a dict of arrays::

        s = Samples({'omega_m': rng.normal(0.3, 0.01, 1000),
                     'sigma8':  rng.normal(0.8, 0.02, 1000)})

    Non-scalar variable (must pass a Variable key so the shape is unambiguous)::

        pk_var = Variable('pk', value=np.zeros(50))   # shape=(50,)
        s[pk_var] = rng.normal(1., 0.1, (1000, 50))

    Slicing, concatenation::

        s2 = s[100:]
        s_all = Samples.concatenate(s1, s2)

    I/O::

        s.write('samples.h5')
        s2 = Samples.read('samples.h5')
    """

    _name = 'Samples'

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, data=None, attrs=None):
        """
        Parameters
        ----------
        data : None, Samples, list[Variable], or dict
            - ``None``: empty collection.
            - ``Samples``: shallow copy.
            - ``list[Variable]``: Variables whose ``_value`` is already set to
              sample arrays of shape ``samples.shape + variable.shape``.
            - ``dict``: keys are either strings (→ scalar Variable created) or
              Variable objects (→ Variable's existing shape is honoured); values
              are NumPy arrays.
        attrs : dict, optional
            Arbitrary metadata stored on the object (e.g. sampler name).
        """
        self._data = []
        self.attrs = dict(attrs or {})
        if data is None:
            return
        if isinstance(data, Samples):
            self._data = [copy.copy(v) for v in data._data]
            self.attrs = dict(data.attrs)
            return
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, Variable):
                    raise ValueError(
                        f'List entries must be Variable instances, '
                        f'got {type(item).__name__}'
                    )
                self.set(item)
            return
        if isinstance(data, dict):
            for key, values in data.items():
                self[key] = values
            return
        raise ValueError(
            f'Cannot construct {type(self).__name__} from {type(data).__name__}'
        )

    # ── shape ─────────────────────────────────────────────────────────────────

    @property
    def shape(self):
        """Leading sample dimensions, derived from the first non-None Variable value."""
        for var in self._data:
            if var._value is None:
                continue
            n = len(var.shape)
            vshape = np.shape(var._value)
            return vshape if n == 0 else vshape[:-n]
        return ()

    @property
    def ndim(self):
        """Number of sample dimensions (``len(self.shape)``)."""
        return len(self.shape)

    @property
    def size(self):
        """Total number of samples (product of ``self.shape``)."""
        shape = self.shape
        return int(np.prod(shape, dtype='intp')) if shape else 0

    def __len__(self):
        """Length along the first sample dimension."""
        shape = self.shape
        return shape[0] if shape else 0

    # ── item access ───────────────────────────────────────────────────────────

    def __getitem__(self, key):
        """
        - ``str`` or ``Variable`` → return the corresponding Variable object.
        - ``int``, ``slice``, or array-like → return a new instance of the same
          type sliced along the first sample axis (integer indices are wrapped
          in a list so the sample dimension is preserved).
        """
        if isinstance(key, (str, Variable)):
            name = key if isinstance(key, str) else key.name
            return VariableCollection.__getitem__(self, name)
        # Sample-axis slice; wrap bare integer so sample dim is not collapsed.
        index = [key] if not isinstance(key, slice) and np.ndim(key) == 0 else key
        new = self.__class__(attrs=dict(self.attrs))
        for var in self._data:
            v2 = copy.copy(var)
            if v2._value is not None:
                v2._value = np.asarray(v2._value)[index]
            new._data.append(v2)
        return new

    def __setitem__(self, key, values):
        """Set sample values for variable *key* (``str`` or ``Variable``)."""
        if not isinstance(key, (str, Variable)):
            raise KeyError(
                f'Key must be str or Variable, not {type(key).__name__!r}'
            )
        self.set(key, np.asarray(values))

    def set(self, var, values=None):
        """Add or replace a Variable in the collection.

        Parameters
        ----------
        var : Variable or str
            The variable whose ``shape`` encodes its intrinsic (per-sample)
            shape.  String keys create a scalar Variable (``shape=()``).
        values : array, optional
            Sample array of shape ``samples.shape + var.shape``.  When omitted
            *var* must already have ``_value`` set correctly.
        """
        if isinstance(var, str):
            if VariableCollection.__contains__(self, var):
                var = copy.copy(VariableCollection.__getitem__(self, var))
            else:
                var = Variable(var)   # shape = ()
        else:
            var = copy.copy(var)

        if values is not None:
            values = np.asarray(values)
            sshape = self.shape
            if sshape:
                lss = len(sshape)
                if values.shape[:lss] != sshape:
                    raise ValueError(
                        f"Variable {var.name!r}: expected array with leading shape "
                        f"{sshape}, got {values.shape}"
                    )
                var.shape = values.shape[lss:]
            var._value = values

        for i, v in enumerate(self._data):
            if v.name == var.name:
                self._data[i] = var
                return
        self._data.append(var)

    # ── shape manipulation ────────────────────────────────────────────────────

    def reshape(self, *shape):
        """Return a copy with the sample dimensions reshaped.

        Accepts the same calling conventions as ``numpy.reshape``:
        ``s.reshape(100, 10)`` or ``s.reshape((100, 10))``.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new = self.__class__(attrs=dict(self.attrs))
        for var in self._data:
            v2 = copy.copy(var)
            if v2._value is not None:
                v2._value = np.asarray(v2._value).reshape(shape + var.shape)
            new._data.append(v2)
        return new

    def copy(self):
        """Return a deep copy of this object."""
        return copy.deepcopy(self)

    def ravel(self):
        """Return a copy flattened to a single sample dimension."""
        return self.reshape(self.size)

    @classmethod
    def concatenate(cls, *others, axis=0):
        """Concatenate samples along *axis* (default 0).

        All inputs must contain the same variable names in the same order.
        Accepts either ``cls.concatenate(s1, s2, …)`` or
        ``cls.concatenate([s1, s2, …])``.
        """
        if len(others) == 1 and isinstance(others[0], (list, tuple)):
            others = list(others[0])
        others = [o for o in others if o.size > 0]
        if not others:
            return cls()
        names0 = [v.name for v in others[0]._data]
        for other in others[1:]:
            names_i = [v.name for v in other._data]
            if names_i != names0:
                raise ValueError(
                    f'Cannot concatenate: variable names differ '
                    f'({names0} vs {names_i})'
                )
        new = cls(attrs=dict(others[0].attrs))
        for name in names0:
            v0 = VariableCollection.__getitem__(others[0], name)
            arrays = [
                np.asarray(VariableCollection.__getitem__(o, name)._value)
                for o in others
            ]
            v2 = copy.copy(v0)
            v2._value = np.concatenate(arrays, axis=axis)
            new._data.append(v2)
        return new

    # ── export ────────────────────────────────────────────────────────────────

    def to_array(self, params=None, struct=True):
        """Convert to a NumPy (structured) array.

        Parameters
        ----------
        params : list or None
            Variable names to include.  Defaults to all.
        struct : bool, default True
            When ``True`` return a structured array of shape ``self.shape``
            with one named field per variable.  When ``False`` return a plain
            ``(n_params, *self.shape, *var.shape)`` array (only sensible when
            all variable shapes are the same).
        """
        if params is None:
            params = [v.name for v in self._data]
        else:
            params = [p if isinstance(p, str) else p.name for p in params]
        vars_ = [VariableCollection.__getitem__(self, n) for n in params]
        if struct:
            dtype = [(n, v._value.dtype, v.shape) for n, v in zip(params, vars_)]
            out = np.empty(self.shape, dtype=dtype)
            for n, v in zip(params, vars_):
                out[n] = np.asarray(v)
            return out
        return np.array([np.asarray(v) for v in vars_])

    def to_dict(self, params=None):
        """Return ``{name: array}`` dict.

        Parameters
        ----------
        params : list or None
            Variable names.  Defaults to all.
        """
        if params is None:
            params = [v.name for v in self._data]
        else:
            params = [p if isinstance(p, str) else p.name for p in params]
        return {
            n: VariableCollection.__getitem__(self, n).value
            for n in params
        }

    # ── I/O ──────────────────────────────────────────────────────────────────

    def __getstate__(self, to_file=False):
        """Serialise.  Variable shapes and sample values are both preserved."""
        state = VariableCollection.__getstate__(self, to_file=to_file)
        if to_file:
            state['attrs'].update({k: str(v) for k, v in self.attrs.items()})
        else:
            state['attrs'] = dict(self.attrs)
        return state

    def __setstate__(self, state):
        is_file = 'attrs' in state and '__class__' in state.get('attrs', {})
        # VariableCollection.__setstate__ already excludes 'attrs' from parameter names.
        VariableCollection.__setstate__(self, state)
        if is_file:
            self.attrs = {k: v for k, v in state.get('attrs', {}).items()
                          if k != '__class__'}
        else:
            # 'attrs' is the current key; accept legacy '_samples_attrs' / '_chain_attrs' too.
            self.attrs = dict(state.get('attrs', state.get('_samples_attrs', state.get('_chain_attrs', {}))))

    def write(self, filename):
        """Write to an HDF5 (``.h5`` / ``.hdf5``) or text (``.txt``) file."""
        _utils_write(filename, self)

    @classmethod
    def read(cls, filename):
        """Read a :class:`Samples` (or subclass) written by :meth:`write`."""
        return _utils_read(filename)

    # ── MPI utilities ─────────────────────────────────────────────────────────

    @default_mpicomm
    def gather(self, mpicomm=None, mpiroot=...):
        """Gather samples from all MPI ranks and return the concatenated result.

        Parameters
        ----------
        mpicomm : MPI communicator, optional
            Communicator to use.  Defaults to :func:`~desilike.distributed.get_mpicomm`.
        mpiroot : int or Ellipsis, optional
            Destination rank.  :data:`Ellipsis` (default) performs an
            Allgatherv — every rank receives the full result.

        Returns
        -------
        Samples or None
            The concatenated samples on *mpiroot*, or on every rank when
            ``mpiroot`` is :data:`Ellipsis`.  Returns ``None`` on non-root
            ranks for a directional gather.
        """
        if mpicomm.size == 1:
            return copy.copy(self)
        new = self.__class__(attrs=dict(self.attrs))
        for var in self._data:
            gathered = _mpi_gather(np.asarray(var._value), mpiroot=mpiroot, mpicomm=mpicomm)
            if gathered is not None:
                var_copy = copy.copy(var)
                var_copy._value = gathered
                new._data.append(var_copy)
        return new if (mpiroot is ... or mpicomm.rank == mpiroot) else None

    @default_mpicomm
    def scatter(self, mpicomm=None):
        """Scatter samples from rank 0 to all MPI ranks.

        Each rank receives a contiguous slice of the sample axis of length
        approximately ``len(self) // mpicomm.size``.

        Parameters
        ----------
        mpicomm : MPI communicator, optional
            Communicator to use.  Defaults to :func:`~desilike.distributed.get_mpicomm`.

        Returns
        -------
        Samples
            The local slice on each rank.
        """
        if mpicomm.size == 1:
            return copy.copy(self)
        total_size = mpicomm.bcast(len(self), root=0)
        rank   = mpicomm.rank
        nranks = mpicomm.size
        local_start = rank * total_size // nranks
        local_stop  = (rank + 1) * total_size // nranks
        local_size  = local_stop - local_start
        new = self.__class__(attrs=dict(self.attrs))
        for var in self._data:
            scattered = _mpi_scatter(np.asarray(var._value) if rank == 0 else None,
                                     size=local_size, mpicomm=mpicomm)
            var_copy = copy.copy(var)
            var_copy._value = scattered
            new._data.append(var_copy)
        return new

    @classmethod
    @default_mpicomm
    def sendrecv(cls, value, source=0, dest=0, tag=0, mpicomm=None):
        """Send samples from rank *source* to rank *dest* and receive them.

        Parameters
        ----------
        value : Samples or None
            Samples to send.  Only used on rank *source*.
        source : int, optional
            Sending rank.  Default is 0.
        dest : int, optional
            Receiving rank.  Default is 0.
        tag : int, optional
            MPI message tag.  Default is 0.
        mpicomm : MPI communicator, optional
            Communicator to use.  Defaults to :func:`~desilike.distributed.get_mpicomm`.

        Returns
        -------
        Samples or None
            The received samples on rank *dest*, ``None`` on all other ranks.
            When ``source == dest == mpicomm.rank`` a deep copy of *value* is
            returned immediately without any MPI communication.
        """
        if source == dest == mpicomm.rank:
            return copy.deepcopy(value)
        if mpicomm.rank == source and value is not None:
            # Send metadata dict (variable names, shapes, derived flags, attrs)
            meta = {
                'names':   [v.name for v in value._data],
                'derived': [v._derived for v in value._data],
                'shapes':  [v.shape for v in value._data],
                'attrs':   dict(value.attrs),
            }
            mpicomm.send(meta, dest=dest, tag=tag)
            for var in value._data:
                _mpi_send(np.asarray(var._value), dest=dest, tag=tag, mpicomm=mpicomm)
        result = None
        if mpicomm.rank == dest:
            meta = mpicomm.recv(source=source, tag=tag)
            result = cls(attrs=meta['attrs'])
            for name, derived, shape in zip(meta['names'], meta['derived'], meta['shapes']):
                arr = _mpi_recv(source=source, tag=tag, mpicomm=mpicomm)
                var = Variable(name, derived=derived)
                var.shape = shape
                var._value = arr
                result._data.append(var)
        return result

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self):
        return f'{type(self).__name__}(shape={self.shape}, params={self.names()})'
