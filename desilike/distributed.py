"""MPI and JAX distributed-execution utilities for desilike.

Provides a thin wrapper around :mod:`mpi4py` that degrades gracefully to a
no-op stub when mpi4py is not installed (single-process mode), plus an
:func:`initialize` entry-point that wires up the global MPI communicator and
configures JAX to use the right CPU devices for this process.

Typical multi-process usage::

    from desilike.distributed import initialize
    initialize(nshards=4)   # call before any JAX work or desilike imports
    # ... rest of the script
"""

import os
import copy
import functools
from typing import Callable
import numpy as np

_use_mpi = True

try:
    import mpi4py  # noqa: F401
except ImportError:
    _use_mpi = False

if _use_mpi:
    from mpi4py import MPI


class _Comm:
    """Stub MPI communicator for single-process (no mpi4py) mode."""

    rank = 0
    size = 1

    def Barrier(self):
        pass

    def barrier(self):
        pass

    def bcast(self, value, root=0, **kwargs):
        return copy.copy(value)

    def allgather(self, value):
        return [copy.copy(value)]

    def allreduce(self, value):
        return copy.copy(value)

    def gather(self, value, root=0):
        return [copy.copy(value)]


# Module-level communicator override set by initialize(); None means "use
# MPI.COMM_WORLD (if available) or the single-process stub".
_mpicomm = None

# Number of JAX shards per MPI process, set by initialize().
_nshards = 1


def get_nshards():
    """Return the number of JAX shards per MPI process set by :func:`initialize`."""
    return _nshards


def get_mpicomm():
    """Return the active MPI communicator.

    Resolution order:

    1. The communicator installed by :func:`initialize` (if any).
    2. ``MPI.COMM_WORLD`` when *mpi4py* is available.
    3. A single-process stub communicator otherwise.

    Returns
    -------
    MPI communicator
    """
    if _mpicomm is not None:
        return _mpicomm
    if _use_mpi:
        return MPI.COMM_WORLD
    return _Comm()


def default_mpicomm(func: Callable):
    """Decorator: inject the default MPI communicator into *func* when ``mpicomm`` is not given.

    The wrapped function must accept ``mpicomm`` as a keyword argument.
    When the caller omits it (or passes ``None``), :func:`get_mpicomm` is
    called to supply the active communicator.
    """
    @functools.wraps(func)
    def wrapper(*args, mpicomm=None, **kwargs):
        if mpicomm is None:
            mpicomm = get_mpicomm()
        return func(*args, mpicomm=mpicomm, **kwargs)
    return wrapper


def initialize(mpicomm=None, nshards: int = 1):
    """Configure the global MPI communicator and JAX device setup.

    **Must be called before any JAX computation** (i.e. before calling
    ``jax.devices()``, running JIT-compiled functions, or importing modules
    that trigger JAX initialisation) so that ``jax_num_cpu_devices`` takes
    effect.

    Parameters
    ----------
    mpicomm : MPI communicator, optional
        The communicator to use as the process-wide default.  When ``None``
        (the default) the existing result of :func:`get_mpicomm` is kept.
    nshards : int, optional
        Number of JAX CPU devices visible to *this* MPI rank.  Default is 1.

        When greater than 1 the function performs two actions so that each
        MPI rank has its own private set of physical CPU cores:

        1. **JAX device count** — calls
           ``jax.config.update('jax_num_cpu_devices', nshards)`` to expose
           exactly ``nshards`` CPU devices to JAX on this process.

        2. **CPU affinity** — if the current affinity mask is large enough to
           accommodate all ranks (``len(available_cpus) >= nshards * size``),
           the process is pinned to the ``nshards`` consecutive CPUs that
           belong to this rank (``rank * nshards … (rank+1) * nshards - 1``
           within the available set).  When the MPI launcher has already
           performed per-rank binding (smaller affinity mask), the existing
           binding is left untouched.

    Examples
    --------
    Single-process, 4 JAX shards::

        import desilike
        desilike.distributed.initialize(nshards=4)

    4 MPI ranks, 2 JAX shards each (8 physical cores total)::

        # mpirun -n 4 python script.py
        from mpi4py import MPI
        import desilike
        desilike.distributed.initialize(MPI.COMM_WORLD, nshards=2)
    """
    global _mpicomm, _nshards

    if mpicomm is not None:
        _mpicomm = mpicomm

    _nshards = nshards
    active = get_mpicomm()
    rank = active.rank
    size = active.size

    # Assign each MPI rank its own GPU: rank 0 sees GPU 0, rank 1 sees GPU 1,
    # etc.  Must be done before JAX is imported / initialised.
    # ``rank`` is the index of this process within its node (same as the
    # global rank when there is only one node, which is the typical case here).
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    # Mirror the guard used by jax.distributed.initialize() — raise early if
    # the XLA backend is already initialised, because device-count changes and
    # CUDA_VISIBLE_DEVICES will be silently ignored at that point.
    # (ref: jax/_src/distributed.py, ~line 321)
    import jax._src.xla_bridge as _xb
    if _xb.backends_are_initialized():
        raise RuntimeError(
            'desilike.distributed.initialize() must be called before any JAX '
            'computation or calls to jax.devices() / jax.device_put(), '
            'otherwise CUDA_VISIBLE_DEVICES and jax_num_cpu_devices will have '
            'no effect.'
        )

    import jax

    if nshards > 1:
        jax.config.update('jax_num_cpu_devices', nshards)

    # CPU affinity: bind this rank to a disjoint slice of physical cores so
    # that MPI ranks do not compete for the same CPUs.
    # Skip when the MPI launcher has already performed binding (the affinity
    # mask is already narrow) or when the OS does not expose sched_setaffinity.
    if nshards > 1 and hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        available_cpus = sorted(os.sched_getaffinity(0))
        if len(available_cpus) >= nshards * size:
            cpu_slice = available_cpus[rank * nshards:(rank + 1) * nshards]
            try:
                os.sched_setaffinity(0, cpu_slice)
            except OSError:
                pass  # Not permitted in this environment (containers, some HPC setups)


@default_mpicomm
def gather(data, mpiroot=0, mpicomm=None):
    """Gather *data* from all ranks to *mpiroot* (or to all ranks when ``mpiroot`` is :data:`Ellipsis`).

    When only one rank is present the data is returned immediately without
    any MPI call.

    Parameters
    ----------
    data : array_like
        The local data to gather.  Must have the same dtype and ``shape[1:]``
        on every rank.
    mpiroot : int or Ellipsis, default 0
        Destination rank.  Pass :data:`Ellipsis` (or ``None``) to perform an
        Allgatherv — every rank receives the full concatenated result.
    mpicomm : communicator, optional
        MPI communicator.  Defaults to :func:`get_mpicomm`.

    Returns
    -------
    array or None
        The concatenated array on *mpiroot* (or on every rank when
        ``mpiroot`` is :data:`Ellipsis`).  Returns ``None`` on non-root
        ranks for a directional gather.
    """
    if mpiroot is None:
        mpiroot = Ellipsis

    # Fast path: single rank — no communication needed.
    if mpicomm.size == 1:
        return np.asarray(data)

    data = np.asarray(data)
    data = np.ascontiguousarray(data)
    shape  = data.shape
    dtype  = data.dtype

    # Sanity checks (only on the receive side to keep communication cheap).
    shapes = mpicomm.allgather(shape)
    dtypes = mpicomm.allgather(dtype)

    bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
    bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    if bad_shape:
        raise ValueError('gather(): mismatch between shape[1:] across ranks')
    if bad_dtype:
        raise ValueError('gather(): mismatch between dtypes across ranks')

    local_length = shape[0]
    total_length = mpicomm.allreduce(local_length)
    new_shape    = (total_length,) + shape[1:]

    # Build a custom MPI datatype to handle multi-dimensional arrays without
    # exceeding the 2 GB mpi4py bytes limit.
    duplicity = int(np.prod(shape[1:], dtype='intp')) if shape[1:] else 1
    itemsize  = duplicity * dtype.itemsize
    mpi_dtype = MPI.BYTE.Create_contiguous(itemsize)
    mpi_dtype.Commit()

    counts  = np.array(mpicomm.allgather(local_length), dtype='i', order='C')
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    if mpiroot is Ellipsis:
        recvbuffer = np.empty(new_shape, dtype=dtype, order='C')
        mpicomm.Allgatherv([data, mpi_dtype], [recvbuffer, (counts, offsets), mpi_dtype])
    else:
        recvbuffer = np.empty(new_shape, dtype=dtype, order='C') if mpicomm.rank == mpiroot else None
        mpicomm.Gatherv([data, mpi_dtype], [recvbuffer, (counts, offsets), mpi_dtype] if recvbuffer is not None else None, root=mpiroot)

    mpi_dtype.Free()
    return recvbuffer


@default_mpicomm
def scatter(data, size=None, mpiroot=0, mpicomm=None):
    """Scatter *data* from rank *mpiroot* to all ranks.

    Parameters
    ----------
    data : array_like or None
        Array to scatter, present only on rank *mpiroot*.  Non-root ranks
        should pass ``None``.
    size : int, optional
        Number of rows the *calling* rank should receive.  When ``None``
        the rows are divided as evenly as possible.
    mpiroot : int, optional
        Sending rank.  Default is 0.
    mpicomm : communicator, optional
        MPI communicator.  Defaults to :func:`get_mpicomm`.

    Returns
    -------
    array
        The local slice on each rank.
    """
    if mpicomm.size == 1:
        return np.asarray(data)

    # Exchange metadata
    if mpicomm.rank == mpiroot:
        data = np.ascontiguousarray(np.asarray(data))
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=mpiroot)

    # Compute per-rank sizes
    if size is not None:
        counts = np.array(mpicomm.allgather(size), dtype='i', order='C')
    else:
        total  = shape[0]
        counts = np.array([
            (rank_idx + 1) * total // mpicomm.size - rank_idx * total // mpicomm.size
            for rank_idx in range(mpicomm.size)
        ], dtype='i', order='C')

    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    local_size = int(counts[mpicomm.rank])
    local_shape = (local_size,) + shape[1:]
    recvbuffer = np.empty(local_shape, dtype=dtype, order='C')

    duplicity = int(np.prod(shape[1:], dtype='intp')) if shape[1:] else 1
    itemsize  = duplicity * dtype.itemsize
    mpi_dtype = MPI.BYTE.Create_contiguous(itemsize)
    mpi_dtype.Commit()

    if mpicomm.rank != mpiroot:
        data = np.empty(0, dtype=dtype, order='C')

    mpicomm.Scatterv([data, (counts, offsets), mpi_dtype], [recvbuffer, mpi_dtype], root=mpiroot)
    mpi_dtype.Free()
    return recvbuffer


@default_mpicomm
def send(data, dest, tag=0, mpicomm=None):
    """Send *data* array to rank *dest*.

    Parameters
    ----------
    data : array_like
        Array to send.
    dest : int
        Destination rank.
    tag : int, optional
        Message tag.  Default is 0.
    mpicomm : communicator, optional
        MPI communicator.  Defaults to :func:`get_mpicomm`.
    """
    data = np.ascontiguousarray(np.asarray(data))
    shape, dtype = data.shape, data.dtype
    mpicomm.send((shape, dtype), dest=dest, tag=tag)
    if data.size:
        duplicity = int(np.prod(shape[1:], dtype='intp')) if shape[1:] else 1
        itemsize  = duplicity * dtype.itemsize
        mpi_dtype = MPI.BYTE.Create_contiguous(itemsize)
        mpi_dtype.Commit()
        mpicomm.Send([data, mpi_dtype], dest=dest, tag=tag)
        mpi_dtype.Free()


@default_mpicomm
def recv(source=None, tag=0, mpicomm=None):
    """Receive an array from rank *source*.

    Parameters
    ----------
    source : int, optional
        Source rank.  ``None`` accepts from any source (``MPI.ANY_SOURCE``).
    tag : int, optional
        Message tag.  Default is 0.
    mpicomm : communicator, optional
        MPI communicator.  Defaults to :func:`get_mpicomm`.

    Returns
    -------
    array
        The received array.
    """
    if source is None:
        source = MPI.ANY_SOURCE
    shape, dtype = mpicomm.recv(source=source, tag=tag)
    data = np.empty(shape, dtype=dtype, order='C')
    if data.size:
        duplicity = int(np.prod(shape[1:], dtype='intp')) if shape[1:] else 1
        itemsize  = duplicity * dtype.itemsize
        mpi_dtype = MPI.BYTE.Create_contiguous(itemsize)
        mpi_dtype.Commit()
        mpicomm.Recv([data, mpi_dtype], source=source, tag=tag)
        mpi_dtype.Free()
    return data
