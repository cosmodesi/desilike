"""
A few MPI utilities.
TODO: write ersatz for all MPI functions used in the package, in case mpi4py is not available.
"""

import time
import copy
import logging
import functools
from contextlib import contextmanager

import random
import numpy as np

use_mpi = True

try:
    import mpi4py
except ImportError:
    use_mpi = False

if use_mpi:
    from mpi4py import MPI
    COMM_WORLD = MPI.COMM_WORLD
    COMM_SELF = MPI.COMM_SELF
    ANY_SOURCE = MPI.ANY_SOURCE
    ANY_TAG = MPI.ANY_TAG
    Status = MPI.Status
    Get_processor_name = MPI.Get_processor_name

    #from mpi4py.futures import MPIPoolExecutor
    #MPIPool = MPIPoolExecutor

else:
    # Fake MPI
    class Comm(object):
        rank = 0
        size = 1

        def Barrier(self):
            return

        def barrier(self):
            return

        def bcast(self, value, **kwargs):
            return copy.copy(value)


    def Status():
        return 0


    def Get_processor_name():
        return ''

    COMM_WORLD = Comm()
    COMM_SELF = Comm()
    ANY_SOURCE = 0
    ANY_TAG = 0


class CurrentMPIComm(object):
    """Class to facilitate getting and setting the current MPI communicator, taken from nbodykit."""
    logger = logging.getLogger('CurrentMPIComm')

    _stack = [COMM_WORLD]

    @staticmethod
    def enable(func):
        """
        Decorator to attach the current MPI communicator to the input
        keyword arguments of ``func``, via the ``mpicomm`` keyword.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mpicomm = kwargs.get('mpicomm', None)
            if mpicomm is None:
                for arg in args:
                    mpicomm = getattr(arg, 'mpicomm', None)
            if mpicomm is None:
                mpicomm = CurrentMPIComm.get()
            kwargs['mpicomm'] = mpicomm
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    @contextmanager
    def enter(cls, mpicomm):
        """
        Enter a context where the current default MPI communicator is modified to the
        argument ``mpicomm``. After leaving the context manager the communicator is restored.
        """
        cls.push(mpicomm)

        yield

        cls.pop()

    @classmethod
    def push(cls, mpicomm):
        """Switch to a new current default MPI communicator."""
        cls._stack.append(mpicomm)
        if mpicomm.rank == 0:
            cls.logger.info('Entering a current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()

    @classmethod
    def pop(cls):
        """Restore to the previous current default MPI communicator."""
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Leaving current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()
        cls._stack.pop()
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Restored current communicator to size {:d}'.format(mpicomm.size))

    @classmethod
    def get(cls):
        """Get the default current MPI communicator. The initial value is ``MPI.COMM_WORLD``."""
        return cls._stack[-1]


@CurrentMPIComm.enable
def local_size(size, mpicomm=None):
    """
    Divide global ``size`` into local (process) size.

    Parameters
    ----------
    size : int
        Global size.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    localsize : int
        Local size. Sum of local sizes over all processes equals global size.
    """
    start = mpicomm.rank * size // mpicomm.size
    stop = (mpicomm.rank + 1) * size // mpicomm.size
    # localsize = size // mpicomm.size
    # if mpicomm.rank < size % mpicomm.size: localsize += 1
    return stop - start


@CurrentMPIComm.enable
def gather(data, mpiroot=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py.
    Gather the input data array from all ranks to the specified ``mpiroot``.
    This uses ``Gatherv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.

    Parameters
    ----------
    data : array_like
        The data on each rank to gather.

    mpiroot : int, Ellipsis, default=0
        The rank number to gather the data to. If mpiroot is Ellipsis or None,
        broadcast the result to all ranks.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like, None
        The gathered data on mpiroot, and `None` otherwise.
    """
    if mpiroot is None: mpiroot = Ellipsis

    if all(mpicomm.allgather(np.isscalar(data))):
        if mpiroot is Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=mpiroot)
        if mpicomm.rank == mpiroot:
            return np.array(gathered)
        return None

    # Need C-contiguous order
    data = np.asarray(data)
    shape, dtype = data.shape, data.dtype
    data = np.ascontiguousarray(data)

    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = mpicomm.allgather(data.shape)
    dtypes = mpicomm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError('mismatch between data type fields in structured data')

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError('object data types ("O") not allowed in structured data in gather')

        # compute the new shape for each rank
        newlength = mpicomm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = gather(data[name], mpiroot=mpiroot, mpicomm=mpicomm)
            if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in structured data in gather')

    # check for bad dtypes and bad shapes
    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape, bad_dtype = None, None

    if mpiroot is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype), root=mpiroot)

    if bad_shape:
        raise ValueError('mismatch between shape[1:] across ranks in gather')
    if bad_dtype:
        raise ValueError('mismatch between dtypes across ranks in gather')

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = mpicomm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = mpicomm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to mpiroot
    if mpiroot is Ellipsis:
        mpicomm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        mpicomm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=mpiroot)

    dt.Free()

    return recvbuffer


@CurrentMPIComm.enable
def bcast(data, mpiroot=0, mpicomm=None):
    """
    Broadcast the input data array across all ranks, assuming ``data`` is
    initially only on `mpiroot` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.

    Parameters
    ----------
    data : array_like or None
        On `mpiroot`, this gives the data to broadcast.

    mpiroot : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        ``data`` on each rank.
    """
    if mpicomm.bcast(np.isscalar(data) if mpicomm.rank == mpiroot else None, root=mpiroot):
        return mpicomm.bcast(data, root=mpiroot)

    if mpicomm.rank == mpiroot:
        # Need C-contiguous order
        data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=mpiroot)

    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in send; please specify specific data type')

    # initialize empty data on non-mpiroot ranks
    if mpicomm.rank != mpiroot:
        data = np.empty(shape, dtype=dtype)

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.Barrier()
    mpicomm.Bcast([data, dt], root=mpiroot)
    dt.Free()
    return data


@CurrentMPIComm.enable
def scatter(data, size=None, mpiroot=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming ``data`` is
    initially only on `mpiroot` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        On `mpiroot`, this gives the data to split and scatter.

    size : int
        Length of data on current rank.

    mpiroot : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        The chunk of ``data`` that each rank gets.
    """
    counts = None
    if size is not None:
        counts = np.asarray(mpicomm.allgather(size), order='C')

    if mpicomm.rank == mpiroot:
        # Need C-contiguous order
        data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=mpiroot)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in scatter; please specify specific data type')

    # initialize empty data on non-mpiroot ranks
    if mpicomm.rank != mpiroot:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newshape[0] = newlength = local_size(shape[0], mpicomm=mpicomm)
    else:
        if counts.sum() != shape[0]:
            raise ValueError('The sum of the `size` needs to be equal to data length')
        newshape[0] = counts[mpicomm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = mpicomm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=mpiroot)
    dt.Free()
    return recvbuffer


@CurrentMPIComm.enable
def send(data, dest, tag=0, mpicomm=None):
    """
    Send input array ``data`` to process ``dest``.

    Parameters
    ----------
    data : array
        Array to send.

    dest : int
        Rank of process to send array to.

    tag : int, default=0
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.
    """
    data = np.asarray(data)
    shape, dtype = (data.shape, data.dtype)
    data = np.ascontiguousarray(data)

    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in send; please specify specific data type')

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.send((shape, dtype), dest=dest, tag=tag)
    mpicomm.Send([data, dt], dest=dest, tag=tag)
    dt.Free()


@CurrentMPIComm.enable
def recv(source=ANY_SOURCE, tag=ANY_TAG, mpicomm=None):
    """
    Receive array from process ``source``.

    Parameters
    ----------
    source : int, default=MPI.ANY_SOURCE
        Rank of process to receive array from.

    tag : int, default=MPI.ANY_TAG
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    data : array
    """
    shape, dtype = mpicomm.recv(source=source, tag=tag)
    data = np.zeros(shape, dtype=dtype)

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.Recv([data, dt], source=source, tag=tag)
    dt.Free()
    return data


@CurrentMPIComm.enable
def bcast_seed(seed=None, mpicomm=None, size=None):
    """
    Generate array of seeds.

    Parameters
    ---------
    seed : int, default=None
        Random seed to use when generating seeds.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    size : int, default=None
        Number of seeds to be generated.

    Returns
    -------
    seeds : array
        Array of seeds.
    """
    if mpicomm.rank == 0:
        seeds = np.random.RandomState(seed=seed).randint(0, high=0xffffffff, size=size)
    return bcast(seeds if mpicomm.rank == 0 else None, mpiroot=0, mpicomm=mpicomm)


@CurrentMPIComm.enable
def set_common_seed(seed=None, mpicomm=None):
    """
    Set same global :mod:`np.random` and :mod:`random` seed for all MPI processes.

    Parameters
    ----------
    seed : int, default=None
        Random seed to broadcast on all processes.
        If ``None``, draw random seed.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    Returns
    -------
    seed : int
        Seed used to initialize :mod:`np.random` and :mod:`random` global states.
    """
    if seed is None:
        if mpicomm.rank == 0:
            seed = np.random.randint(0, high=0xffffffff)
    seed = mpicomm.bcast(seed, root=0)
    np.random.seed(seed)
    random.seed(seed)
    return seed


@CurrentMPIComm.enable
def set_independent_seed(seed=None, mpicomm=None, size=10000):
    """
    Set independent global :mod:`np.random` and :mod:`random` seed for all MPI processes.

    Parameters
    ---------
    seed : int, default=None
        Random seed to use when generating seeds.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    size : int, default=10000
        Number of seeds to be generated.
        To ensure random draws are independent of the number of ranks,
        this should be larger than the total number of processes that will ever be used.

    Returns
    -------
    seed : int
        Seed used to initialize :mod:`np.random` and :mod:`random` global states.
    """
    seed = bcast_seed(seed=seed, mpicomm=mpicomm, size=size)[mpicomm.rank]
    np.random.seed(seed)
    random.seed(seed)
    return seed


def barrier_idle(mpicomm, tag=0, sleep=0.1):
    """
    MPI barrier fonction that solves the problem that idle processes occupy 100% CPU.
    See https://goo.gl/NofOO9.
    """
    size = mpicomm.size
    if size == 1: return
    rank = mpicomm.rank
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = mpicomm.isend(None, dst, tag)
        while not mpicomm.Iprobe(src, tag):
            time.sleep(sleep)
        mpicomm.recv(None, src, tag)
        req.Wait()
        mask <<= 1