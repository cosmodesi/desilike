"""MPI pool for distributing function evaluations across ranks.

Adapted from CosmoSIS (https://github.com/joezuntz/cosmosis).

When ``mpi4py`` is not installed or the communicator has only one rank, an
in-process serial pool is used transparently so all sampler code can call
``pool.map(fn, tasks)`` without special-casing.

Copyright 2014-23 The CosmoSIS Team

The CosmoSIS core is licensed as described below.  Some individual components
within the software have their own licenses - see the notices in their
directories.  Notably, Polychord and Multinest are licensed only for academic
use.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import builtins
from functools import partial

import numpy as np


def _apply_batched(function, tasks, batch_size):
    """Apply *function* to *tasks* with the given batching strategy.

    Parameters
    ----------
    function : callable
    tasks : list
    batch_size : int or None
        ``0``  — call ``function(task)`` once per element (no batching).
        ``None`` — call ``function(np.stack(tasks))`` once for all tasks.
        ``N > 0`` — call ``function(np.stack(chunk))`` for chunks of N.
    """
    if batch_size == 0:
        return list(builtins.map(function, tasks))
    if batch_size is None:
        return list(function(np.stack(tasks)))
    results = []
    for start in range(0, len(tasks), batch_size):
        results.extend(function(np.stack(tasks[start:start + batch_size])))
    return results


class FunctionWrapper:
    """Thin wrapper that avoids pickling the function across MPI processes.

    The function is stored in the local registry of each :class:`MPIPool` and
    only its *name* is transmitted when the wrapper is pickled and sent to
    worker ranks.
    """

    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __getstate__(self):
        # Do not pickle the function itself; workers reload it from registry.
        return dict(function=None, name=self.name)

    def __setstate__(self, state):
        self.__dict__ = state


class _stop_wait_message:
    def __repr__(self):
        return '<Stop wait message>'


def _error_function(task):
    raise RuntimeError('Pool was sent tasks before being told what function to apply.')


class _SerialPool:
    """Single-process pool; used when MPI is unavailable or comm.size == 1."""

    class _SerialComm:
        rank = 0
        size = 1

    def __init__(self, batch_size=0):
        self.comm = self._SerialComm()
        self.rank = 0
        self.size = 1
        self.batch_size = batch_size
        self._registry = {}

    @property
    def main(self):
        return True

    def save_function(self, function, name):
        self._registry[name] = function
        return FunctionWrapper(function, name)

    def load_function(self, obj):
        if isinstance(obj, FunctionWrapper):
            obj.function = self._registry.get(obj.name, obj.function)

    def wait(self):
        raise RuntimeError('Serial pool told to await jobs')

    def stop_wait(self):
        pass

    def map(self, function, tasks):
        tasks = list(tasks)
        if not tasks:
            return []
        return _apply_batched(function, tasks, self.batch_size)


class MPIPool:
    """An MPI pool that distributes tasks across ranks.

    On the main rank (rank 0) call :meth:`map`; on worker ranks call
    :meth:`wait` to enter the task-receive loop.
    """

    def __init__(self, comm=None, batch_size=0):
        try:
            from mpi4py import MPI
            self.MPI = MPI
        except ImportError:
            raise RuntimeError('MPI environment not found!')
        if comm is None:
            comm = self.MPI.COMM_WORLD
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.batch_size = batch_size
        self.function = _error_function
        self.registry = {}

    def save_function(self, function, name):
        """Register *function* locally and return a :class:`FunctionWrapper`."""
        self.registry[name] = function
        return FunctionWrapper(function, name)

    def load_function(self, obj):
        """Recursively restore :class:`FunctionWrapper` instances from registry."""
        if isinstance(obj, FunctionWrapper):
            obj.function = self.registry[obj.name]
            return
        if isinstance(obj, partial):
            self.load_function(obj.func)
        if isinstance(obj, dict):
            for elem in obj.values():
                self.load_function(elem)
        if hasattr(obj, '__dict__'):
            for elem in obj.__dict__.values():
                self.load_function(elem)
        if isinstance(obj, (list, tuple, set)):
            for elem in obj:
                self.load_function(elem)

    @property
    def main(self):
        return self.rank == 0

    def wait(self):
        """Worker loop: receive tasks, execute, return results."""
        if self.main:
            raise RuntimeError('Main node told to await jobs')
        status = self.MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=self.MPI.ANY_TAG, status=status)
            if isinstance(task, _stop_wait_message):
                return
            elif callable(task):
                self.load_function(task)
                self.function = task
                continue
            else:
                self.load_function(task)
                results = _apply_batched(self.function, task, self.batch_size)
                self.comm.send(results, dest=0, tag=status.tag)

    def stop_wait(self):
        """Signal all workers to exit their :meth:`wait` loop."""
        if self.main:
            for worker_rank in range(1, self.size):
                self.comm.isend(_stop_wait_message(), dest=worker_rank)

    def map(self, function, tasks):
        """Apply *function* to every element of *tasks* in parallel.

        The ``batch_size`` attribute controls batching per rank:
        ``0`` calls ``function(task)`` per element; ``None`` calls
        ``function(np.stack(slice))`` once per rank; ``N`` calls it in
        chunks of N.

        Must be called from the main rank only; workers should call
        :meth:`wait`.
        """
        if not self.main:
            self.wait()
            return

        tasks = list(tasks)
        if function is not self.function:
            self.function = function
            for worker_rank in range(1, self.size):
                self.comm.send(function, dest=worker_rank)

        # Distribute tasks to workers (round-robin).
        for worker_rank in range(1, self.size):
            self.comm.send(tasks[worker_rank::self.size], dest=worker_rank)

        # Process the main rank's share.
        main_slice = tasks[::self.size]
        results = [None] * len(tasks)
        results[::self.size] = _apply_batched(self.function, main_slice, self.batch_size)

        # Collect worker results in arrival order.
        status = self.MPI.Status()
        for _ in range(self.size - 1):
            result = self.comm.recv(source=self.MPI.ANY_SOURCE, status=status)
            results[status.source::self.size] = result
        return results


def make_pool(mpicomm, batch_size=0):
    """Return the appropriate pool for *mpicomm*.

    Parameters
    ----------
    batch_size : int or None
        ``0`` (default) — evaluate one task at a time (no batching).
        ``None`` — pass all tasks as a single stacked array per rank.
        ``N > 0`` — group tasks into chunks of N per rank.

    Returns a :class:`_SerialPool` when *mpicomm* has only one rank or when
    ``mpi4py`` is unavailable; otherwise returns an :class:`MPIPool`.
    """
    if mpicomm.size == 1:
        return _SerialPool(batch_size=batch_size)
    try:
        return MPIPool(comm=mpicomm, batch_size=batch_size)
    except RuntimeError:
        return _SerialPool(batch_size=batch_size)
