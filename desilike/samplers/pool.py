"""Adapted from CosmoSIS (https://github.com/joezuntz/cosmosis).

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
from functools import partial


class FunctionWrapper:
    """A simple function wrapper to avoid sending large functions."""

    def __init__(self, function, name):
        """Initialize the function wrapper.

        Parameters
        ----------
        function : function
            Function to wrap.
        name : str
            Name to be used. This is later used to identify and restore the
            function after the function wrapper is pickled and sent across
            MPI processes.

        """
        self.function = function
        self.name = name

    def __call__(self, *args, **kwargs):
        """Evaluate the function."""
        return self.function(*args, **kwargs)

    def __getstate__(self):
        """Pickle the function wrapper to be sent accross MPI processes.

        Notably, the function itself is not pickled. This needs to, instead, be
        re-loaded locally in each MPI process after unpickling.
        """
        return dict(function=None, name=self.name)

    def __setstate__(self, state):
        """Restore the function wrapper."""
        self.__dict__ = state


class _stop_wait_message(object):
    def __repr__(self):
        return "<Stop wait message>"


def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what "
                       "function to apply.")


class MPIPool(object):
    """An MPI pool capable of mapping functions."""

    def __init__(self, comm=None):
        try:
            from mpi4py import MPI
            self.MPI = MPI
        except ImportError:
            raise RuntimeError("MPI environment not found!")
        if comm is None:
            comm = self.MPI.COMM_WORLD
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.function = _error_function
        self.registry = dict()

    def save_function(self, function, name):
        """Save a function to the local registry.

        Parameters
        ----------
        function : function
            Function to save locally in the pool.
        name : str
            Name to be used. This is later used to identify and restore the
            function after the function wrapper is pickled and send across
            MPI processes.

        Returns
        -------
        FunctionWrapper
            Wrapper around the function that can be called normally. If later
            passed to :meth:`map`, the wrapped function is not send accross
            processes and instead loaded locally from the registry.

        """
        self.registry[name] = function
        return FunctionWrapper(function, name)

    def load_function(self, obj):
        """Load registered functions onto an object.

        This function scans the object and finds all ``FunctionWrapper``
        instances and loads the function from the local registry, avoiding
        sending functions across processes.

        Parameters
        ----------
        obj : object
            Object that may contain ``FunctionWrapper`` instances.

        """
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
        """Check if pool is main MPI process."""
        return self.rank == 0

    def wait(self):
        """Wait for instructions. Should only be used by worker processes."""
        if self.main:
            raise RuntimeError("Main node told to await jobs")
        status = self.MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=self.MPI.ANY_TAG,
                                  status=status)

            if isinstance(task, _stop_wait_message):
                return
            elif callable(task):
                # Load functions from the local registry.
                self.load_function(task)
                self.function = task
                continue
            else:
                # Some packages, e.g., dynesty, may package functions inside
                # the arguments.
                self.load_function(task)
                results = list(map(self.function, task))
                self.comm.send(results, dest=0, tag=status.tag)

    def stop_wait(self):
        """Signal to worker processes to stop waiting."""
        if self.main:
            for i in range(1, self.size):
                self.comm.isend(_stop_wait_message(), dest=i)

    def map(self, function, tasks):
        """Apply a function to a list of tasks across MPI processes..

        Parameters
        ----------
        function : callable
            Function to be evaluated.
        tasks : iterable
            List of tasks or arguments passed to the function.

        Returns
        -------
        results : list
            List of results.

        Notes
        -----
        This should only be called from the main process. Worker (non-main)
        process should instead call :meth:`wait`.

        """
        if not self.main:
            self.wait()
            return

        tasks = list(tasks)
        # Send function if necessary.
        if function is not self.function:
            self.function = function
            requests = [self.comm.send(function, dest=i) for i in
                        range(1, self.size)]

        # Distribute tasks to workers.
        requests = []
        for i in range(1, self.size):
            req = self.comm.send(tasks[i::self.size], dest=i)
            requests.append(req)

        # Process local work.
        results = [None]*len(tasks)
        results[::self.size] = list(map(self.function, tasks[::self.size]))

        # Recover results from workers (in any order).
        status = self.MPI.Status()
        for i in range(self.size - 1):
            result = self.comm.recv(source=self.MPI.ANY_SOURCE,
                                    status=status)
            results[status.source::self.size] = result
        return results
