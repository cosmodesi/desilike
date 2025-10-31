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


class FunctionWrapper:

    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __getstate__(self):
        return dict(function=None, name=self.name)

    def __setstate__(self, state):
        self.__dict__ = state


class _close_pool_message(object):
    def __repr__(self):
        return "<Close pool message>"


class _stop_wait_message(object):
    def __repr__(self):
        return "<Stop wait message>"


def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what "
                       "function to apply.")


class MPIPool(object):
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
        self.library = dict()

    def cache_function(self, function, name):
        self.library[name] = function
        return FunctionWrapper(function, name)

    @property
    def main(self):
        return self.rank == 0

    def wait(self):
        if self.main:
            raise RuntimeError("Main node told to await jobs")
        status = self.MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=self.MPI.ANY_TAG,
                                  status=status)

            if isinstance(task, _close_pool_message):
                break
            elif isinstance(task, _stop_wait_message):
                return
            elif callable(task):
                self.function = task
                continue
            elif isinstance(task, str):
                self.function = self.library[task]
                continue
            else:
                results = list(map(self.function, task))
                self.comm.send(results, dest=0, tag=status.tag)

    def map(self, function, tasks):
        # Should be called by the main only.
        if not self.main:
            self.wait()
            return

        tasks = list(tasks)
        # Send function if necessary.
        if function is not self.function:

            # Check if function is cached.
            if isinstance(function, FunctionWrapper):
                function = function.name
            for attr in ['f', 'func']:
                try:
                    if isinstance(getattr(function, attr), FunctionWrapper):
                        function = getattr(function, attr).name
                except AttributeError:
                    pass

            if isinstance(function, str):
                self.function = self.library[function]
            else:
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

    def close(self):
        if self.main:
            for i in range(1, self.size):
                self.comm.isend(_close_pool_message(), dest=i)

    def stop_wait(self):
        if self.main:
            for i in range(1, self.size):
                self.comm.isend(_stop_wait_message(), dest=i)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
