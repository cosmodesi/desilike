import os
import sys

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0


def pytest_configure(config):
    """Suppress output from worker processes."""
    if rank > 0:
        sys.stdout = open(os.devnull, 'w')
