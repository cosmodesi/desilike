class BaseTaskManager(object):
    """A dumb task manager, that simply iterates through the tasks in series."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __enter__(self):
        """Return self."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Do nothing."""

    def iterate(self, tasks):
        """
        Iterate through a series of tasks.

        Parameters
        ----------
        tasks : iterable
            An iterable of tasks that will be yielded.

        Yields
        -------
        task :
            The individual items of ```tasks``, iterated through in series.
        """
        for task in tasks:
            yield task

    def map(self, function, tasks):
        """
        Apply a function to all of the values in a list and return the list of results.

        If ``tasks`` contains tuples, the arguments are passed to
        ``function`` using the ``*args`` syntax.

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
        return [function(*(t if isinstance(t, tuple) else (t,))) for t in tasks]


def TaskManager(mpicomm=None, **kwargs):
    """
    Switch between non-MPI (ntasks=1) and MPI task managers. To be called as::

        with TaskManager(...) as tm:
            # do stuff

    """
    if mpicomm is None or mpicomm.size == 1:
        cls = BaseTaskManager
    else:
        from . import mpi
        cls = mpi.MPITaskManager

    self = cls.__new__(cls)
    self.__init__(mpicomm=mpicomm, **kwargs)
    return self


# Taken from https://stackoverflow.com/questions/44313620/converting-to-and-from-numpys-np-random-randomstate-and-pythons-random-random
PY_VERSION = 3
NP_VERSION = 'MT19937'


def numpy_to_python_random_state(npstate):
    """
    Convert state of a :class:`numpy.random.RandomState` object to a state
    that can be used by Python's :mod:`random`.
    """
    version, keys, pos, has_gauss, cached_gaussian = npstate
    pystate = (
        PY_VERSION,
        tuple(map(int, keys)) + (int(pos),),
        cached_gaussian if has_gauss else None,
    )
    return pystate


def python_to_numpy_random_state(pystate):
    """
    Convert state of a Python's :mod:`random` object to a state
    that can be used by :class:`numpy.random.RandomState`.
    """
    version, (*keys, pos), cached_gaussian = pystate
    has_gauss = cached_gaussian is not None
    npstate = (
        NP_VERSION,
        keys,
        pos,
        has_gauss,
        cached_gaussian if has_gauss else 0.0
    )
    return npstate
