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
