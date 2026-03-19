"""Module implementing diagnostics for Markov chains."""

import numpy as np
from scipy.signal import correlate

from .samples import Samples


def chains_to_array(chains, keys=None):
    """Convert a ``desilike.Samples`` object or list thereof into arrays.

    Parameters
    ----------
    chains : desilike.Samples or list of desilike.Samples
        Chains to convert.
    keys : list of str, optional
        Keys to extract. If ``None``, use all keys in the first chain. Default
        is ``None``.

    Returns
    -------
    data : numpy.ndarray of shape (n_chains, n_steps, n_dim)
        Chains as numpy array.
    keys : list
        List of keys used.

    Raises
    ------
    ValueError
        If trying to access a key for which the data is higher-dimensional.

    """
    if isinstance(chains, Samples):
        chains = [chains]

    if keys is None:
        keys = chains[0].keys

    n_chains = len(chains)
    n_steps = len(chains[0])
    n_dim = len(keys)
    data = np.zeros((n_chains, n_steps, n_dim))

    for i, chain in enumerate(chains):
        for k, key in enumerate(keys):
            if chain[key].ndim != 1:
                raise ValueError(
                    "Cannot compute diagnostics for higher-dimensional "
                    f"parameter '{key}'.")
            data[i, :, k] = chain[key]

    return data, keys


def autocorrelation_time(chains, keys=None):
    """Estimate the integrated autocorrelation time for Markov chains.

    Autocorrelation times are computed in the same way as in ``emcee``. See
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/ for details.
    While the results have been verified to agree with those from ``emcee``,
    although the implementation is independent.

    Parameters
    ----------
    chains : desilike.Samples, list of desilike.Samples, or numpy.ndarray
        Chains for which to compute the autocorrelation time. If a numpy
        array, the expected shapes are as follows.
            - (n_steps,) if one-dimensional
            - (n_steps, n_dim) if two-dimensional
            - (n_chains, n_steps, n_dim) if three-dimensional
    keys : list of str, optional
        Keys for which to compute the autocorrelation time. Only used if
        ``chains`` is a ``desilike.Samples`` or list thereof. If ``None``, use
        all keys in the chain. Default is ``None``.

    Returns
    -------
    tau : dict, float, or numpy.ndarray
        The estimated autocorrelation times.
            - dict if ``chains`` is a ``desilike.Samples`` or list thereof
            - float if ``chains`` is a one-dimensional array
            - numpy.ndarray otherwise
        In all cases, the autocorrelation function (not time) for each
        parameter is averaged across chains, if multiple chains are provided.

    """
    if isinstance(chains, np.ndarray) and chains.ndim == 1:
        return_type = float
    elif isinstance(chains, np.ndarray):
        return_type = np.ndarray
    else:
        chains, keys = chains_to_array(chains, keys=keys)
        return_type = dict

    if chains.ndim == 1:
        n_steps = len(chains)
        n_chains = 1
        n_dim = 1
    elif chains.ndim == 2:
        n_steps, n_dim = chains.shape
        n_chains = 1
    else:
        n_chains, n_steps, n_dim = chains.shape

    if chains.ndim != 3:
        chains = chains.reshape((n_chains, n_steps, n_dim))

    c = np.zeros_like(chains)

    for i_chain in range(n_chains):
        for i_dim in range(n_dim):
            x = chains[i_chain, :, i_dim]
            x = x - np.mean(x)
            c[i_chain, :, i_dim] = correlate(x, x, mode='full')[len(x) - 1:]
            c[i_chain, :, i_dim] /= c[i_chain, 0, i_dim]

    c = np.mean(c, axis=0)  # average over chains

    tau = 2 * np.cumsum(c, axis=0) - 1  # taus for all possible summing ranges
    stop = (np.arange(n_steps) + 1)[
        :, np.newaxis] > 5 * tau  # +1 to match emcee
    tau = np.where(
        np.any(stop, axis=0),
        tau[np.argmax(stop, axis=0) + 1, np.arange(n_dim)], tau[-1])

    if return_type == float:
        return tau[0]
    elif return_type == dict:
        return dict(zip(keys, tau))
    else:
        return tau
