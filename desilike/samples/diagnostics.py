"""MCSamples diagnostics: Gelman-Rubin, autocorrelation, and Geweke statistics.

Adapted from desilike_bak/desilike/samples/diagnostics.py.
"""

import logging
import warnings

import numpy as np

from .samples import _vals, _normalise_params


logger = logging.getLogger('Diagnostics')


# ── internal helpers ──────────────────────────────────────────────────────────

def _is_chain_sequence(chains):
    """Return ``True`` if *chains* is a list/tuple of chains."""
    return isinstance(chains, (list, tuple))


def _inv(mat, check_valid='raise'):
    """Return the inverse of a 2-D matrix with optional accuracy check.

    Parameters
    ----------
    mat : array_like
        Square 2-D matrix.
    check_valid : str, default='raise'
        What to do when :math:`mat \\cdot mat^{-1}` deviates from the identity:
        ``'raise'``, ``'warn'``, or ``'ignore'``.

    Returns
    -------
    invmat : array
        Inverse of *mat*, or ``None`` if the inversion failed and
        ``check_valid`` is not ``'raise'``.
    """
    mat = np.asarray(mat)
    if mat.ndim == 0:
        return 1.0 / mat
    invmat = None
    try:
        invmat = np.linalg.inv(mat)
    except np.linalg.LinAlgError as exc:
        if check_valid == 'raise':
            raise exc
        elif check_valid == 'warn':
            warnings.warn('Matrix inversion failed: {}'.format(exc))
        elif check_valid != 'ignore':
            raise ValueError('check_valid must be one of ["raise", "warn", "ignore"]')
        return None

    # Accuracy check: mat @ invmat ≈ I
    if check_valid != 'ignore':
        tmp = mat.dot(invmat)
        ref = np.eye(tmp.shape[0], dtype=tmp.dtype)
        if not np.allclose(tmp, ref, rtol=1e-3, atol=1e-3):
            msg = 'Numerically inaccurate inverse matrix, max absolute diff {:.6f}.'.format(np.max(np.abs(tmp - ref)))
            if check_valid == 'raise':
                raise np.linalg.LinAlgError(msg)
            elif check_valid == 'warn':
                warnings.warn(msg)
    return invmat


def _varied_names(chain):
    """Return names of non-derived variables in *chain*."""
    return chain.names(varied=True)


# ── public API ────────────────────────────────────────────────────────────────

def gelman_rubin(chains, params=None, nsplits=None, statistic='mean', method='eigen', return_matrices=False, check_valid='raise'):
    """Estimate Gelman-Rubin statistics.

    Compares the covariance of chain means to the mean of intra-chain
    covariances.  For 2-D chains (shape ``(nsteps, nwalkers)``), GR is computed
    independently for each walker index across all chains and the results are
    averaged over walkers.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
        One or more :class:`~desilike.samples.MCSamples` instances.
    params : str, list of str, optional
        Parameters to include.  Defaults to all varied (non-derived)
        parameters.
    nsplits : int, optional
        If fewer than 2 chains are provided, split each chain into
        *nsplits* parts to produce at least 2 sub-chains.
    statistic : str or callable, default='mean'
        If ``'mean'``, compare chain means.  Otherwise a callable
        ``(chain, params) -> array`` that returns one value per parameter.
    method : str, default='eigen'
        If ``'eigen'``, return eigenvalues of the covariance ratio; if
        ``'diag'``, return diagonal ratios.
    return_matrices : bool, default=False
        If ``True``, also return the pair ``(V, Wn1)`` of covariance
        matrices.
    check_valid : str, default='raise'
        Inversion accuracy policy: ``'raise'``, ``'warn'``, or
        ``'ignore'``.

    Returns
    -------
    gr : scalar or array
        Gelman-Rubin statistics; scalar when *params* is a single string,
        else 1-D array of length ``len(params)``.
    matrices : tuple of array, optional
        Only when *return_matrices* is ``True``: ``(V, Wn1)``.

    Reference
    ---------
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
    """
    if not _is_chain_sequence(chains):
        chains = [chains]

    if any(chain.ndim == 2 for chain in chains):
        nwalkers = chains[0].shape[1]
        gr_per_walker = np.array([
            gelman_rubin(
                [chain[:, walker_idx] for chain in chains],
                params=params, nsplits=nsplits, statistic=statistic,
                method=method, return_matrices=False, check_valid=check_valid,
            )
            for walker_idx in range(nwalkers)
        ])
        return gr_per_walker.mean(axis=0)

    nchains = len(chains)
    if nchains < 2:
        if nsplits is None or nchains * nsplits < 2:
            raise ValueError(
                'Provide at least 2 chains to estimate Gelman-Rubin, or specify '
                'nsplits >= {:d}'.format(int(2.0 / nchains + 0.5))
            )
        chains = [
            chain[islab * len(chain) // nsplits:(islab + 1) * len(chain) // nsplits]
            for islab in range(nsplits)
            for chain in chains
        ]

    sizes = [chain.size for chain in chains]
    if any(size < 2 for size in sizes):
        raise ValueError('Not enough samples ({}) to estimate Gelman-Rubin'.format(sizes))

    if params is None:
        params = _varied_names(chains[0])

    is_scalar, params = _normalise_params(chains[0], params)

    nchains = len(chains)  # may have grown after splitting

    if statistic == 'mean':
        def statistic(chain, params):
            return [chain.mean(param) for param in params]

    means  = np.asarray([statistic(chain, params) for chain in chains])          # (nchains, nparams)
    covs   = np.asarray([np.asarray(chain.covariance(params)) for chain in chains])  # (nchains, nparams, nparams)
    wsums  = np.asarray([chain.weight.sum() for chain in chains])
    w2sums = np.asarray([(chain.weight * chain.aweight).sum() for chain in chains])

    # W = "within-chain" covariance
    Wn1 = np.average(covs, weights=wsums, axis=0)
    Wn  = np.average(
        ((wsums - w2sums / wsums) / wsums)[:, None, None] * covs,
        weights=wsums,
        axis=0,
    )
    # B = "between-chain" covariance (unweighted by length — to detect short-chain outliers)
    B = np.cov(means.T, ddof=1)
    V = Wn + (nchains + 1.0) / nchains * B

    if method == 'eigen':
        # Normalise by std for numerical stability
        stddev  = np.sqrt(np.diag(V).real)
        V_norm  = V / stddev[:, None] / stddev[None, :]
        Wn1_norm = Wn1 / stddev[:, None] / stddev[None, :]
        invWn1  = _inv(Wn1_norm, check_valid=check_valid)
        if invWn1 is None:
            raise ValueError('Cannot compute inverse of within-chain covariance')
        try:
            toret = np.linalg.eigvalsh(invWn1.dot(V_norm))
        except np.linalg.LinAlgError as exc:
            raise ValueError('Eigenvalue decomposition failed') from exc
    else:
        toret = np.diag(V) / np.diag(Wn1)

    if is_scalar:
        toret = toret[0]

    if return_matrices:
        return toret, (V, Wn1)
    return toret


def autocorrelation(chains, params=None):
    """Estimate the weighted autocorrelation function.

    Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
        One or more :class:`~desilike.samples.MCSamples` instances.
    params : str, list of str, optional
        Parameters to compute autocorrelation for.  Defaults to all varied
        (non-derived) parameters.

    Returns
    -------
    autocorr : array
        Normalised autocorrelation; shape ``(nsamples,)`` for a single
        parameter, or ``(nparams, nsamples)`` when *params* is a list.
    """
    if not _is_chain_sequence(chains):
        chains = [chains]

    if params is None:
        params = _varied_names(chains[0])

    if isinstance(params, (list, tuple)):
        return np.array([autocorrelation(chains, param) for param in params])

    # params is a single name string from here on
    toret = 0.0
    for chain in chains:
        value  = _vals(chain, params).ravel()
        weight = chain.weight.ravel()
        x = (value - np.average(value, weights=weight)) * weight
        toret += _autocorrelation_1d(x)
    return toret / len(chains)


def integrated_autocorrelation_time(chains, params=None, criterion='sokal', reliable=50, check_valid='warn', **kwargs):
    r"""Estimate the integrated autocorrelation time (IAT).

    Averaged over all chains.  2-D chains (shape ``(nsteps, nwalkers)``) are
    automatically expanded into ``nwalkers`` independent 1-D chains.
    Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py
    and https://github.com/blackjax-devs/blackjax/blob/main/blackjax/diagnostics.py

    The effective sample size (ESS) is ``(number of samples) / IAT``.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
        One or more :class:`~desilike.samples.MCSamples` instances.
    params : str, list of str, optional
        Parameters to compute IAT for.  Defaults to all varied (non-derived)
        parameters.
    criterion : str, default='sokal'
        Stopping criterion for the autocorrelation sum
        :math:`\\hat{\\tau} = -1 + 2 \\sum_{t=0}^{N} \\hat{\\rho}_{t}`:

        - ``'min_corr'``: largest :math:`N` with
          :math:`\\hat{\\rho}_{N} > \\text{min\\_corr}`.
          Pass ``min_corr`` in *kwargs* (default 0).
        - ``'sokal'``: smallest :math:`N` with :math:`N > c \\hat{\\tau}_{N}`.
          Pass ``c`` in *kwargs* (default 5).
        - ``'geyer'``: largest :math:`N` with
          :math:`\\hat{\\rho}_{2N} + \\hat{\\rho}_{2N+1} > 0`.

    reliable : int, default=50
        Minimum ratio of chain length to IAT for the estimate to be
        considered reliable.
    check_valid : str, default='warn'
        What to do when the chain is shorter than ``reliable * IAT``:
        ``'raise'``, ``'warn'``, or ``'ignore'``.

    Returns
    -------
    iat : scalar or array
        Integrated autocorrelation time; scalar for a single parameter,
        else 1-D array of length ``len(params)``.
    """
    if not _is_chain_sequence(chains):
        chains = [chains]

    flat_chains = []
    for chain in chains:
        if chain.ndim == 2:
            for walker_idx in range(chain.shape[1]):
                flat_chains.append(chain[:, walker_idx])
        else:
            flat_chains.append(chain)
    chains = flat_chains

    if params is None:
        params = _varied_names(chains[0])

    if isinstance(params, (list, tuple)):
        return np.array([
            integrated_autocorrelation_time(
                chains, param,
                criterion=criterion, reliable=reliable,
                check_valid=check_valid, **kwargs,
            )
            for param in params
        ])

    # Single parameter from here on
    def _auto_window(taus, c):
        """Automated windowing procedure following Sokal (1989)."""
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    sizes = [chain.size for chain in chains]
    if not all(size == sizes[0] for size in sizes):
        raise ValueError('All chains must have the same length; found {}'.format(sizes))
    if any(size < 2 for size in sizes):
        raise ValueError('Not enough samples ({}) to estimate IAT'.format(sizes))

    size = chains[0].size
    corr = autocorrelation(chains, params)

    if criterion == 'min_corr':
        min_corr_val = kwargs.get('min_corr', 0.0)
        ix = np.argmin(corr > min_corr_val * corr[0])
        toret = 2.0 * np.sum(corr[:ix]) - 1.0
    elif criterion == 'sokal':
        c_val  = kwargs.get('c', 5.0)
        taus   = 2.0 * np.cumsum(corr) - 1.0
        window = _auto_window(taus, c_val)
        toret  = taus[window]
    elif criterion == 'geyer':
        size_even  = size - size % 2
        corr       = corr[:size_even]
        corr_even  = corr[0::2].copy()
        corr_odd   = corr[1::2].copy()
        corr_sum   = corr_even + corr_odd
        mask       = np.ones_like(corr_sum, dtype=bool)
        ix         = np.argmin(mask)
        mask[:ix]  = False
        corr_odd[mask] = 0.0
        if ix < len(mask):
            mask[ix] = corr_even[ix] <= 0.0
        corr_even[mask] = 0.0
        corr_sum = corr_even + corr_odd
        updated  = np.minimum.accumulate(corr_sum)
        corr_even[corr_sum > updated] = updated[corr_sum > updated] / 2.0
        corr_odd[corr_sum > updated]  = updated[corr_sum > updated] / 2.0
        corr_sum = corr_even + corr_odd
        toret    = 2.0 * np.sum(corr_sum) - 1.0 - corr_even[ix]
    else:
        raise ValueError(
            'Unknown criterion {!r}; must be one of "min_corr", "sokal", "geyer"'.format(criterion)
        )

    if reliable * toret > size:
        msg = (
            'The chain is shorter than {:d} times the integrated autocorrelation '
            'time for {!r}. Use this estimate with caution and run a longer chain!\n'
            'N/{:d} = {:.0f};\ntau: {}'.format(reliable, params, reliable, size / reliable, toret)
        )
        if check_valid == 'raise':
            raise ValueError(msg)
        elif check_valid == 'warn':
            warnings.warn(msg)
        elif check_valid != 'ignore':
            raise ValueError('check_valid must be one of ["raise", "warn", "ignore"]')

    return toret


def _autocorrelation_1d(x):
    """Estimate the normalised autocorrelation function of a 1-D time series.

    Taken from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    x : array_like
        1-D time series (already mean-subtracted and weighted if desired).

    Returns
    -------
    acf : array
        Normalised autocorrelation function of length ``len(x)``.
    """
    from numpy import fft

    x = np.atleast_1d(x)
    if x.ndim != 1:
        raise ValueError(
            'Expected a 1-D array; got shape {}'.format(x.shape)
        )
    if x.size < 2:
        raise ValueError(
            'Need at least 2 samples to compute autocorrelation; got {:d}'.format(x.size)
        )

    # Next power-of-2 length for zero-padding
    n   = 2 ** (2 * len(x) - 1).bit_length()
    f   = fft.fft(x, n=n)
    acf = fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf


def geweke(chains, params=None, first=0.1, last=0.5):
    """Estimate Geweke convergence statistics.

    Tests stationarity by comparing the mean of the first and last
    fractions of each chain relative to the combined spectral variance.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
        One or more :class:`~desilike.samples.MCSamples` instances.
    params : str, list of str, optional
        Parameters to include.  Defaults to all varied (non-derived)
        parameters.
    first : float, default=0.1
        Fraction of samples to take from the start of each chain.
    last : float, default=0.5
        Fraction of samples to take from the end of each chain.

    Returns
    -------
    geweke : array
        Geweke statistics; shape ``(nchains,)`` for a single parameter,
        or ``(nparams, nchains)`` for multiple parameters.
    """
    if not _is_chain_sequence(chains):
        chains = [chains]

    if params is None:
        params = _varied_names(chains[0])

    if isinstance(params, (list, tuple)):
        return np.array([geweke(chains, param, first=first, last=last) for param in params])

    # Single parameter from here on
    toret = []
    for chain in chains:
        value   = _vals(chain, params).ravel()
        aweight = chain.aweight.ravel()
        fweight = chain.fweight.ravel()

        ifirst = int(first * value.size + 0.5)
        ilast  = int(last  * value.size + 0.5)

        value_first,   value_last   = value[:ifirst],   value[ilast:]
        aweight_first, aweight_last = aweight[:ifirst], aweight[ilast:]
        fweight_first, fweight_last = fweight[:ifirst], fweight[ilast:]

        if value_first.size < 2 or value_last.size < 2:
            raise ValueError(
                'Not enough samples ({:d}) to estimate Geweke statistics'.format(value.size)
            )

        mean_first = np.average(value_first, weights=aweight_first * fweight_first)
        mean_last  = np.average(value_last,  weights=aweight_last  * fweight_last)
        diff = np.abs(mean_first - mean_last)
        var_first = np.cov(value_first, aweights=aweight_first, fweights=fweight_first)
        var_last  = np.cov(value_last,  aweights=aweight_last,  fweights=fweight_last)
        diff /= (float(var_first) + float(var_last)) ** 0.5
        toret.append(diff)

    return np.array(toret)
