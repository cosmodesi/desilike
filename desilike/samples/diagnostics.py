import logging
import warnings

import numpy as np

from desilike.parameter import is_parameter_sequence
from . import utils


logger = logging.getLogger('Diagnostics')


def gelman_rubin(chains, params=None, nsplits=None, statistic='mean', method='eigen', return_matrices=False, check_valid='raise'):
    """
    Estimate Gelman-Rubin statistics, which compares covariance of chain means to (mean of) intra-chain covariances.

    Parameters
    ----------
    chains : list, Chain
        List of or single :class:`Chain` instance(s).

    params : list, ParameterCollection
        Parameters to compute Gelman-Rubin statistics for.
        Defaults to all parameters.

    nsplits : int, default=None
        The Gelman-Rubin criterion requires at least 2 chains.
        If provided, split input chains into ``nsplits`` parts.

    statistic : str, callable, default='mean'
        If 'mean', compares covariance of chain means to (mean of) intra-chain covariances.
        Else, must be a callable taking :class:`Chain` instance and parameter list as input
        and returning array of values (one for each parameter).

    method : str, default='eigen'
        If 'eigen', return eigenvalues of covariance ratios, else diagonal.

    return_matrices : bool, default=False
        If ``True``, also return pair of covariance matrices (of chain means and (mean of) intra-chain covariances).

    check_valid : str, default='raise'
        If inversion of intra-chain covariances is inaccurate, and ``check_valid`` is:

        - 'raise': raise a :class:`LinAlgError`
        - 'warn': issue a warning
        - 'ignore': ignore

    Returns
    -------
    gr : scalar, array, tuple
        Gelman-Rubin statistics (scalar if single parameter provided, else array of size ``params``).
        If ``return_matrices``, also return pair covariance matrices.

    Reference
    ---------
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
    """
    if not utils.is_sequence(chains):
        chains = [chains]
    nchains = len(chains)
    if nchains < 2:
        if nsplits is None or nchains * nsplits < 2:
            raise ValueError('Provide a list of at least 2 chains to estimate Gelman-Rubin, or specify nsplits >= {:d}'.format(int(2. / nchains + 0.5)))
        chains = [chain[islab * len(chain) // nsplits:(islab + 1) * len(chain) // nsplits] for islab in range(nsplits) for chain in chains]
    sizes = [chain.size for chain in chains]
    if any(size < 2 for size in sizes):
        raise ValueError('Not enough samples ({}) to estimate Gelman-Rubin'.format(sizes))
    if params is None: params = chains[0].params(varied=True)
    isscalar = not is_parameter_sequence(params)
    if isscalar: params = [params]
    nchains = len(chains)

    if statistic == 'mean':

        def statistic(chain, params):
            return [chain.mean(param) for param in params]

    means = np.asarray([statistic(chain, params) for chain in chains])
    covs = np.asarray([chain.covariance(params) for chain in chains])
    wsums = np.asarray([chain.weight.sum() for chain in chains])
    w2sums = np.asarray([(chain.weight * chain.aweight).sum() for chain in chains])
    # W = "within"
    Wn1 = np.average(covs, weights=wsums, axis=0)
    Wn = np.average(((wsums - w2sums / wsums) / wsums)[:, None, None] * covs, weights=wsums, axis=0)
    # B = "between"
    # We do not weight with the number of samples in the chains here:
    # shorter chains will likely be outliers, and we want to notice them
    B = np.cov(means.T, ddof=1)
    V = Wn + (nchains + 1.) / nchains * B
    if method == 'eigen':
        # Divide by stddev for numerical stability
        stddev = np.sqrt(np.diag(V).real)
        V = V / stddev[:, None] / stddev[None, :]
        invWn1 = utils.inv(Wn1 / stddev[:, None] / stddev[None, :], check_valid=check_valid)
        if invWn1 is None:
            raise ValueError('cannot compute inverse')
        try:
            toret = np.linalg.eigvalsh(invWn1.dot(V))
        except np.linalg.LinAlgError as exc:
            raise ValueError from exc
    else:
        toret = np.diag(V) / np.diag(Wn1)
    if isscalar:
        toret = toret[0]
    if return_matrices:
        return toret, (V, Wn1)
    return toret


def autocorrelation(chains, params=None):
    """
    Estimate weighted autocorrelation.
    Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    chains : list, Chain
        List of or single :class:`Chain` instance(s).

    params : list, ParameterCollection
        Parameters to compute autocorrelation statistics for.
        Defaults to all parameters.

    Returns
    -------
    autocorr : 1D or 2D array
        Autocorrelation (array of size the number of samples if single parameter provided, else of shape (number of parameters, samples)).
    """
    if not utils.is_sequence(chains):
        chains = [chains]

    if params is None: params = chains[0].params(varied=True)
    if is_parameter_sequence(params):
        return np.array([autocorrelation(chains, param) for param in params])

    toret = 0
    for chain in chains:
        value = chain[params].ravel()
        weight = chain.weight.ravel()
        x = (value - np.average(value, weights=weight)) * weight
        toret += _autocorrelation_1d(x)
    return toret / len(chains)


def integrated_autocorrelation_time(chains, params=None, criterion='sokal', min_corr=None, reliable=50, check_valid='warn', **kwargs):
    r"""
    Estimate integrated autocorrelation time (averaged over all chains).
    Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py
    and https://github.com/blackjax-devs/blackjax/blob/main/blackjax/diagnostics.py
    Effective sample size (ESS) can be computed as (number of samples) / (integrated autocorrelation time).

    Parameters
    ----------
    chains : list, Chain
        List of or single :class:`Chain` instance(s).

    params : list, ParameterCollection
        Parameters to compute integrated autocorrelation time for.
        Defaults to all parameters.

    criterion : str, default='sokal'
        Criterion to stop (:math:`N`) the integration of autocorrelation time:

        .. math::

            \hat{\tau} = -1 + 2 \sum_{t=0}^{N} \hat{\rho}_{t}

        If 'min_corr', maximum index :math:`N` for which :math:`\hat{\rho}_{N} > \mathrm{min_corr}`. ``min_corr`` can be provided in ``kwargs``.
        If 'sokal', minimum index :math:`N` for which :math:`N > C \hat{\rho}_{N}`. . ``c`` can be provided in ``kwargs``, defaults to 5.
        If 'geyer', maximum index :math:`N` for which :math:`\hat{\rho}_{2N} + \hat{\rho}_{2N + 1} > 0`.

    min_corr : float, default=None
        Integrate starting from this lower autocorrelation threshold.
        If ``None``, use ``c``.

    c : float, int, default=5
        Step size for the window search.

    reliable : float, int, default=50
        Minimum ratio between the chain length and estimated autocorrelation time
        for it to be considered reliable.

    check_valid : bool, default=False
        If estimate of autocorrelation time (based on ``reliable``) is not reliable, and ``check_valid`` is:

        - 'raise': raise a :class:`LinAlgError`
        - 'warn': issue a warning
        - 'ignore': ignore

    Returns
    -------
    iat : scalar, array
        Integrated autocorrelation time (scalar if single parameter provided, else array of size ``params``).
    """
    if not utils.is_sequence(chains):
        chains = [chains]

    if params is None: params = chains[0].params(varied=True)

    if is_parameter_sequence(params):
        return np.array([integrated_autocorrelation_time(chains, param, criterion=criterion, reliable=reliable, check_valid=check_valid, **kwargs) for param in params])

    # Automated windowing procedure following Sokal (1989)
    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    sizes = [chain.size for chain in chains]
    if not all(size == sizes[0] for size in sizes):
        raise ValueError('Input chains must have same length, found {}'.format(sizes))
    if any(size < 2 for size in sizes):
        raise ValueError('Not enough samples ({}) to estimate autocorrelation time'.format(sizes))

    size = chains[0].size
    corr = autocorrelation(chains, params)
    toret = None
    if criterion == 'min_corr':
        min_corr = kwargs.get('min_corr', 0.)
        mask = corr > min_corr * corr[0] # 1's, then 0's
        ix = len(corr) if mask.all() else np.argmin(mask)
        ix = np.argmin(corr > min_corr * corr[0])  # 1 + 2 sum_{i=1}^{N} f_{i} as corr[0] = 1.
        toret = 2 * np.sum(corr[:ix]) - 1.
    elif criterion == 'sokal':
        c = kwargs.get('c', 5.)
        taus = 2 * np.cumsum(corr) - 1.  # 1 + 2 sum_{i=1}^{N} f_{i} as corr[0] = 1.
        window = auto_window(taus, c)
        toret = taus[window]
    elif criterion == 'geyer':
        size_even = size - size % 2
        corr = corr[:size_even]
        corr_even = corr[0::2]
        corr_odd = corr[1::2]
        corr_sum = corr_even + corr_odd
        mask = corr_sum > 0.
        mask = np.ones_like(corr_sum, dtype='?')
        ix = np.argmin(mask)
        mask[:ix] = False
        corr_odd[mask] = 0.
        if ix < len(mask): mask[ix] = corr_even[ix] <= 0.
        corr_even[mask] = 0.
        corr_sum = corr_even + corr_odd
        updated = np.minimum.accumulate(corr_sum)
        corr_even[corr_sum > updated] = updated / 2.
        corr_odd[corr_sum > updated] = updated / 2.
        corr_sum = corr_even + corr_odd
        toret = 2 * np.sum(corr_sum) - 1. - corr_even[ix]
    else:
        raise ValueError('could not understand {}; criterion must be provided to stop integration of correlation time'.format(criterion))
    if reliable * toret > size:
        msg = 'The chain is shorter than {:d} times the integrated autocorrelation time for {}. Use this estimate with caution and run a longer chain!\n'.format(reliable, params)
        msg += 'N/{:d} = {:.0f};\ntau: {}'.format(reliable, size / reliable, toret)
        if check_valid == 'raise':
            raise ValueError(msg)
        elif check_valid == 'warn':
            warnings.warn(msg)
        elif check_valid != 'ignore':
            raise ValueError('check_valid must be one of ["raise", "warn", "ignore"]')
    return toret


def _autocorrelation_1d(x):
    """
    Estimate the normalized autocorrelation function.
    Taken from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    x : array
        1D time series.

    Returns
    -------
    acf : array
        The autocorrelation function of the time series.
    """
    from numpy import fft
    x = np.atleast_1d(x)
    if x.ndim != 1:
        raise ValueError('Invalid dimensions for 1D autocorrelation function, found {:d}'.format(x.ndim))
    if x.size < 2:
        raise ValueError('Not enough samples to estimate autocorrelation, found {:d}'.format(x.size))

    n = 2**(2 * len(x) - 1).bit_length()

    # Compute the FFT and then (from that) the auto-correlation function
    f = fft.fft(x, n=n)
    acf = fft.ifft(f * np.conjugate(f))[:len(x)].real

    acf /= acf[0]
    return acf


def geweke(chains, params=None, first=0.1, last=0.5):
    """
    Estimate Geweke statistics, i.e. the difference of chain averages in the first and last samples,
    w.r.t. the sum of covariances in the first and last samples.

    Parameters
    ----------
    chains : list, Chain
        List of or single :class:`Chain` instance(s).

    params : list, ParameterCollection
        Parameters to compute Geweke statistics for.
        Defaults to all parameters.

    first : float, default=0.1
        Fraction of samples in the first part of the chain.

    last : float, default=0.5
        Fraction of samples in the last part of the chain.

    Returns
    -------
    geweke : 2D or 1D array
        Geweke statistics (array of size number of chains if single parameter provided, else array of shape (number of parameters, number of chains)).
    """
    if not utils.is_sequence(chains):
        chains = [chains]

    if params is None: params = chains[0].params(varied=True)
    if is_parameter_sequence(params):
        return np.array([geweke(chains, param, first=first, last=last) for param in params])

    toret = []
    for chain in chains:
        # params is single param
        value, aweight, fweight = chain[params].ravel(), chain.aweight.ravel(), chain.fweight.ravel()
        ifirst, ilast = int(first * value.size + 0.5), int(last * value.size + 0.5)
        value_first, value_last = value[:ifirst], value[ilast:]
        if value_first.size < 2 or value_last.size < 2:
            raise ValueError('Not enough samples ({:d}) to estimate geweke'.format(value.size))
        aweight_first, aweight_last = aweight[:ifirst], aweight[ilast:]
        fweight_first, fweight_last = fweight[:ifirst], fweight[ilast:]
        diff = np.abs(np.average(value_first, weights=aweight_first * fweight_first) - np.average(value_last, weights=aweight_last * fweight_last))
        # np.cov is 0-d
        diff /= (np.cov(value_first, aweights=aweight_first, fweights=fweight_first) + np.cov(value_last, aweights=aweight_last, fweights=fweight_last))**0.5
        toret.append(diff)

    return np.array(toret)
