"""Utilities for confidence level conversions."""


import numpy as np
from scipy import stats

from desilike.utils import *


def is_sequence(item):
    """Whether input item is a tuple or list."""
    from desilike.parameter import ParameterCollection
    return isinstance(item, (list, tuple, ParameterCollection))


def nsigmas_to_quantiles_1d(nsigmas):
    r"""
    Turn number of Gaussian sigmas ``nsigmas`` into quantiles,
    e.g. :math:`\simeq 0.68` for :math:`1 \sigma`.
    """
    return stats.norm.cdf(nsigmas, loc=0, scale=1) - stats.norm.cdf(-nsigmas, loc=0, scale=1)


def nsigmas_to_quantiles_1d_sym(nsigmas):
    r"""
    Turn number of Gaussian sigmas ``nsigmas`` into lower and upper quantiles,
    e.g. :math:`\simeq 0.16, 0.84` for :math:`1 \sigma`.
    """
    total = nsigmas_to_quantiles_1d(nsigmas)
    out = (1. - total) / 2.
    return out, 1. - out


def nsigmas_to_deltachi2(nsigmas, ddof=1):
    r"""Turn number of Gaussian sigmas ``nsigmas`` into :math:`\chi^{2}` levels at ``ddof`` degrees of freedom."""
    if ddof == 1:
        return np.array(nsigmas, dtype='f8')**2
    quantile = nsigmas_to_quantiles_1d(nsigmas)
    return stats.chi2.ppf(quantile, ddof)  # inverse of cdf


def outputs_to_latex(name):
    """Turn outputs ``name`` to latex string."""
    toret = txt_to_latex(name)
    for full, symbol in [('loglikelihood', 'L'), ('logposterior', '\\mathcal{L}'), ('logprior', 'p')]:
        toret = toret.replace(full, symbol)
    return toret


def interval(samples, weights=None, nsigmas=1.):
    """
    Return n-sigmas confidence interval(s).

    Parameters
    ----------
    columns : list, ParameterCollection, default=None
        Parameters to compute confidence interval for.

    nsigmas : int
        Return interval for this number of sigmas.

    Returns
    -------
    interval : tuple
    """
    if weights is None:
        weights = np.ones_like(samples)
    idx = np.argsort(samples)
    x = samples[idx]
    weights = weights[idx]
    nquantile = nsigmas_to_quantiles_1d(nsigmas)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    cdfpq = cdf + nquantile
    ixmaxup = np.searchsorted(cdf, cdfpq, side='left')
    mask = ixmaxup < len(x)
    if not mask.any():
        raise ValueError('Not enough samples ({:d}) for interval estimation'.format(x.size))
    indices = np.array([np.flatnonzero(mask), ixmaxup[mask]])
    xmin, xmax = x[indices]
    argmin = np.argmin(xmax - xmin)
    return (xmin[argmin], xmax[argmin])


def weighted_quantile(x, q, weights=None, axis=None, method='linear'):
    """
    Compute the q-th quantile of the weighted data along the specified axis.

    Parameters
    ----------
    a : array
        Input array or object that can be converted to an array.

    q : tuple, list, array
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.

    weights : array, default=None
        An array of weights associated with the values in ``a``. Each value in
        ``a`` contributes to the cumulative distribution according to its associated weight.
        The weights array can either be 1D (in which case its length must be
        the size of ``a`` along the given axis) or of the same shape as ``a``.
        If ``weights=None``, then all data in ``a`` are assumed to have a
        weight equal to one.
        The only constraint on ``weights`` is that ``sum(weights)`` must not be 0.

    axis : int, tuple, default=None
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.

    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default='linear'
        This optional parameter specifies the method method to
        use when the desired quantile lies between two data points
        ``i < j``:

        * linear: ``i + (j - i) * fraction``, where ``fraction``
          is the fractional part of the index surrounded by ``i``
          and ``j``.
        * lower: ``i``.
        * higher: ``j``.
        * nearest: ``i`` or ``j``, whichever is nearest.
        * midpoint: ``(i + j) / 2``.

    Returns
    -------
    quantile : scalar, array
        If ``q`` is a single quantile and ``axis=None``, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of ``a``. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If ``out`` is specified, that array is
        returned instead.

    Note
    ----
    Inspired from https://github.com/minaskar/cronus/blob/master/cronus/plot.py.
    """
    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.quantile(x, q, axis=axis, method=method)

    # Initial check.
    x = np.atleast_1d(x)
    isscalar = np.ndim(q) == 0
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.) or np.any(q > 1.):
        raise ValueError('Quantiles must be between 0. and 1.')

    if axis is None:
        axis = range(x.ndim)

    if np.ndim(axis) == 0:
        axis = (axis,)

    if weights.ndim > 1:
        if x.shape != weights.shape:
            raise ValueError('Dimension mismatch: shape(weights) != shape(x).')

    x = np.moveaxis(x, axis, range(x.ndim - len(axis), x.ndim))
    x = x.reshape(x.shape[:-len(axis)] + (-1,))
    if weights.ndim > 1:
        weights = np.moveaxis(weights, axis, range(x.ndim - len(axis), x.ndim))
        weights = weights.reshape(weights.shape[:-len(axis)] + (-1,))
    else:
        reps = x.shape[:-1] + (1,)
        weights = np.tile(weights, reps)

    idx = np.argsort(x, axis=-1)  # sort samples
    x = np.take_along_axis(x, idx, axis=-1)
    sw = np.take_along_axis(weights, idx, axis=-1)  # sort weights
    cdf = np.cumsum(sw, axis=-1)  # compute CDF
    cdf = cdf[..., :-1]
    cdf = cdf / cdf[..., -1][..., None]  # normalize CDF
    zeros = np.zeros_like(cdf, shape=cdf.shape[:-1] + (1,))
    cdf = np.concatenate([zeros, cdf], axis=-1)  # ensure proper span
    idx0 = np.apply_along_axis(np.searchsorted, -1, cdf, q, side='right') - 1
    idx1 = np.clip(idx0 + 1, None, x.shape[-1] - 1)
    q0, q1 = x[..., [idx0, idx1]]
    cdf0, cdf1 = cdf[..., [idx0, idx1]]
    if method == 'lower':
        quantiles = q0
    elif method == 'higher':
        quantiles = q1
    elif method == 'nearest':
        mask_lower = q - cdf0 < cdf1 - q
        quantiles = q1
        # in place, q1 not used in the following
        quantiles[mask_lower] = q0[mask_lower]
    elif method == 'linear':
        step = cdf1 - cdf0
        diff = q - cdf0
        mask = idx1 == idx0
        step[mask] = diff[mask]
        fraction = diff / step
        quantiles = q0 + fraction * (q1 - q0)
    elif method == 'midpoint':
        quantiles = (q0 + q1) / 2.
    else:
        raise ValueError('"method" must be one of ["linear", "lower", "higher", "midpoint", "nearest"]')
    quantiles = quantiles.swapaxes(-1, 0)
    if isscalar:
        return quantiles[0]
    return quantiles
