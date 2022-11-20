"""Utilities for confidence level conversions."""

import math

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


def std_notation(value, sigfigs, positive_sign=False):
    """
    Standard notation (US version).
    Return a string corresponding to value with the number of significant digits ``sigfigs``.

    >>> std_notation(5, 2)
    '5.0'
    >>> std_notation(5.36, 2)
    '5.4'
    >>> std_notation(5360, 2)
    '5400'
    >>> std_notation(0.05363, 3)
    '0.0536'

    Created by William Rusnack:
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    return ('-' if is_neg else '+' if positive_sign else '') + _place_dot(sig_digits, power)


def sci_notation(value, sigfigs, filler='e', positive_sign=False):
    """
    Scientific notation.

    Return a string corresponding to value with the number of significant digits ``sigfigs``,
    with 10s exponent filler ``filler`` placed between the decimal value and 10s exponent.

    >>> sci_notation(123, 1, 'e')
    '1e2'
    >>> sci_notation(123, 3, 'e')
    '1.23e2'
    >>> sci_notation(0.126, 2, 'e')
    '1.3e-1'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    dot_power = min(-(sigfigs - 1), 0)
    ten_power = power + sigfigs - 1
    return ('-' if is_neg else '+' if positive_sign else '') + _place_dot(sig_digits, dot_power) + filler + str(ten_power)


def _place_dot(digits, power):
    """
    Place dot in the correct spot, given by integer ``power`` (starting from the right of ``digits``)
    in the string ``digits``.
    If the dot is outside the range of the digits zeros will be added.

    >>> _place_dot('123', 2)
    '12300'
    >>> _place_dot('123', -2)
    '1.23'
    >>> _place_dot('123', 3)
    '0.123'
    >>> _place_dot('123', 5)
    '0.00123'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    if power > 0: out = digits + '0' * power

    elif power < 0:
        power = abs(power)
        sigfigs = len(digits)

        if power < sigfigs:
            out = digits[:-power] + '.' + digits[-power:]

        else:
            out = '0.' + '0' * (power - sigfigs) + digits

    else:
        out = digits + ('.' if digits[-1] == '0' else '')

    return out


def _number_profile(value, sigfigs):
    """
    Return elements to turn number into string representation.

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Parameters
    ----------
    value : float
        Number.

    sigfigs : int
        Number of significant digits.

    Returns
    -------
    sig_digits : string
        Significant digits.

    power : int
        10s exponent to get the dot to the proper location in the significant digits

    is_neg : bool
        ``True`` if value is < 0 else ``False``
    """
    if value == 0:
        sig_digits = '0' * sigfigs
        power = -(1 - sigfigs)
        is_neg = False

    else:
        is_neg = value < 0
        if is_neg: value = abs(value)

        power = -1 * math.floor(math.log10(value)) + sigfigs - 1
        sig_digits = str(int(round(abs(value) * 10.0**power)))

    return sig_digits, int(-power), is_neg


def round_measurement(x, u=0.1, v=None, sigfigs=2, positive_sign=False, notation='auto'):
    """
    Return string representation of input central value ``x`` with uncertainties ``u`` and ``v``.

    Parameters
    ----------
    x : float
        Central value.

    u : float, default=0.1
        Upper uncertainty on ``x`` (positive).

    v : float, default=None
        Lower uncertainty on ``v`` (negative).
        If ``None``, only returns string representation for ``x`` and ``u``.

    sigfigs : int, default=2
        Number of digits to keep for the uncertainties (hence fixing number of digits for ``x``).

    Returns
    -------
    xr : string
        String representation for central value ``x``.

    ur : string
        String representation for upper uncertainty ``u``.

    vr : string
        If ``v`` is not ``None``, string representation for lower uncertainty ``v``.
    """
    x, u = float(x), float(u)
    return_v = True
    if v is None:
        return_v = False
        v = -abs(u)
    else:
        v = float(v)
    if x == 0. or not np.isfinite(u): logx = 0
    else: logx = math.floor(math.log10(abs(x)))
    if u == 0. or not np.isfinite(u): logu = logx
    else: logu = math.floor(math.log10(abs(u)))
    if v == 0. or not np.isfinite(u): logv = logx
    else: logv = math.floor(math.log10(abs(v)))
    if x == 0.: logx = max(logu, logv)

    def round_notation(val, sigfigs, notation='auto', positive_sign=False):
        if not np.isfinite(val):
            return str(val)
        if notation == 'auto':
            # if 1e-3 < abs(val) < 1e3 or center and (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
            if (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
                notation = 'std'
            else:
                notation = 'sci'
        notation_dict = {'std': std_notation, 'sci': sci_notation}

        if notation in notation_dict:
            return notation_dict[notation](val, sigfigs=sigfigs, positive_sign=positive_sign)
        return notation(val, sigfigs=sigfigs, positive_sign=positive_sign)

    if logv > logu:
        sigfigs = (logx - logu + sigfigs, sigfigs, logv - logu + sigfigs)
    else:
        sigfigs = (logx - logv + sigfigs, logu - logv + sigfigs, sigfigs)

    xr = round_notation(x, sigfigs=sigfigs[0], notation=notation, positive_sign=bool(positive_sign) and positive_sign != 'u')
    ur = round_notation(u, sigfigs=sigfigs[1], notation=notation, positive_sign=bool(positive_sign))
    vr = round_notation(v, sigfigs=sigfigs[2], notation=notation, positive_sign=bool(positive_sign))

    if return_v: return xr, ur, vr
    return xr, ur
