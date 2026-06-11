"""MCSamples — Samples subclass that adds weights, log-posterior, and statistics."""

import os
import re

import numpy as np
from scipy import stats as _scipy_stats

from ..parameter import Variable, VariableCollection
from ..utils import register_type, round_measurement
from .samples import Samples, _vals, _normalise_params


# ── weighted-statistics helpers ───────────────────────────────────────────────

def _nsigmas_to_quantiles_1d(nsigmas):
    """Fraction of a Gaussian contained within ±nsigmas (e.g. ≈0.68 for 1σ)."""
    return (_scipy_stats.norm.cdf(nsigmas) - _scipy_stats.norm.cdf(-nsigmas))


def _nsigmas_to_quantiles_1d_sym(nsigmas):
    """Lower and upper quantile bounds for a symmetric nsigmas interval."""
    total = _nsigmas_to_quantiles_1d(nsigmas)
    lo = (1. - total) / 2.
    return lo, 1. - lo


def _weighted_quantile(x, q, weights=None, axis=0, method='linear'):
    """Weighted quantile of *x* along *axis*.

    Adapted from https://github.com/minaskar/cronus/blob/master/cronus/plot.py.
    """
    if weights is None:
        return np.quantile(x, q, axis=axis, method=method)

    x = np.asarray(x, dtype=float)
    isscalar = np.ndim(q) == 0
    q = np.atleast_1d(q)
    if np.any(q < 0.) or np.any(q > 1.):
        raise ValueError('Quantiles must be between 0 and 1.')

    ax = (axis,) if np.ndim(axis) == 0 else tuple(axis)
    x = np.moveaxis(x, ax, range(x.ndim - len(ax), x.ndim))
    x = x.reshape(x.shape[:-len(ax)] + (-1,))

    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        reps = x.shape[:-1] + (1,)
        weights = np.tile(weights, reps)
    else:
        weights = np.moveaxis(weights, ax, range(weights.ndim - len(ax), weights.ndim))
        weights = weights.reshape(weights.shape[:-len(ax)] + (-1,))

    idx = np.argsort(x, axis=-1)
    x = np.take_along_axis(x, idx, axis=-1)
    sw = np.take_along_axis(weights, idx, axis=-1)
    cdf = np.cumsum(sw, axis=-1)
    cdf = cdf[..., :-1] / cdf[..., -1:]
    cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], axis=-1)

    idx0 = np.apply_along_axis(np.searchsorted, -1, cdf, q, side='right') - 1
    idx1 = np.clip(idx0 + 1, None, x.shape[-1] - 1)
    # Use take_along_axis so multi-dim x (e.g. shape (15, 200)) is handled correctly.
    q0 = np.take_along_axis(x, idx0, axis=-1)
    q1 = np.take_along_axis(x, idx1, axis=-1)
    cdf0 = np.take_along_axis(cdf, idx0, axis=-1)
    cdf1 = np.take_along_axis(cdf, idx1, axis=-1)

    if method == 'lower':
        quantiles = q0
    elif method == 'higher':
        quantiles = q1
    elif method == 'nearest':
        quantiles = np.where(q - cdf0 < cdf1 - q, q0, q1)
    elif method == 'midpoint':
        quantiles = (q0 + q1) / 2.
    elif method == 'linear':
        step = cdf1 - cdf0
        frac = np.where(step == 0, 0., (q - cdf0) / np.where(step == 0, 1., step))
        quantiles = q0 + frac * (q1 - q0)
    else:
        raise ValueError(f'Unknown method {method!r}')

    quantiles = np.moveaxis(quantiles, -1, 0)
    return quantiles[0] if isscalar else quantiles


def _interval(x, weights=None, nsigmas=1.):
    """Shortest interval containing the nsigmas probability mass (1D)."""
    x = np.asarray(x, dtype=float).ravel()
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.asarray(weights, dtype=float).ravel()
    idx = np.argsort(x)
    x = x[idx]
    weights = weights[idx]
    nquantile = _nsigmas_to_quantiles_1d(nsigmas)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    cdfpq = cdf + nquantile
    ixmaxup = np.searchsorted(cdf, cdfpq, side='left')
    mask = ixmaxup < len(x)
    if not mask.any():
        raise ValueError(f'Not enough samples ({x.size}) for interval estimation')
    lo = x[np.flatnonzero(mask)]
    hi = x[ixmaxup[mask]]
    argmin = np.argmin(hi - lo)
    return float(lo[argmin]), float(hi[argmin])


def _var_impl(chain, params_list, ddof):
    """Core weighted variance — always takes a list, always returns a list."""
    w     = chain.weight.ravel().astype(float)
    W     = w.sum()
    W2    = (w ** 2).sum()
    denom = W - ddof * W2 / W
    results = []
    for p in params_list:
        vals = _vals(chain, p)
        mu   = np.average(vals, weights=w, axis=0)
        results.append(np.average((vals - mu) ** 2, weights=w, axis=0) * W / denom)
    return results


# ── MCSamples ─────────────────────────────────────────────────────────────────────

@register_type
class MCSamples(Samples):
    """Samples subclass that adds log-posterior, weights, and statistical methods.

    Every Variable's ``_value`` is stored as a NumPy array of shape
    ``chain.shape + variable.shape``.  ``chain.shape`` is the leading
    batch of sample dimensions (e.g. ``(n_steps,)`` for a flat chain,
    ``(n_chains, n_steps)`` for a 2-D chain).  ``variable.shape`` is the
    intrinsic per-sample shape (``()`` for scalar parameters, ``(15,)`` for a
    power-spectrum vector, etc.).

    Structure, I/O, and export are inherited from :class:`~desilike.samples.samples.Samples`.

    Examples
    --------
    Building from a dict of arrays::

        c = MCSamples({'omega_m': rng.normal(0.3, 0.01, 1000),
                   'sigma8':  rng.normal(0.8, 0.02, 1000)})
        c.logposterior = -0.5 * rng.chisquare(2, 1000)

    Non-scalar variable (must pass a Variable key so the shape is unambiguous)::

        pk_var = Variable('pk', value=np.zeros(50))   # shape=(50,)
        c[pk_var] = rng.normal(1., 0.1, (1000, 50))

    Slicing, concatenation::

        c_burned = c.remove_burnin(0.2)
        c_all    = MCSamples.concatenate(c1, c2)

    I/O::

        c.write('chain.h5')
        c2 = MCSamples.read('chain.h5')
    """

    _name = 'MCSamples'

    # Names of the special "bookkeeping" variables.
    _logposterior = 'logposterior'
    _logprior = 'logprior'
    _aweight = 'aweight'
    _fweight = 'fweight'
    _weight = 'weight'   # virtual — never stored in _data

    # ── special attributes ────────────────────────────────────────────────────

    @property
    def logposterior(self):
        """Log-posterior array of shape ``self.shape``, or zeros if not set."""
        if VariableCollection.__contains__(self, self._logposterior):
            return np.asarray(VariableCollection.__getitem__(self, self._logposterior)._value)
        return np.zeros(self.shape)

    @logposterior.setter
    def logposterior(self, item):
        v = Variable(self._logposterior, derived=True)
        self.set(v, np.asarray(item))

    @property
    def logprior(self):
        """Log-prior array of shape ``self.shape``, or zeros if not set."""
        if VariableCollection.__contains__(self, self._logprior):
            return np.asarray(VariableCollection.__getitem__(self, self._logprior)._value)
        return np.zeros(self.shape)

    @logprior.setter
    def logprior(self, item):
        v = Variable(self._logprior, derived=True)
        self.set(v, np.asarray(item))

    @property
    def aweight(self):
        """Analytic weight array of shape ``self.shape``, or ones if not set."""
        if VariableCollection.__contains__(self, self._aweight):
            return np.asarray(VariableCollection.__getitem__(self, self._aweight)._value)
        return np.ones(self.shape)

    @aweight.setter
    def aweight(self, item):
        v = Variable(self._aweight, derived=True)
        self.set(v, np.asarray(item))

    @property
    def fweight(self):
        """Frequency (integer) weight array of shape ``self.shape``, or ones if not set."""
        if VariableCollection.__contains__(self, self._fweight):
            return np.asarray(VariableCollection.__getitem__(self, self._fweight)._value)
        return np.ones(self.shape, dtype='i8')

    @fweight.setter
    def fweight(self, item):
        v = Variable(self._fweight, derived=True)
        self.set(v, np.asarray(item))

    @property
    def weight(self):
        """Total weight = ``aweight * fweight``, shape ``self.shape``."""
        return self.aweight * self.fweight

    # ── statistics ────────────────────────────────────────────────────────────

    def remove_burnin(self, burnin=0):
        """Return a new MCSamples with the first *burnin* steps removed.

        Parameters
        ----------
        burnin : int or float
            If in ``(0, 1)`` remove that fraction of the total length.
            Otherwise remove the first *burnin* integer samples (axis 0).
        """
        if 0. < burnin < 1.:
            burnin = int(burnin * len(self) + 0.5)
        return self[int(burnin):]

    def mean(self, params=None):
        """Weighted mean.

        Parameters
        ----------
        params : str, Variable, list, or None
            Parameter(s) to average.  ``None`` → all variables.

        Returns
        -------
        array or list of arrays
            Shape ``variable.shape`` per parameter.
        """
        scalar, params = _normalise_params(self, params)
        w = self.weight.ravel().astype(float)
        results = [np.average(_vals(self, p), weights=w, axis=0) for p in params]
        return results[0] if scalar else results

    def var(self, params=None, ddof=1):
        """Weighted variance (reliability-weights formula), shape ``variable.shape``.

        Parameters
        ----------
        params : str, Variable, list, or None
        ddof : int, default 1
        """
        scalar, params_list = _normalise_params(self, params)
        results = _var_impl(self, params_list, ddof)
        return results[0] if scalar else results

    def std(self, params=None, ddof=1):
        """Weighted standard deviation, shape ``variable.shape``."""
        scalar, params_list = _normalise_params(self, params)
        results = [np.sqrt(r) for r in _var_impl(self, params_list, ddof)]
        return results[0] if scalar else results

    def median(self, params=None, method='linear'):
        """Weighted median."""
        return self.quantile(params=params, q=0.5, method=method)

    def quantile(self, params=None, q=(0.1587, 0.8413), method='linear'):
        """Weighted quantile(s).

        Parameters
        ----------
        params : str, Variable, list, or None
        q : float or sequence of floats
            Quantile(s) in [0, 1].
        method : str
            Interpolation method (see ``numpy.quantile``).

        Returns
        -------
        array or list of arrays
            Shape ``(len(q), *variable.shape)`` if *q* is a sequence,
            otherwise ``variable.shape``.
        """
        scalar, params = _normalise_params(self, params)
        w = self.weight.ravel().astype(float)
        results = [_weighted_quantile(_vals(self, p), q, weights=w, axis=0, method=method)
                   for p in params]
        return results[0] if scalar else results

    def interval(self, params=None, nsigmas=1.):
        """Shortest n-sigma credible interval(s).

        For vector variables the interval is computed element-wise, returning
        two arrays of shape ``variable.shape`` (lower, upper).

        Parameters
        ----------
        params : str, Variable, list, or None
        nsigmas : float

        Returns
        -------
        tuple of arrays ``(low, high)`` or list thereof.
        """
        scalar, params = _normalise_params(self, params)
        w = self.weight.ravel().astype(float)
        results = []
        for p in params:
            vals = _vals(self, p)          # (size, *var.shape)
            vshape = vals.shape[1:]
            if not vshape:
                lo, hi = _interval(vals, weights=w, nsigmas=nsigmas)
                results.append((lo, hi))
            else:
                los = np.empty(vshape)
                his = np.empty(vshape)
                for idx in np.ndindex(*vshape):
                    lo, hi = _interval(vals[(slice(None),) + idx], weights=w, nsigmas=nsigmas)
                    los[idx] = lo
                    his[idx] = hi
                results.append((los, his))
        return results[0] if scalar else results

    def argmax(self, params=None):
        """Parameter value(s) at the sample with the highest log-posterior.

        Parameters
        ----------
        params : str, Variable, list, or None

        Returns
        -------
        array or list of arrays, shape ``variable.shape``.
        """
        scalar, params = _normalise_params(self, params)
        flat_idx = np.argmax(self.logposterior.ravel())
        results = []
        for p in params:
            vals = _vals(self, p)          # (size, *var.shape)
            results.append(vals[flat_idx])
        return results[0] if scalar else results

    def to_stats(self, params=None, quantities=None, sigfigs=2,
                 tablefmt='latex_raw', fn=None):
        r"""Export a summary table of sampling statistics.

        Parameters
        ----------
        params : list[str or Variable], optional
            Parameters to include.  Defaults to all varied (non-derived) params.
        quantities : list[str], optional
            Quantities to compute per parameter.  Each entry is one of:

            * ``'argmax'`` — value at the highest-posterior sample
            * ``'mean'`` — weighted mean
            * ``'median'`` — weighted median
            * ``'std'`` — weighted standard deviation
            * ``'quantile:Nsigma'`` — symmetric quantile interval,
              e.g. ``'quantile:1sigma'``
            * ``'interval:Nsigma'`` — shortest credible interval,
              e.g. ``'interval:1sigma'``

            Defaults to ``['argmax', 'mean', 'median', 'std', 'quantile:1sigma', 'interval:1sigma']``.
        sigfigs : int, default=2
            Number of significant figures (passed to :func:`~desilike.utils.round_measurement`).
        tablefmt : str, default='latex_raw'
            ``tabulate`` format string.  Use ``'list'`` to return
            ``(rows, headers)`` instead of a formatted string.
        fn : str, optional
            If given, write the table to this file path.

        Returns
        -------
        str or tuple[list, list]
            Formatted table, or ``(rows, headers)`` when ``tablefmt='list'``.
        """
        import tabulate as _tabulate

        if params is None:
            # Default: non-derived variables only (exclude logposterior, weights, …)
            varied_params = [v.name for v in self._data if not v.derived]
        else:
            _, varied_params = _normalise_params(self, params)
        if quantities is None:
            quantities = ['argmax', 'mean', 'median', 'std',
                          'quantile:1sigma', 'interval:1sigma']
        is_latex = 'latex' in tablefmt

        rows = []
        for p in varied_params:
            row = []
            if is_latex and hasattr(p, 'latex'):
                row.append(p.latex(inline=True))
            else:
                row.append(str(p.name) if hasattr(p, 'name') else str(p))

            ref_center = float(np.ravel(self.mean(p))[0])
            ref_error  = float(np.ravel(self.std(p))[0])

            def _fmt_val(val):
                xr, _ = round_measurement(val, ref_error, sigfigs=sigfigs)
                return f'${xr}$' if is_latex else xr

            def _fmt_errs(lo_offset, hi_offset):
                _, lo_r, hi_r = round_measurement(
                    0.0, hi_offset, lo_offset,
                    sigfigs=sigfigs, positive_sign='u',
                )
                if is_latex:
                    return '${{}}_{{{lo}}}^{{{hi}}}$'.format(lo=lo_r, hi=hi_r)
                return f'{lo_r}/{hi_r}'

            for quantity in quantities:
                if quantity in ('argmax', 'mean', 'median', 'std'):
                    val = float(np.ravel(getattr(self, quantity)(p))[0])
                    row.append(_fmt_val(val))
                elif quantity.startswith('quantile:'):
                    match = re.match(r'quantile:(\d+)sigma', quantity)
                    if match is None:
                        raise ValueError(f'Cannot parse quantity {quantity!r}; expected e.g. quantile:1sigma')
                    nsigmas = int(match.group(1))
                    q_lo, q_hi = _nsigmas_to_quantiles_1d_sym(nsigmas)
                    lo, hi = (float(np.ravel(v)[0])
                               for v in self.quantile(p, q=(q_lo, q_hi)))
                    row.append(_fmt_errs(lo - ref_center, hi - ref_center))
                elif quantity.startswith('interval:'):
                    match = re.match(r'interval:(\d+)sigma', quantity)
                    if match is None:
                        raise ValueError(f'Cannot parse quantity {quantity!r}; expected e.g. interval:1sigma')
                    nsigmas = int(match.group(1))
                    lo, hi = (float(v) for v in self.interval(p, nsigmas=nsigmas))
                    row.append(_fmt_errs(lo - ref_center, hi - ref_center))
                else:
                    raise ValueError(f'Unknown quantity {quantity!r}')
            rows.append(row)

        headers = quantities
        if 'list' in tablefmt:
            return rows, headers

        tab = _tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
            with open(fn, 'w') as fh:
                fh.write(tab)
        return tab

    def covariance(self, params=None):
        """Weighted parameter covariance matrix.

        Uses the reliability-weights formula (same denominator as :meth:`var`).

        Parameters
        ----------
        params : str, Variable, list, or None
            Parameters to include.  ``None`` → all non-derived variables.
            Only scalar parameters (``shape == ()``) are supported; vector
            params are flattened to multiple columns in the output.

        Returns
        -------
        Covariance
            Named covariance matrix.  ``np.asarray(result)`` returns the
            plain ``(flat_size, flat_size)`` NumPy array.
        """
        from .covariance import Covariance
        if params is None:
            names = [v.name for v in self._data if not v.derived]
        else:
            _, names = _normalise_params(self, params)
        w = self.weight.ravel().astype(float)
        W = w.sum()
        W2 = (w ** 2).sum()
        denom = W ** 2 - W2   # reliability-weights denominator
        # Stack columns: (size, total_flat_size)
        cols = []
        for name in names:
            v = _vals(self, name)               # (size, *var.shape)
            cols.append(v.reshape(self.size, -1))
        X   = np.concatenate(cols, axis=1)      # (size, total_flat_size)
        mu  = np.average(X, weights=w, axis=0)
        D   = (X - mu) * w[:, None]             # weighted deviations
        cov_arr = (D.T @ (X - mu)) * W / denom  # (total_flat_size, total_flat_size)
        params_objs = [VariableCollection.__getitem__(self, name) for name in names]
        return Covariance(cov_arr, params=params_objs)

    def to_getdist(self, params=None, label=None, **kwargs):
        """Return a :class:`getdist.MCSamples` object from this chain.

        Requires the ``getdist`` package.

        Parameters
        ----------
        params : list or None
            Parameters to include.  ``None`` → all non-derived scalar variables.
        label : str, optional
            Name for GetDist to use for this set of samples.
        **kwargs
            Extra keyword arguments forwarded to :class:`getdist.MCSamples`.

        Returns
        -------
        getdist.MCSamples
        """
        from getdist import MCSamples

        if params is None:
            params_objs = [v for v in self._data if not v.derived]
        else:
            _, names = _normalise_params(self, params)
            params_objs = [VariableCollection.__getitem__(self, name) for name in names]

        names   = [p.name for p in params_objs]
        labels  = [p.latex() for p in params_objs]   # without $ delimiters
        ranges  = {}
        for param in params_objs:
            if hasattr(param, 'prior') and param.prior is not None:
                lo, hi = param.prior.limits
                ranges[param.name] = (
                    'N' if lo is None or not np.isfinite(float(lo)) else float(lo),
                    'N' if hi is None or not np.isfinite(float(hi)) else float(hi),
                )

        # Build (n_samples, n_params) array — scalar params only; vector params flattened
        samples  = np.column_stack([_vals(self, name).reshape(self.size, -1) for name in names])
        weights  = self.weight.ravel()
        loglikes = -self.logposterior.ravel()

        return MCSamples(samples=samples, weights=weights, loglikes=loglikes,
                         names=names, labels=labels, ranges=ranges,
                         label=label, **kwargs)
