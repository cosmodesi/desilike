"""Profiles — container for the results of likelihood profiling."""

import os
import copy

import numpy as np

from ..parameter import VariableCollection
from ..utils import register_type, write as _utils_write, read as _utils_read


_SLOT_NAMES = ('start', 'best', 'logpdf', 'error', 'interval', 'profile', 'grid', 'contour', 'covariance')

# Slots that hold plain dicts and benefit from ParameterDict wrapping
_DICT_SLOTS = frozenset(('start', 'best', 'error', 'interval', 'profile', 'grid'))


class ParameterDict(dict):
    """dict subclass that accepts :class:`~desilike.parameter.Variable` or
    :class:`~desilike.parameter.Parameter` objects as keys by resolving them
    to their ``.name`` string attribute.

    Tuple keys (e.g. contour pairs ``(param1, param2)``) are resolved
    element-wise, so ``d[(param1, param2)]`` and ``d[('p1', 'p2')]`` are
    equivalent.
    """

    @staticmethod
    def _key(key):
        if isinstance(key, tuple):
            return tuple(k.name if hasattr(k, 'name') else k for k in key)
        return key.name if hasattr(key, 'name') else key

    def __getitem__(self, key):
        return super().__getitem__(self._key(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._key(key), value)

    def __contains__(self, key):
        return super().__contains__(self._key(key))

    def get(self, key, *args):
        return super().get(self._key(key), *args)


@register_type
class Profiles:
    r"""Container for likelihood profiling results.

    Attributes
    ----------
    attrs : dict
        Free-form metadata (e.g. ``ndof``, ``sampler``).
    params : VariableCollection, optional
        Parameter metadata (priors, latex labels, …).
    start : ParameterDict[str, np.ndarray], optional
        Starting points per minimiser run.
        Shape per key: ``(n_runs,) + param.shape``.
    best : ParameterDict[str, np.ndarray], optional
        Best-fit values per run.
        Shape per key: ``(n_runs,) + param.shape``.
    logpdf : np.ndarray, shape ``(n_runs,)``, optional
        Log-posterior value at each run's best-fit point.
    error : ParameterDict[str, np.ndarray], optional
        Parabolic errors per run.
        Shape per key: ``(n_runs,) + param.shape``.
    interval : ParameterDict[str, tuple[np.ndarray, np.ndarray]], optional
        Confidence intervals per run.
        ``interval[p] = (lo, hi)`` — each array has shape ``(n_runs,) + param.shape``.
    profile : ParameterDict[str, tuple[np.ndarray, np.ndarray]], optional
        1-D profiles.
        ``profile[p] = (scan_values, lp_values)`` — shapes ``(n_scan,) + param.shape``
        and ``(n_scan,)`` respectively (scalar params only for file I/O).
    grid : ParameterDict[str, np.ndarray], optional
        Parameter grid.
        Shape per key: arbitrary leading dimensions + ``param.shape``; no flatness
        requirement is enforced.
    contour : dict[cl, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]], optional
        2-D contours.  ``contour[cl][(p1, p2)] = (x_arr, y_arr)``.
    """

    _name = 'Profiles'

    def __init__(self, params=None, attrs=None, **kwargs):
        self.attrs = dict(attrs or {})
        self.params = VariableCollection(params) if params is not None else None
        for name in _SLOT_NAMES:
            object.__setattr__(self, name, None)
        if kwargs:
            self.set(**kwargs)

    # ── auto-wrap dict slots with ParameterDict ──────────────────────────────

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            if name in _DICT_SLOTS and not isinstance(value, ParameterDict):
                value = ParameterDict(value)
            elif name == 'contour':
                # Nested dict: keep the per-cl outer dict plain, wrap each
                # pair-dict so contour[cl][(param1, param2)] resolves Parameter keys.
                value = {cl: ParameterDict(pairs) for cl, pairs in value.items()}
        object.__setattr__(self, name, value)

    # ── set / get ─────────────────────────────────────────────────────────────

    def set(self, **kwargs):
        """Set one or more slots by name."""
        for name, value in kwargs.items():
            if name == 'params':
                self.params = VariableCollection(value) if value is not None else None
            elif name == 'attrs':
                self.attrs.update(value)
            elif name in _SLOT_NAMES:
                setattr(self, name, value)
            else:
                raise ValueError(
                    f'Unknown Profiles slot {name!r}; '
                    f'valid: params, attrs, {list(_SLOT_NAMES)}'
                )

    def get(self, name, *args):
        """Return attribute *name*, with an optional default."""
        return getattr(self, name, *args)

    def __contains__(self, name):
        return hasattr(self, name) and getattr(self, name) is not None

    # ── per-run convenience ───────────────────────────────────────────────────

    @property
    def chi2min(self):
        r"""Minimum :math:`\chi^2 = -2 \times \max(\text{logpdf})`."""
        return float(-2. * np.max(self.logpdf))

    @property
    def argmax(self):
        """Index of the run with the highest log-posterior."""
        return int(np.argmax(self.logpdf))

    @property
    def nruns(self):
        """Number of minimiser runs stored in :attr:`best`."""
        if self.logpdf is not None:
            return len(self.logpdf)
        return 0

    def choice(self, index='argmax', squeeze=False):
        """Return a new :class:`Profiles` sliced to one or more runs.

        ``start``, ``best``, ``logpdf``, ``error``, and ``interval`` are sliced
        along axis 0.  ``profile``, ``grid``, and ``contour`` are copied unchanged.
        A scalar *index* is wrapped in a list so the leading axis is always
        preserved (result shape ``(1,) + …``), unless *squeeze* is ``True``.

        Parameters
        ----------
        index : 'argmax' or int or array-like
        squeeze : bool, default=False
            When ``True`` and *index* was a scalar, drop the leading axis of every
            sliced array so shapes are ``param.shape`` rather than ``(1,) + param.shape``.
        """
        if isinstance(index, str) and index == 'argmax':
            index = self.argmax
        was_scalar = np.ndim(index) == 0
        if was_scalar:
            index = [index]
        new = copy.copy(self)
        for name in ('start', 'best', 'error'):
            d = getattr(self, name)
            if d is not None:
                setattr(new, name, {k: v[index] for k, v in d.items()})
        if self.logpdf is not None:
            new.logpdf = self.logpdf[index]
        if self.interval is not None:
            new.interval = {k: (lo[index], hi[index])
                            for k, (lo, hi) in self.interval.items()}
        if squeeze and was_scalar:
            for name in ('start', 'best', 'error'):
                d = getattr(new, name)
                if d is not None:
                    setattr(new, name, {k: v[0] for k, v in d.items()})
            if new.logpdf is not None:
                new.logpdf = new.logpdf[0]
            if new.interval is not None:
                new.interval = {k: (lo[0], hi[0]) for k, (lo, hi) in new.interval.items()}
        return new

    def select(self, **kwargs):
        """Return a new :class:`Profiles` restricted to matching parameters.

        Selection criteria are forwarded to
        :meth:`~desilike.parameter.VariableCollection.select` on :attr:`params`
        (e.g. name wildcards, ``fixed=False``), so :attr:`params` must be set.
        Every per-parameter slot (``start``, ``best``, ``error``, ``interval``,
        ``profile``, ``grid``) and ``contour`` is filtered to the selected names.
        :attr:`logpdf` is not per-parameter and is always carried over unchanged.

        Examples
        --------
        >>> profiles.select(varied=True)
        >>> profiles.select(name='omega_*')
        """
        if self.params is None:
            raise ValueError('Profiles.select requires params to be set')
        selected = self.params.select(**kwargs)
        names = set(selected.names())
        new = copy.copy(self)
        new.params = selected
        for name in ('start', 'best', 'error', 'interval', 'profile', 'grid'):
            d = getattr(self, name)
            if d is None:
                continue
            setattr(new, name, {k: v for k, v in d.items() if k in names})
        if self.contour is not None:
            new.contour = {cl: {(p1, p2): xy for (p1, p2), xy in pairs.items()
                                if p1 in names and p2 in names}
                           for cl, pairs in self.contour.items()}
        return new

    # ── concatenation / merge ─────────────────────────────────────────────────

    def update(self, other):
        """Merge *other* :class:`Profiles` into *self* in-place.

        ``start``, ``best``, ``error``, and ``interval`` are concatenated
        along axis 0.  ``profile``, ``grid``, and ``contour`` are merged:
        entries already present in *self* are kept; new keys from *other* are
        added.
        """
        self.attrs.update(other.attrs)
        if other.params is not None:
            self.params = (other.params if self.params is None
                           else self.params + other.params)

        # Concatenate logpdf along axis 0
        if other.logpdf is not None:
            self.logpdf = (other.logpdf.copy() if self.logpdf is None
                           else np.concatenate([self.logpdf, other.logpdf], axis=0))

        # Concatenate per-run plain dicts along axis 0
        for name in ('start', 'best', 'error'):
            other_d = getattr(other, name)
            if other_d is None:
                continue
            self_d = getattr(self, name)
            if self_d is None:
                setattr(self, name, {k: v.copy() for k, v in other_d.items()})
                continue
            merged = {}
            for k in dict.fromkeys(list(self_d) + list(other_d)):
                if k in self_d and k in other_d:
                    merged[k] = np.concatenate([self_d[k], other_d[k]], axis=0)
                else:
                    merged[k] = (self_d if k in self_d else other_d)[k].copy()
            setattr(self, name, merged)

        # Interval: concatenate (lo, hi) tuples along axis 0
        if other.interval is not None:
            if self.interval is None:
                self.interval = {k: (lo.copy(), hi.copy())
                                 for k, (lo, hi) in other.interval.items()}
            else:
                merged = {}
                for k in dict.fromkeys(list(self.interval) + list(other.interval)):
                    if k in self.interval and k in other.interval:
                        lo_s, hi_s = self.interval[k]
                        lo_o, hi_o = other.interval[k]
                        merged[k] = (np.concatenate([lo_s, lo_o], axis=0),
                                     np.concatenate([hi_s, hi_o], axis=0))
                    elif k in self.interval:
                        merged[k] = self.interval[k]
                    else:
                        merged[k] = other.interval[k]
                self.interval = merged

        # Profile / grid: merge — existing entries in self win
        for name in ('profile', 'grid'):
            other_d = getattr(other, name)
            if other_d is None:
                continue
            self_d = getattr(self, name)
            if self_d is None:
                setattr(self, name, dict(other_d))
            else:
                for k, v in other_d.items():
                    self_d.setdefault(k, v)

        # Contour: merge nested dicts — existing entries in self win
        if other.contour is not None:
            if self.contour is None:
                self.contour = {cl: dict(pairs)
                                for cl, pairs in other.contour.items()}
            else:
                for cl, pairs in other.contour.items():
                    if cl not in self.contour:
                        self.contour[cl] = ParameterDict(pairs)
                    else:
                        for pair, xy in pairs.items():
                            self.contour[cl].setdefault(pair, xy)

        # Covariance: keep self's value if present, otherwise take other's
        if self.covariance is None and other.covariance is not None:
            self.covariance = other.covariance

    @classmethod
    def concatenate(cls, *others):
        """Concatenate multiple :class:`Profiles` instances.

        Accepts either ``Profiles.concatenate(p1, p2, …)`` or
        ``Profiles.concatenate([p1, p2, …])``.
        """
        if not others:
            return cls()
        if len(others) == 1 and isinstance(others[0], (list, tuple)):
            others = others[0]
        new = copy.deepcopy(others[0])
        for other in others[1:]:
            new.update(other)
        return new

    def extend(self, other):
        """In-place :meth:`concatenate`."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def items(self):
        """Yield ``(name, value)`` for all non-None slots (including params)."""
        if self.params is not None:
            yield 'params', self.params
        for name in _SLOT_NAMES:
            val = getattr(self, name, None)
            if val is not None:
                yield name, val

    # ── copy ──────────────────────────────────────────────────────────────────

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        object.__setattr__(new, 'attrs', dict(self.attrs))
        object.__setattr__(new, 'params', self.params)  # shared VariableCollection reference
        for name in ('start', 'best', 'error', 'grid'):
            d = getattr(self, name, None)
            object.__setattr__(new, name, ParameterDict(d) if d is not None else None)
        lp = getattr(self, 'logpdf', None)
        object.__setattr__(new, 'logpdf', lp.copy() if lp is not None else None)
        for name in ('interval', 'profile'):
            d = getattr(self, name, None)
            object.__setattr__(new, name, ParameterDict(d) if d is not None else None)
        if self.contour is not None:
            object.__setattr__(new, 'contour', {cl: ParameterDict(pairs) for cl, pairs in self.contour.items()})
        else:
            object.__setattr__(new, 'contour', None)
        object.__setattr__(new, 'covariance', self.covariance)
        return new

    def deepcopy(self):
        """Return a deep copy."""
        return copy.deepcopy(self)

    # ── serialisation ─────────────────────────────────────────────────────────

    def __getstate__(self, to_file=False):
        state = {}
        if to_file:
            state['attrs'] = {'__class__': self._name,
                              **{str(k): str(v) for k, v in self.attrs.items()}}
        else:
            state['attrs'] = dict(self.attrs)

        if self.params is not None:
            state['params'] = self.params.__getstate__(to_file=to_file)

        # Plain per-run dicts: already arrays
        for name in ('start', 'best', 'error', 'grid'):
            d = getattr(self, name, None)
            if d is not None:
                state[name] = {k: np.asarray(v) for k, v in d.items()}

        # logpdf: plain 1-D array
        lp = getattr(self, 'logpdf', None)
        if lp is not None:
            state['logpdf'] = np.asarray(lp)

        # Interval: (lo, hi) → (2, n_runs, …) when to_file
        if self.interval is not None:
            if to_file:
                state['interval'] = {
                    k: np.stack([np.asarray(lo), np.asarray(hi)], axis=0)
                    for k, (lo, hi) in self.interval.items()
                }
            else:
                state['interval'] = dict(self.interval)

        # Profile: (scan_vals, lp_vals) → (2, n_scan, …) when to_file
        if self.profile is not None:
            if to_file:
                state['profile'] = {
                    k: np.stack([np.asarray(s), np.asarray(lp)], axis=0)
                    for k, (s, lp) in self.profile.items()
                }
            else:
                state['profile'] = dict(self.profile)

        # Covariance object
        if self.covariance is not None:
            state['covariance'] = self.covariance.__getstate__(to_file=to_file)

        # Contour: contour[cl][(p1, p2)] = (x, y)
        # to_file → {str(cl): {f'{p1}::{p2}': (2, n_pts) array}}
        if self.contour is not None:
            if to_file:
                contour_s = {}
                for cl, pairs in self.contour.items():
                    contour_s[str(cl)] = {
                        f'{p1}::{p2}': np.stack(
                            [np.asarray(x), np.asarray(y)], axis=0
                        )
                        for (p1, p2), (x, y) in pairs.items()
                    }
                state['contour'] = contour_s
            else:
                state['contour'] = {cl: dict(pairs)
                                    for cl, pairs in self.contour.items()}

        return state

    def __setstate__(self, state):
        is_file = '__class__' in state.get('attrs', {})
        object.__setattr__(self, 'attrs',
                           {k: v for k, v in state.get('attrs', {}).items() if k != '__class__'})

        # params
        if 'params' in state:
            vc = VariableCollection.__new__(VariableCollection)
            vc.__setstate__(state['params'])
            object.__setattr__(self, 'params', vc)
        else:
            object.__setattr__(self, 'params', None)

        # Plain per-run dicts
        for name in ('start', 'best', 'error', 'grid'):
            d = state.get(name)
            val = ParameterDict({k: np.asarray(v) for k, v in d.items()}) if d is not None else None
            object.__setattr__(self, name, val)

        # logpdf: dedicated 1-D array; fall back to best['logpdf'] for old files
        if 'logpdf' in state:
            object.__setattr__(self, 'logpdf', np.asarray(state['logpdf']))
        else:
            best_d = getattr(self, 'best', None)
            if best_d is not None and 'logpdf' in best_d:
                object.__setattr__(self, 'logpdf', best_d.pop('logpdf'))
            else:
                object.__setattr__(self, 'logpdf', None)

        # Interval: (2, n_runs, …) → (lo, hi) from file
        d = state.get('interval')
        if d is not None:
            val = ParameterDict({k: (arr[0], arr[1]) for k, arr in d.items()} if is_file else d)
            object.__setattr__(self, 'interval', val)
        else:
            object.__setattr__(self, 'interval', None)

        # Profile: (2, n_scan, …) → (scan_vals, lp_vals) from file
        d = state.get('profile')
        if d is not None:
            val = ParameterDict({k: (arr[0], arr[1]) for k, arr in d.items()} if is_file else d)
            object.__setattr__(self, 'profile', val)
        else:
            object.__setattr__(self, 'profile', None)

        # Covariance object
        cov_state = state.get('covariance')
        if cov_state is not None:
            from .covariance import Covariance
            cov_obj = Covariance.__new__(Covariance)
            cov_obj.__setstate__(cov_state)
            object.__setattr__(self, 'covariance', cov_obj)
        else:
            object.__setattr__(self, 'covariance', None)

        # Contour: str keys → (cl, (p1, p2)) keys from file
        d = state.get('contour')
        if d is not None:
            if is_file:
                contour = {}
                for cl_key, pairs in d.items():
                    try:
                        cl = int(cl_key)
                    except ValueError:
                        try:
                            cl = float(cl_key)
                        except ValueError:
                            cl = cl_key
                    contour[cl] = ParameterDict()
                    for pair_key, arr in pairs.items():
                        p1, p2 = pair_key.split('::', 1)
                        contour[cl][(p1, p2)] = (arr[0], arr[1])
                object.__setattr__(self, 'contour', contour)
            else:
                object.__setattr__(self, 'contour',
                                   {cl: ParameterDict(pairs) for cl, pairs in d.items()})
        else:
            object.__setattr__(self, 'contour', None)

    def write(self, filename):
        """Write to an HDF5 (``.h5``) or text (``.txt``) file."""
        _utils_write(filename, self)

    @classmethod
    def read(cls, filename):
        """Read a :class:`Profiles` from an HDF5 or text file."""
        return _utils_read(filename)

    # ── repr / eq ─────────────────────────────────────────────────────────────

    def __repr__(self):
        slots = [name for name in ('params',) + _SLOT_NAMES if name in self]
        return f'Profiles(slots={slots})'

    def __eq__(self, other):
        if not isinstance(other, Profiles):
            return NotImplemented
        if (self.logpdf is None) != (other.logpdf is None):
            return False
        if self.logpdf is not None and not np.array_equal(self.logpdf, other.logpdf):
            return False
        for name in ('start', 'best', 'error', 'grid'):
            sd, od = getattr(self, name), getattr(other, name)
            if (sd is None) != (od is None):
                return False
            if sd is not None:
                if set(sd) != set(od):
                    return False
                if not all(np.array_equal(sd[k], od[k]) for k in sd):
                    return False
        for name in ('interval', 'profile'):
            sd, od = getattr(self, name), getattr(other, name)
            if (sd is None) != (od is None):
                return False
            if sd is not None:
                if set(sd) != set(od):
                    return False
                for k in sd:
                    if not all(np.array_equal(a, b) for a, b in zip(sd[k], od[k])):
                        return False
        if (self.contour is None) != (other.contour is None):
            return False
        if self.contour is not None:
            if set(self.contour) != set(other.contour):
                return False
            for cl in self.contour:
                if set(self.contour[cl]) != set(other.contour[cl]):
                    return False
                for pair in self.contour[cl]:
                    sx, sy = self.contour[cl][pair]
                    ox, oy = other.contour[cl][pair]
                    if not (np.array_equal(sx, ox) and np.array_equal(sy, oy)):
                        return False
        return True

    # ── to_stats ──────────────────────────────────────────────────────────────

    def to_stats(self, params=None, quantities=None, sigfigs=2,
                 tablefmt='latex_raw', fn=None):
        r"""Export a summary table of profiling results.

        Parameters
        ----------
        params : list[str], optional
            Parameter names to include.  Defaults to all varied params in
            :attr:`params`, or all best keys excluding ``'logpdf'``.
        quantities : list[str], optional
            Quantities to include.  Defaults to every member of
            ``['best', 'error', 'interval']`` that is present.
        sigfigs : int, default=2
            Significant figures for value formatting.
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

        if self.best is None:
            raise ValueError('No best available on this Profiles object')

        idx = self.argmax

        if params is None:
            if self.params is not None:
                params = self.params.select(fixed=False).names()
            else:
                params = list(self.best.keys())

        allowed = ['best', 'error', 'interval']
        if quantities is None:
            quantities = [q for q in allowed if getattr(self, q) is not None]

        is_latex = 'latex' in tablefmt

        def _fmt(val, ref=None):
            if ref is None or ref == 0.:
                return f'{val:.{sigfigs}g}'
            mag = int(np.floor(np.log10(abs(ref)))) - (sigfigs - 1)
            rounded = round(float(val), -mag)
            decimals = max(0, -mag)
            return f'{rounded:.{decimals}f}'

        rows = []
        for pname in params:
            row = []
            param = (self.params[pname]
                     if self.params is not None and pname in self.params else None)
            if is_latex and param is not None and hasattr(param, 'latex'):
                row.append(param.latex(inline=True))
            else:
                row.append(str(pname))

            ref_err = None
            if self.error is not None and pname in self.error:
                ref_err = abs(float(self.error[pname].ravel()[idx]))

            for q in quantities:
                slot = getattr(self, q, None)
                if slot is None or pname not in slot:
                    row.append('')
                    continue
                if q in ('best', 'error'):
                    val = float(slot[pname].ravel()[idx])
                    s = _fmt(val, ref_err)
                    row.append(f'${s}$' if is_latex else s)
                elif q == 'interval':
                    lo, hi = slot[pname]
                    lo_v = float(lo.ravel()[idx]) if lo.ndim > 0 else float(lo)
                    hi_v = float(hi.ravel()[idx]) if hi.ndim > 0 else float(hi)
                    lo_s, hi_s = _fmt(lo_v, ref_err), _fmt(hi_v, ref_err)
                    s = (f'$[{lo_s},\\,{hi_s}]$' if is_latex
                         else f'[{lo_s}, {hi_s}]')
                    row.append(s)
            rows.append(row)

        chi2_str = f'{self.chi2min:.2f}'
        ndata = self.attrs.get('ndata')
        nvaried = self.attrs.get('nvaried')
        if ndata is not None and nvaried is not None:
            chi2_str += f' / ({int(ndata):d} - {int(nvaried):d})'
        elif self.attrs.get('ndof') is not None:
            chi2_str += f' / {int(self.attrs["ndof"]):d}'
        header_chi2 = (r'$\chi^2 = ' + chi2_str + '$') if is_latex else f'chi2 = {chi2_str}'
        headers = [header_chi2] + quantities

        if 'list' in tablefmt:
            return rows, headers

        tab = _tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
            with open(fn, 'w') as f:
                f.write(tab)
        return tab
