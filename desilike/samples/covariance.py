"""Covariance and Precision — named parameter matrix containers."""

import os
import copy

import numpy as np

from ..parameter import Variable, VariableCollection
from ..utils import register_type, write as _utils_write, read as _utils_read, round_measurement


def _cov_to_corrcoef(cov):
    """Convert a covariance matrix to a correlation matrix."""
    diag = np.diag(cov)
    with np.errstate(invalid='ignore'):
        corr = cov / np.sqrt(np.outer(diag, diag))
    corr[~np.isfinite(corr)] = 0.
    return corr


# ── flat-layout helpers ───────────────────────────────────────────────────────

def _flat_size(params):
    """Total number of matrix rows/columns implied by *params*."""
    return sum(max(1, int(np.prod(p.shape))) for p in params)


def _param_sizes(params):
    """List of flat sizes, one per parameter in *params*."""
    return [max(1, int(np.prod(p.shape))) for p in params]


def _name_to_pos(params):
    """Return a ``{name: position}`` dict for *params*."""
    return {param.name: pos for pos, param in enumerate(params)}


class BaseMatrix:
    """Base class for a named parameter matrix.

    The matrix :attr:`value` has shape ``(flat_size, flat_size)`` where
    ``flat_size = sum(max(1, prod(param.shape)) for param in params)``.
    A scalar parameter occupies one row/column; a parameter of shape ``(n,)``
    occupies ``n`` consecutive rows/columns.

    Subclasses set ``_fill_value`` to control what is written into the
    diagonal block when a missing parameter is added via :meth:`select`.
    """

    _fill_value = np.nan

    def __init__(self, value, params, attrs=None):
        """
        Parameters
        ----------
        value : array_like, shape (flat_size, flat_size)
            Matrix entries.  Must be square with size equal to the total
            flat size of *params*.
        params : list or VariableCollection
            Parameters labelling the rows/columns.
        attrs : dict, optional
            Free-form metadata.
        """
        if isinstance(params, VariableCollection):
            self.params = params
        else:
            params_list = list(params)
            converted = [Variable(p) if isinstance(p, str) else p for p in params_list]
            self.params = VariableCollection(converted)
        self._value = np.atleast_2d(np.asarray(value, dtype='f8'))
        if self._value.ndim != 2:
            raise ValueError('Matrix value must be 2D')
        if self._value.shape[0] != self._value.shape[1]:
            raise ValueError('Matrix must be square')
        flat_size = _flat_size(self.params)
        if self._value.shape[0] != flat_size:
            raise ValueError(
                f'Matrix size ({self._value.shape[0]}) does not match '
                f'total flat size of params ({flat_size})'
            )
        self.attrs = dict(attrs or {})

    # ── param access ──────────────────────────────────────────────────────────

    def __contains__(self, key):
        """Test whether a parameter (name or Variable) is in the matrix."""
        return key in self.params

    # ── select ────────────────────────────────────────────────────────────────

    def select(self, params=None, **kwargs):
        """Return the sub-matrix for *params*.

        Parameters
        ----------
        params : str, Variable, list, or None
            Parameters to include.  ``None`` → all (optionally filtered by
            ``**kwargs`` forwarded to
            :meth:`~desilike.parameter.VariableCollection.select`).
            Unknown parameters are added as zero off-diagonal blocks with
            :attr:`_fill_value` on the diagonal.

        Returns
        -------
        instance of the same class
        """
        if params is None:
            params = self.params.select(**kwargs)
        if isinstance(params, (str, Variable)):
            params = [params]
        else:
            params = list(params)

        # Resolve each requested param to a Variable object.
        # Params present in self.params keep the stored object (with all attrs);
        # unknown params keep the caller-supplied object if it is a Variable/Parameter
        # (so that .ref, .prior, etc. are preserved), or fall back to a bare
        # Variable(name) placeholder when only a string was given.
        resolved = []
        for p in params:
            name = p.name if isinstance(p, Variable) else p
            if name in self.params:
                resolved.append(self.params[name])
            elif isinstance(p, Variable):
                resolved.append(p)
            else:
                resolved.append(Variable(name))

        resolved_names     = [p.name for p in resolved]
        params_in_self     = [p for p in resolved if p.name in self.params]
        params_not_in_self = [p for p in resolved if p.name not in self.params]

        resolved_name_to_pos = {name: pos for pos, name in enumerate(resolved_names)}
        self_name_to_pos     = _name_to_pos(self.params)

        resolved_sizes = [max(1, int(np.prod(p.shape))) for p in resolved]
        cumsizes_new   = np.cumsum([0] + resolved_sizes)
        cumsizes_self  = np.cumsum([0] + _param_sizes(self.params))

        total     = int(cumsizes_new[-1])
        new_value = np.zeros((total, total), dtype='f8')

        for p in params_not_in_self:
            param_pos = resolved_name_to_pos[p.name]
            for flat_idx in range(cumsizes_new[param_pos], cumsizes_new[param_pos + 1]):
                new_value[flat_idx, flat_idx] = self._fill_value

        for param in params_in_self:
            param_pos_new  = resolved_name_to_pos[param.name]
            param_pos_self = self_name_to_pos[param.name]
            row_new  = list(range(cumsizes_new [param_pos_new],  cumsizes_new [param_pos_new  + 1]))
            row_self = list(range(cumsizes_self[param_pos_self], cumsizes_self[param_pos_self + 1]))
            for param2 in params_in_self:
                param2_pos_new  = resolved_name_to_pos[param2.name]
                param2_pos_self = self_name_to_pos[param2.name]
                col_new  = list(range(cumsizes_new [param2_pos_new],  cumsizes_new [param2_pos_new  + 1]))
                col_self = list(range(cumsizes_self[param2_pos_self], cumsizes_self[param2_pos_self + 1]))
                new_value[np.ix_(row_new, col_new)] = self._value[np.ix_(row_self, col_self)]

        new = self.__class__.__new__(self.__class__)
        new.params = VariableCollection(resolved)
        new._value = new_value
        new.attrs  = dict(self.attrs)
        return new

    # ── numpy interface ───────────────────────────────────────────────────────

    @property
    def value(self):
        """The underlying 2-D numpy matrix array."""
        return self._value

    def __array__(self, dtype=None):
        return np.asarray(self._value, dtype=dtype)

    @property
    def shape(self):
        """Shape of the matrix value array."""
        return self._value.shape

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __mul__(self, other):
        new = copy.deepcopy(self)
        new._value = new._value * other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        new = copy.deepcopy(self)
        new._value = new._value / other
        return new

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    # ── misc ─────────────────────────────────────────────────────────────────

    def det(self, params=None):
        """Determinant of the matrix (optionally for a sub-set of *params*)."""
        return float(np.linalg.det(self.select(params).value))

    def clone(self, value=None, params=None, attrs=None):
        """Return a copy, optionally replacing *value*, *params*, or *attrs*."""
        new = copy.deepcopy(self.select(params))
        if value is not None:
            new._value[...] = value
        if attrs is not None:
            new.attrs = dict(attrs)
        return new

    def deepcopy(self):
        """Return a deep copy."""
        return copy.deepcopy(self)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def __getstate__(self, to_file=False):
        state = {
            'value':  self._value,
            'params': self.params.__getstate__(to_file=to_file),
        }
        if to_file:
            state['attrs'] = {'__class__': self._name,
                              **{k: str(v) for k, v in self.attrs.items()}}
        else:
            state['attrs'] = dict(self.attrs)
        return state

    def __setstate__(self, state):
        vc = VariableCollection.__new__(VariableCollection)
        vc.__setstate__(state['params'])
        self.params = vc
        self._value = np.asarray(state['value'])
        self.attrs  = {k: v for k, v in state.get('attrs', {}).items()
                       if k != '__class__'}

    def write(self, filename):
        """Write to an HDF5 or text file."""
        _utils_write(filename, self)

    @classmethod
    def read(cls, filename):
        """Read from a file written by :meth:`write`."""
        return _utils_read(filename)

    # ── repr / eq ─────────────────────────────────────────────────────────────

    def __repr__(self):
        return f'{type(self).__name__}(params={self.params.names()})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.params.names() == other.params.names()
                and np.array_equal(self._value, other._value))


# ── Covariance ────────────────────────────────────────────────────────────────

@register_type
class Covariance(BaseMatrix):
    """Named parameter covariance matrix.

    Missing parameters added via :meth:`select` get ``NaN`` on the diagonal
    (indicating unknown variance).

    Examples
    --------
    Building from a numpy array::

        params = ['omega_m', 'sigma8']
        cov = Covariance(np.diag([0.01**2, 0.02**2]), params=params)

    Extracting a sub-matrix and getting errors::

        sub = cov.select(['omega_m'])
        print(cov.std('omega_m'))   # → [0.01]

    Converting to precision::

        prec = cov.to_precision()
    """

    _name = 'Covariance'
    _fill_value = np.nan

    # ── covariance-specific select: fill with ref.std()^2 ────────────────────

    def select(self, params=None, fill=None, **kwargs):
        """Return the sub-matrix for *params*.

        Parameters
        ----------
        params : str, Variable, list, or None
        fill : {'ref', None}
            When ``'ref'``, unknown params whose ``ref.std()`` is finite get
            ``ref.std()**2`` on the diagonal instead of ``NaN``.

        Returns
        -------
        Covariance
        """
        new = super().select(params=params, **kwargs)
        if fill == 'ref':
            cumsizes = np.cumsum([0] + _param_sizes(new.params))
            for param_idx, param in enumerate(new.params):
                if param.name not in self.params:
                    ref = getattr(param, 'ref', None)
                    std = ref.std() if ref is not None else None
                    if std is not None and np.isfinite(float(std)):
                        diag_start = cumsizes[param_idx]
                        diag_end   = cumsizes[param_idx + 1]
                        flat_size  = diag_end - diag_start
                        new._value[diag_start:diag_end, diag_start:diag_end] = (
                            np.eye(flat_size) * float(std) ** 2
                        )
        return new

    # ── center ───────────────────────────────────────────────────────────────

    @property
    def center(self):
        """Flat vector of parameter values (center of the Gaussian), in ``params`` order."""
        parts = []
        for param in self.params:
            val = np.asarray(param.value).ravel()
            size = max(1, int(np.prod(param.shape)))
            if val.size == 1 and size > 1:
                val = np.full(size, float(val[0]))
            parts.append(val.astype('f8'))
        return np.concatenate(parts) if parts else np.array([], dtype='f8')

    # ── statistics ────────────────────────────────────────────────────────────

    def var(self, params=None):
        """Variance vector (diagonal of the covariance sub-matrix for *params*)."""
        return np.diag(self.select(params).value)

    def std(self, params=None):
        """Standard deviation (square root of :meth:`var`)."""
        return np.sqrt(self.var(params=params))

    def corrcoef(self, params=None):
        """Correlation matrix for *params* (or all params)."""
        return _cov_to_corrcoef(self.select(params).value)

    def fom(self, params=None):
        """Figure-of-merit: ``det(C)^{-1/2}``."""
        return self.det(params=params) ** (-0.5)

    # ── conversion ────────────────────────────────────────────────────────────

    def to_precision(self, params=None):
        """Return the inverse covariance (precision matrix) for *params*.

        Parameters
        ----------
        params : list or None

        Returns
        -------
        Precision
        """
        sub = self.select(params)
        return Precision(np.linalg.inv(sub.value), params=sub.params, attrs=sub.attrs)

    # ── export ────────────────────────────────────────────────────────────────

    def to_stats(self, params=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        """Export the covariance to a summary table.

        Parameters
        ----------
        params : list or None
        sigfigs : int
        tablefmt : str
            ``tabulate`` format string.
        fn : str, optional
            If given, write the table to this file.

        Returns
        -------
        str
        """
        import tabulate as _tabulate

        sub      = self.select(params)
        is_latex = 'latex' in tablefmt

        def _label(param):
            if is_latex and hasattr(param, 'latex'):
                return param.latex(inline=True)
            return str(param.name)

        headers = [''] + [_label(p) for p in sub.params]
        rows = []
        for param, row in zip(sub.params, sub.value):
            row_str = [_label(param)]
            for val in row:
                xr, = round_measurement(val, val, sigfigs=sigfigs)[:1]
                row_str.append(f'${xr}$' if is_latex else xr)
            rows.append(row_str)

        txt = _tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
            with open(fn, 'w') as file:
                file.write(txt)
        return txt

    def to_getdist(self, params=None, label=None, center=None, ignore_limits=True):
        """Return a GetDist Gaussian distribution with this covariance.

        Parameters
        ----------
        params : list or None
        label : str, optional
            Name for the GetDist distribution.
        center : array_like, optional
            Override :attr:`~desilike.parameter.Parameter.value` for the mean.
        ignore_limits : bool
            Drop parameter limits (GetDist struggles with bounded priors).

        Returns
        -------
        getdist.gaussian_mixtures.MixtureND
        """
        from getdist.gaussian_mixtures import MixtureND

        sub    = self.select(params)
        names  = [p.name for p in sub.params]
        labels = [p.latex(inline=False) if hasattr(p, 'latex') else p.name for p in sub.params]
        if center is None:
            center = np.array([float(np.asarray(p.value).ravel()[0])
                               if p.value is not None else 0.
                               for p in sub.params])
        else:
            center = np.asarray(center)
        ranges = None
        if not ignore_limits:
            ranges = [
                tuple(None if not np.isfinite(lim) else float(lim)
                      for lim in p.prior.limits)
                for p in sub.params
            ]
        return MixtureND([center], [sub.value],
                         lims=ranges, names=names, labels=labels, label=label)

    @classmethod
    def read_getdist(cls, base_fn):
        """Read a covariance matrix from a GetDist/CosmoMC ``.covmat`` file.

        Parameters
        ----------
        base_fn : str
            Base file name; ``'.covmat'`` is appended.

        Returns
        -------
        Covariance
        """
        from ..parameter import Parameter

        fn = f'{base_fn}.covmat'
        rows = []
        param_names = []
        with open(fn) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == '#':
                    param_names = parts[1:]
                else:
                    rows.append([float(v) for v in parts])
        params = [Parameter(name, fixed=False) for name in param_names]
        return cls(np.array(rows), params=params)


# ── Precision ────────────────────────────────────────────────────────────────

@register_type
class Precision(BaseMatrix):
    """Named parameter precision (inverse-covariance) matrix.

    Missing parameters added via :meth:`select` get ``0`` on the diagonal
    (zero precision = infinite variance = parameter unconstrained).

    Precision matrices are additive under independent experiments:
    ``P_total = P1 + P2`` (use :meth:`__add__` or :meth:`sum`).

    Examples
    --------
    ::

        prec = cov.to_precision()
        cov2 = prec.to_covariance()
        combined = prec1 + prec2
    """

    _name = 'Precision'
    _fill_value = 0.

    def to_covariance(self, params=None):
        """Return the inverse precision matrix (covariance) for *params*.

        Parameters
        ----------
        params : list or None

        Returns
        -------
        Covariance
        """
        sub = self.select(params)
        return Covariance(np.linalg.inv(sub.value), params=sub.params, attrs=sub.attrs)

    def fom(self, params=None):
        """Figure-of-merit delegated to the covariance: ``det(C)^{-1/2}``."""
        return self.to_covariance(params=params).fom()

    # ── addition ──────────────────────────────────────────────────────────────

    @classmethod
    def sum(cls, *others):
        """Sum precision matrices, unioning their parameter sets.

        Accepts ``Precision.sum(p1, p2, …)`` or ``Precision.sum([p1, p2, …])``.
        Unknown parameters in any operand are filled with zero precision.
        """
        if len(others) == 1 and isinstance(others[0], (list, tuple)):
            others = list(others[0])
        seen = {}
        for other in others:
            for param in other.params:
                seen.setdefault(param.name, param)
        all_params = list(seen.values())
        result = others[0].select(all_params)
        for other in others[1:]:
            other_view = other.select(all_params)
            result._value += other_view._value
            result.attrs.update(other_view.attrs)
        return result

    def __add__(self, other):
        """Sum of ``self`` and *other* precision matrices."""
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0:
            return self.deepcopy()
        return self.__add__(other)
