"""Parameter classes for desilike."""

import re
import copy
import json
import threading
import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats as sp
from .utils import NumpyEncoder, register_type, write as _utils_write, read as _utils_read


_compile_context = threading.local()


class _CompileContext:
    def __init__(self):
        self.traced = set()           # id(node) seen during dependency discovery (phase 1)
        self.stack = []               # currently-tracing Calculator stack
        self.node_deps = {}           # id(node) -> list[Node], in access order, deduplicated
        self.node_order = []          # topological order: leaves first, root last
        self.call_returns = {}        # id(node) -> return value of node.__call__()
        self.phase = 'post_init'      # 'post_init' or 'call'
        self.call_activated = set()   # id(Calculator) nodes accessed during __call__ (phase 2)

jax.config.update('jax_enable_x64', True)

NAMESPACE_SEP = '.'


# ── Name-matching helpers ──────────────────────────────────────────────────────

def decode_name(name, default_start=0, default_stop=None, default_step=1):
    """Split *name* into literal string segments and allowed integer index ranges.

    Bracket expressions ``[start:stop:step]`` are decoded into :class:`range`
    objects; ``*`` outside brackets is left as-is and handled by
    :func:`find_names`.

    Examples
    --------
    >>> decode_name('a_[-4:5:2]_b_[0:2]')
    (['a_', '_b_', ''], [range(-4, 5, 2), range(0, 2)])
    """
    name = str(name)
    replaces = re.finditer(r'\[([-+]?\d*):([-+]?\d*):*([-+]?\d*)\]', name)
    strings, ranges = [], []
    string_start = 0
    for replace in replaces:
        start, stop, step = replace.groups()
        start = default_start if not start else int(start)
        stop = default_stop if not stop else int(stop)
        step = default_step if not step else int(step)
        if start is None:
            raise ValueError('Lower limit required for parameter index range')
        if stop is None:
            raise ValueError('Upper limit required for parameter index range')
        strings.append(name[string_start:replace.start()])
        string_start = replace.end()
        ranges.append(range(start, stop, step))
    strings.append(name[string_start:])
    return strings, ranges


def find_names(allnames, name, quiet=True):
    """Return the subset of *allnames* matching *name*.

    *name* may be a single string or a list of strings.  Each pattern supports:

    * ``*`` – wildcard matching any substring (converted to non-greedy ``.*?``).
    * ``[start:stop]`` / ``[start:stop:step]`` – integer index ranges.
    * An already-compiled :class:`re.Pattern` (skips bracket parsing).

    Parameters
    ----------
    allnames : list[str]
        Candidate names to search.
    name : str, list[str], re.Pattern
        Pattern(s) to match against *allnames*.
    quiet : bool, default=True
        If ``False``, raise :class:`ValueError` when no match is found.

    Returns
    -------
    list[str]
        Matching names from *allnames*, in their original order.

    Examples
    --------
    >>> find_names(['a_1', 'a_2', 'b_1'], 'a_*')
    ['a_1', 'a_2']
    >>> find_names(['a_1', 'a_2', 'b_1'], ['a_[:]', 'b_[:]'])
    ['a_1', 'a_2', 'b_1']
    """
    if not allnames:
        return []
    if isinstance(name, (list, tuple)):
        toret = []
        for n in name:
            toret += find_names(allnames, n, quiet=quiet)
        return toret
    if isinstance(name, re.Pattern):
        pattern, ranges = name, []
    else:
        pat_str = name.replace('*', '.*?') + '$'
        strings, ranges = decode_name(pat_str)
        pattern = re.compile(r'([-+]?\d*)'.join(strings))
    toret = []
    for candidate in allnames:
        match = re.match(pattern, candidate)
        if match:
            add = True
            for s, ra in zip(match.groups(), ranges):
                if int(s) not in ra:
                    add = False
                    break
            if add:
                toret.append(candidate)
    if not toret and not quiet:
        raise ValueError('No match found for {}'.format(name))
    return toret


# ── Parameter-specific JSON encoder / decoder ─────────────────────────────────

class ParameterEncoder(NumpyEncoder):
    """JSON encoder that additionally serialises :class:`ParameterPrior` objects.

    ±inf limit values are converted to ``null`` (JSON / Python ``None``).
    :class:`ParameterPrior.__init__` already maps ``None`` limits back to ±inf
    on reconstruction, so the round-trip is lossless.
    """

    def default(self, obj):
        if isinstance(obj, ParameterPrior):
            lo, hi = obj.limits
            d = {'__class__': 'ParameterPrior',
                 'dist': obj.dist,
                 'limits': [None if not np.isfinite(lo) else float(lo),
                            None if not np.isfinite(hi) else float(hi)]}
            if obj.shape is not None:
                d['shape'] = list(obj.shape)
            d.update(obj.attrs)
            return d
        return super().default(obj)


def _parameter_object_hook(d):
    """``object_hook`` for :func:`json.loads` that reconstructs :class:`ParameterPrior`."""
    if d.get('__class__') == 'ParameterPrior':
        d = dict(d)
        d.pop('__class__')
        if d.get('limits') is not None:
            d['limits'] = tuple(d['limits'])   # __init__ maps None → ±inf
        if d.get('shape') is not None:
            d['shape'] = tuple(d['shape'])
        return ParameterPrior(**d)
    return d


def _iter_nodes(value, _seen=None):
    """Yield every :class:`Node` reachable from *value* through standard containers.

    Descends into ``list``/``tuple``/``set``/``frozenset``/``dict`` (both keys and
    values) and :class:`VariableCollection`, but stops at any :class:`Node` (yielding
    it without descending into its own attributes) and at non-container leaves (arrays,
    scalars, strings, arbitrary objects).  Cycle-safe via an id-based visited set.
    """
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return
    _seen.add(id(value))
    if isinstance(value, Node):
        yield value            # a Node is a leaf dependency; do not descend into it
    elif isinstance(value, dict):
        for key, val in value.items():
            yield from _iter_nodes(key, _seen)
            yield from _iter_nodes(val, _seen)
    elif isinstance(value, (list, tuple, set, frozenset, VariableCollection)):
        for val in value:
            yield from _iter_nodes(val, _seen)
    # else: array / scalar / str / arbitrary object → not a dependency container


def _substitute_node(value, match, new):
    """Return *value* with every Node satisfying ``match(node)`` replaced by *new*.

    Mutating, path-aware sibling of :func:`_iter_nodes`: rebuilds standard containers
    (``list``/``tuple``/``set``/``frozenset``/``dict``, keys and values) and
    :class:`VariableCollection` so a node held in e.g. a tuple-of-tuples or a collection
    is replaced.  Does not descend into Nodes (a Node is replaced as a whole when it matches).
    """
    if isinstance(value, Node):
        return new if match(value) else value
    if isinstance(value, dict):
        return {_substitute_node(key, match, new): _substitute_node(val, match, new) for key, val in value.items()}
    if isinstance(value, list):
        return [_substitute_node(val, match, new) for val in value]
    if isinstance(value, tuple):
        return tuple(_substitute_node(val, match, new) for val in value)
    if isinstance(value, set):
        return {_substitute_node(val, match, new) for val in value}
    if isinstance(value, frozenset):
        return frozenset(_substitute_node(val, match, new) for val in value)
    if isinstance(value, VariableCollection):
        substituted = VariableCollection()
        for val in value:
            substituted.set(_substitute_node(val, match, new))
        return substituted
    return value


class Node:
    """Common base for mutable objects traced in the pipeline."""

    _is_calculator = False

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if not name.startswith('_'):
            ctx = getattr(_compile_context, 'ctx', None)
            if ctx is not None and ctx.phase == 'call' and ctx.stack and ctx.stack[-1] is self:
                for node in _iter_nodes(value):
                    if object.__getattribute__(node, '_is_calculator'):
                        if id(node) not in ctx.traced:
                            raise RuntimeError(
                                f"{type(self).__name__}.__call__ introduced new Calculator "
                                f"{type(node).__name__!r} not declared in __post_init__; "
                                f"all Calculator dependencies must be declared in __post_init__"
                            )
                        ctx.call_activated.add(id(node))
        return value


class Variable(Node):
    """Named mutable value container, traced in the pipeline.

    Minimal building block: a name, a current value, and a derived flag.
    Parameter is a subclass adding prior/ref/sampling metadata.
    """

    def __init__(self, name=None, value=None, derived=False, latex=None, namespace=None, basename=None, shape=None):
        # Name may embed a namespace via NAMESPACE_SEP ('.'); ``basename`` overrides the
        # parsed basename and ``namespace`` is prepended to any embedded namespace.
        parts = str(name).split(NAMESPACE_SEP) if name is not None else ['']
        bn = str(basename) if basename is not None else parts[-1]
        embedded_ns = NAMESPACE_SEP.join(parts[:-1])
        ns = NAMESPACE_SEP.join(filter(None, [namespace, embedded_ns])) if namespace else embedded_ns
        self._name = NAMESPACE_SEP.join([ns, bn]) if ns else bn
        self._latex = latex
        self._derived = bool(derived)
        if value is not None:
            v = np.asarray(value)
            self.shape = tuple(shape) if shape is not None else v.shape
            self.value = float(v) if v.shape == () else v
        else:
            self.shape = tuple(shape) if shape is not None else ()
            self._value = None

    @property
    def name(self):
        return self._name

    @property
    def basename(self):
        return self._name.split(NAMESPACE_SEP)[-1]

    @property
    def namespace(self):
        parts = self._name.split(NAMESPACE_SEP)
        return NAMESPACE_SEP.join(parts[:-1]) if len(parts) > 1 else ''

    @property
    def derived(self):
        return self._derived

    def latex(self, namespace=None, inline=False):
        """
        Return latex string for parameter if :attr:`latex` is specified (i.e. not ``None``), else :attr:`name`.

        Parameters
        ----------
        namespace : bool, str, default=None
            If ``False``, no namespace is added to the latex string.
            If ``True``, :attr:`namespace` is turned into a latex string, and added as a subscript.
            If string, add this subscript to the latex string.
            If ``None``, and none of :attr:`namespace` "words" (defined as group of characters separated by ',', ' ', '_', '-')
            are in the current latex string, then same as ``True``; else, same as ``False``.
        inline : bool, default=False
            If ``True``, add '$' around the latex string.
        Returns
        -------
        latex : str
            Latex string.
        """
        auto_namespace = namespace is None
        force_namespace = namespace is True
        provided_namespace = False
        if force_namespace or auto_namespace:
            namespace = str(self.namespace)
        elif namespace is not False:
            namespace = str(namespace)
            provided_namespace = force_namespace = True

        if self._latex is not None:

            def add_namespace(group):
                words = re.split(', |_|-', namespace)  # parse namespace
                for word in words:
                    if word in self._latex and word not in self.basename:
                        return False
                return True

            latex = self._latex
            if namespace and (force_namespace or auto_namespace):
                match1 = re.match('(.*)_(.)$', self._latex)
                match2 = re.match('(.*)_{(.*)}$', self._latex)
                latex_namespace = namespace if provided_namespace else (r'\mathrm{%s}' % namespace.replace(r'\_', '_').replace('_', r'\_'))
                for match in [match1, match2, None]:
                    if match is not None:
                        if force_namespace or (auto_namespace and add_namespace(match.group(2))):  # check namespace is not in latex str already
                            latex = r'%s_{%s, %s}' % (match.group(1), match.group(2), latex_namespace)
                        break
                    elif force_namespace or (auto_namespace and add_namespace(namespace)):
                        latex = r'%s_{%s}' % (self._latex, latex_namespace)
            if inline:
                latex = '${}$'.format(latex)
            return latex
        return str(self.name)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if self.shape:
            v_shape = getattr(v, 'shape', None)
            if v_shape is None:
                v_shape = np.shape(v)
            if v_shape[-len(self.shape):] != self.shape:
                raise ValueError(f"'{self._name}': value shape {v_shape} incompatible with parameter shape {self.shape}")
        self._value = v

    def __call__(self):
        return self._value

    def update(self, **kwargs):
        state = self.__getstate__()
        state.update(kwargs)
        self.__init__(**state)
        return self

    def __getstate__(self, to_file=False):
        if to_file:
            # Include 'shape' so that Chain can distinguish intrinsic Variable.shape
            # from the leading sample dimensions stored in _value (backward-compat: old
            # files without 'shape' fall back to inferring shape from value on load).
            meta = {'__class__': 'Variable', 'name': self._name, 'derived': self._derived,
                    'latex': self._latex, 'shape': list(self.shape)}
            state = {'attrs': {'meta': json.dumps(meta)}}
            if self._value is not None:
                state['value'] = np.asarray(self._value)
            return state
        return {'name': self._name, 'value': self._value, 'derived': self._derived,
                'latex': self._latex, 'shape': self.shape}

    def __setstate__(self, state):
        if 'attrs' in state:
            # file format: metadata in JSON, value as numpy dataset
            raw = state['attrs'].get('meta', '{}')
            if isinstance(raw, bytes):
                raw = raw.decode()
            meta = json.loads(raw)
            self.__init__(name=meta['name'], value=state.get('value'), derived=meta.get('derived', False),
                          latex=meta.get('latex'))
            # Restore intrinsic shape when explicitly stored (used by Chain to keep
            # Variable.shape = per-sample shape, separate from the leading sample dims).
            if 'shape' in meta:
                self.shape = tuple(meta['shape'])
        else:
            # in-memory format
            self.__init__(name=state['name'], value=state.get('value'), derived=state.get('derived', False),
                          latex=state.get('latex'))
            if 'shape' in state:
                self.shape = tuple(state['shape'])

    # Minimal FD defaults so external (_is_external=True) calculators work when Variable is a dep.
    @property
    def fd_eps(self):
        return None

    @property
    def fd_acc(self):
        return 2

    @property
    def dtype(self):
        # Delegate to the stored value's .dtype when available (works for both
        # numpy arrays and JAX-traced values without concretizing the trace).
        if hasattr(self._value, 'dtype'):
            return self._value.dtype
        return np.asarray(self._value).dtype

    @property
    def ndim(self):
        if hasattr(self._value, 'ndim'):
            return self._value.ndim
        return np.ndim(self._value)

    def __jax_array__(self):
        return jnp.asarray(self._value)

    def __array__(self, dtype=None):
        return np.asarray(self._value, dtype=dtype)

    def __float__(self):   return float(self._value)
    def __int__(self):     return int(self._value)

    def __add__(self, o):      return self._value + o
    def __radd__(self, o):     return o + self._value
    def __sub__(self, o):      return self._value - o
    def __rsub__(self, o):     return o - self._value
    def __mul__(self, o):      return self._value * o
    def __rmul__(self, o):     return o * self._value
    def __truediv__(self, o):  return self._value / o
    def __rtruediv__(self, o): return o / self._value
    def __pow__(self, o):      return self._value ** o
    def __rpow__(self, o):     return o ** self._value
    def __neg__(self):         return -self._value
    def __pos__(self):         return +self._value
    def __abs__(self):         return abs(self._value)

    def __eq__(self, other):
        return type(other) is type(self) and self._name == other._name

    def __hash__(self):
        return hash(self._name)

    def clone(self, **kwargs):
        """Return a copy with selected attributes overridden."""
        state = self.__getstate__()
        state.update(kwargs)
        return Variable(**state)

    def __repr__(self):
        return f'Variable({self._name!r})'

    def __str__(self):
        return self._name


class ParameterPrior:
    """1D prior distribution.

    logpdf() is JAX-differentiable (jit/grad/vmap compatible).
    sample() uses the JAX PRNG convention: sample(key, shape=()).
    An improper (flat) prior is dist='uniform' with at least one infinite limit.
    """

    def __init__(self, dist='uniform', limits=None, shape=None, **attrs):
        """
        Parameters
        ----------
        dist : str or ParameterPrior
            Distribution name (jax.scipy.stats), or a ParameterPrior to copy.
        limits : tuple, optional
            (lo, hi) bounds; None/±inf entries become ±inf.
        shape : tuple, optional
            Array shape of the parameter this prior belongs to. None means unset.
        **attrs
            Distribution parameters (e.g. loc=0.3, scale=0.01 for 'norm').
        """
        if isinstance(dist, ParameterPrior):
            for k, v in dist.__dict__.items():
                if k != '_frozen':
                    object.__setattr__(self, k, v)
            object.__setattr__(self, 'attrs', dict(dist.attrs))
            object.__setattr__(self, '_frozen', True)
            return
        if isinstance(dist, dict):
            kw = {**dist, **attrs}
            dist = kw.pop('dist', 'uniform')
            limits = kw.pop('limits', limits)
            shape = kw.pop('shape', shape)
            attrs = kw
        self.dist = str(dist).lower()
        if limits is None:
            limits = (-np.inf, np.inf)
        lo, hi = limits
        if lo is None: lo = -np.inf
        if hi is None: hi = np.inf
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError(f'Prior limits ({lo}, {hi}): lower >= upper')
        self.limits = (lo, hi)
        self.shape = tuple(shape) if shape is not None else None
        self.attrs = dict(attrs)
        self._setup()
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError(f'{self.__class__.__name__} is immutable; use clone() to get a modified copy')
        object.__setattr__(self, name, value)

    def _setup(self):
        """Build JAX logpdf/sample closures and compute moments (via scipy, once at init)."""
        dist, (lo, hi), attrs = self.dist, self.limits, self.attrs

        # ── uniform ──────────────────────────────────────────────────────────
        if dist == 'uniform':
            self._is_proper = np.isfinite(lo) and np.isfinite(hi)
            if not self._is_proper:
                lo_, hi_ = lo, hi
                def _logpdf(x):
                    x = jnp.asarray(x, dtype=float)
                    return jnp.where((lo_ < x) & (x < hi_), 0., -jnp.inf)
                self._logpdf_fn = _logpdf
                self._sample_fn = None
                self._ppf_fn = None
                finite = [l for l in (lo, hi) if np.isfinite(l)]
                self._center = float(np.mean(finite)) if finite else 0.
                self._std = np.inf
            else:
                lo_, hi_ = lo, hi
                def _logpdf(x):
                    return jax.scipy.stats.uniform.logpdf(jnp.asarray(x), loc=lo_, scale=hi_ - lo_)
                def _sample(key, shape):
                    return jax.random.uniform(key, shape=shape, minval=lo_, maxval=hi_, dtype=jnp.float64)
                def _ppf(u, _lo=lo_, _hi=hi_):
                    return _lo + jnp.asarray(u, dtype=float) * (_hi - _lo)
                self._logpdf_fn = _logpdf
                self._sample_fn = _sample
                self._ppf_fn = _ppf
                self._center = (lo + hi) / 2.
                self._std = (hi - lo) / float(np.sqrt(12.))
            self._logpdf_center_val = float(self._logpdf_fn(jnp.asarray(self._center)))
            return

        # ── norm (with optional truncation via truncnorm) ─────────────────────
        if dist == 'norm':
            self._is_proper = True
            loc_ = float(attrs.get('loc', 0.))
            scale_ = float(attrs.get('scale', 1.))
            if np.isinf(lo) and np.isinf(hi):
                def _logpdf(x):
                    return jax.scipy.stats.norm.logpdf(jnp.asarray(x), loc=loc_, scale=scale_)
                def _sample(key, shape):
                    return jax.random.normal(key, shape=shape, dtype=jnp.float64) * scale_ + loc_
                def _ppf(u, _s=scale_, _l=loc_):
                    return jax.scipy.special.ndtri(jnp.asarray(u, dtype=float)) * _s + _l
                self._center, self._std = loc_, scale_
            else:
                a = float((lo - loc_) / scale_) if np.isfinite(lo) else -np.inf
                b = float((hi - loc_) / scale_) if np.isfinite(hi) else np.inf
                def _logpdf(x):
                    return jax.scipy.stats.truncnorm.logpdf(jnp.asarray(x), a, b, loc=loc_, scale=scale_)
                # inverse-CDF sampling: uniform → ndtri (inverse normal CDF)
                p_lo_ = float(sp.norm.cdf(a))
                p_hi_ = float(sp.norm.cdf(b))
                def _sample(key, shape, _plo=p_lo_, _phi=p_hi_, _s=scale_, _l=loc_):
                    u = jax.random.uniform(key, shape=shape, minval=_plo, maxval=_phi, dtype=jnp.float64)
                    return jax.scipy.special.ndtri(u) * _s + _l
                rv = sp.truncnorm(a, b, loc=loc_, scale=scale_)
                # inverse truncated-normal CDF: map u in [0, 1] into the truncated
                # quantile range [p_lo_, p_hi_], then apply the (JAX) inverse normal CDF.
                def _ppf(u, _plo=p_lo_, _phi=p_hi_, _s=scale_, _l=loc_):
                    return jax.scipy.special.ndtri(_plo + jnp.asarray(u, dtype=float) * (_phi - _plo)) * _s + _l
                self._center, self._std = float(rv.mean()), float(rv.std())
            self._logpdf_fn = _logpdf
            self._sample_fn = _sample
            self._ppf_fn = _ppf
            self._logpdf_center_val = float(self._logpdf_fn(jnp.asarray(self._center)))
            return

        # ── other dists via jax.scipy.stats ──────────────────────────────────
        jss = getattr(jax.scipy.stats, dist, None)
        if jss is None or not hasattr(jss, 'logpdf'):
            raise ValueError(f'Distribution {dist!r} not supported; not found in jax.scipy.stats')
        self._is_proper = True

        if np.isinf(lo) and np.isinf(hi):
            def _logpdf(x):
                return jss.logpdf(jnp.asarray(x), **attrs)
        else:
            rv_sp = getattr(sp, dist)(**attrs)
            log_z = float(np.log(max(float(rv_sp.cdf(hi)) - float(rv_sp.cdf(lo)), 1e-300)))
            lo_, hi_ = lo, hi
            def _logpdf(x, _lz=log_z, _lo=lo_, _hi=hi_):
                x = jnp.asarray(x)
                return jnp.where((_lo < x) & (x < _hi), jss.logpdf(x, **attrs) - _lz, -jnp.inf)
        self._logpdf_fn = _logpdf

        # sampling and ppf via inverse CDF (scipy, result converted to JAX)
        rv_sp = getattr(sp, dist)(**attrs)
        p_lo_ = float(rv_sp.cdf(lo) if np.isfinite(lo) else 0.)
        p_hi_ = float(rv_sp.cdf(hi) if np.isfinite(hi) else 1.)
        def _sample(key, shape, _rv=rv_sp, _plo=p_lo_, _phi=p_hi_):
            u = jax.random.uniform(key, shape=shape, minval=_plo, maxval=_phi, dtype=jnp.float64)
            return jnp.asarray(_rv.ppf(np.asarray(u)), dtype=jnp.float64)
        # jax.scipy.stats has no generic ppf; keep scipy's inverse CDF but expose it
        # through jax.pure_callback so it stays traceable under jit/vmap (same pattern
        # as _is_external=True calculators in base.py).
        def _ppf(u, _rv=rv_sp, _plo=p_lo_, _phi=p_hi_):
            u = jnp.asarray(u, dtype=jnp.float64)
            def _scipy_ppf(uu):
                return np.asarray(_rv.ppf(_plo + np.asarray(uu) * (_phi - _plo)), dtype='f8')
            return jax.pure_callback(
                _scipy_ppf, jax.ShapeDtypeStruct(jnp.shape(u), jnp.float64), u,
                vmap_method='broadcast_all')
        self._sample_fn = _sample
        self._ppf_fn = _ppf

        # moments via scipy
        try:
            trunc_name = f'trunc{dist}'
            if not (np.isinf(lo) and np.isinf(hi)) and hasattr(sp, trunc_name):
                loc_ = float(attrs.get('loc', 0.))
                scale_ = float(attrs.get('scale', 1.))
                a = float((lo - loc_) / scale_) if np.isfinite(lo) else -np.inf
                b = float((hi - loc_) / scale_) if np.isfinite(hi) else np.inf
                rv_m = getattr(sp, trunc_name)(a, b, **attrs)
            else:
                rv_m = rv_sp
            m, s = float(rv_m.mean()), float(rv_m.std())
            self._center = m if np.isfinite(m) else 0.
            self._std = s
        except Exception:
            self._center, self._std = 0., None
        self._logpdf_center_val = float(self._logpdf_fn(jnp.asarray(self._center)))

    def logpdf(self, x):
        """Return log PDF relative to center: logpdf(x) - logpdf(center) ≤ 0.

        Removing the constant logpdf at the center (= maximum for unimodal
        priors) means the prior contribution is always ≤ 0 and equals 0 at the
        peak, so the posterior equals the likelihood at the best-fit point.
        This matches the behaviour of the backup desilike ``remove_zerolag=True``
        convention.
        """
        return self._logpdf_fn(jnp.asarray(x)) - self._logpdf_center_val

    def sample(self, key, shape=None):
        """Draw samples using JAX PRNG key; raises if prior is improper.

        shape defaults to self.shape (set by Parameter) when None; falls back to ().
        """
        if self._sample_fn is None:
            raise ValueError('Cannot sample from improper prior')
        shape = tuple(np.atleast_1d(shape)) + (self.shape or ()) if shape is not None else self.shape
        return self._sample_fn(key, shape)

    def ppf(self, u):
        """Percent-point function (inverse CDF) at quantile *u* ∈ [0, 1].

        Maps a uniformly distributed value *u* to parameter space using the
        prior's inverse CDF.  This is the prior transform required by nested
        samplers such as ``dynesty`` and ``nautilus``.

        JAX-traceable (``jit``/``vmap``-able): *u* may be a scalar or an array,
        and the return value is a JAX array of the same shape.

        Parameters
        ----------
        u : float or array_like
            Quantile(s) in [0, 1].

        Returns
        -------
        jax.Array
            Parameter value(s) at quantile *u*.

        Raises
        ------
        ValueError
            If the prior is improper (no finite integral).
        """
        if self._ppf_fn is None:
            raise ValueError('Cannot evaluate ppf of improper prior')
        return self._ppf_fn(u)

    def center(self):
        """Return distribution center as a Python float."""
        return self._center

    def std(self):
        """Return distribution std as a Python float, or None if unavailable."""
        return self._std

    def isin(self, x):
        """Return boolean JAX array: True where x is strictly inside limits."""
        x = jnp.asarray(x)
        return (self.limits[0] < x) & (x < self.limits[1])

    def is_proper(self):
        """True if the distribution has a finite integral."""
        return self._is_proper

    def is_limited(self):
        """True if at least one limit is finite."""
        return np.isfinite(self.limits[0]) or np.isfinite(self.limits[1])

    def affine_transform(self, loc=0., scale=1.):
        """Return new ParameterPrior for the transformed variable y = scale * x + loc."""
        state = self.__getstate__()
        lo, hi = self.limits
        state['limits'] = (lo * scale + loc if np.isfinite(lo) else lo, hi * scale + loc if np.isfinite(hi) else hi)
        if 'loc' in state:
            state['loc'] = state['loc'] * scale + loc
        if 'scale' in state:
            state['scale'] = state['scale'] * abs(scale)
        return ParameterPrior(**state)

    def __getstate__(self):
        state = {'dist': self.dist, 'limits': self.limits}
        if self.shape is not None:
            state['shape'] = self.shape
        state.update(self.attrs)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def __eq__(self, other):
        return (type(other) is type(self) and self.dist == other.dist and self.limits == other.limits
                and self.attrs == other.attrs and self.shape == other.shape)

    def __repr__(self):
        parts = [repr(self.dist)]
        if self.is_limited():
            parts.append(f'limits={self.limits}')
        parts += [f'{k}={v}' for k, v in self.attrs.items()]
        if self.shape is not None:
            parts.append(f'shape={self.shape}')
        return f'ParameterPrior({", ".join(parts)})'

    def clone(self, **kwargs):
        """Return a new ParameterPrior with selected attributes overridden."""
        state = self.__getstate__()
        state.update(kwargs)
        return ParameterPrior(**state)

    def copy(self):
        return self.clone()


class Parameter(Variable):
    """A single named parameter with prior, value, and metadata.

    Names may embed a namespace using NAMESPACE_SEP ('.'): e.g. 'galaxy.omega_m'.
    An optional ``namespace`` keyword prepends to whatever is parsed from ``name``.
    """

    _solved_values = frozenset(['best', 'marg'])

    def __init__(self, name, value=None, prior=None, ref=None, latex=None, fixed=None,
                 derived=False, shape=(), fd_eps=None, fd_acc=2,
                 namespace=None, depends=None):
        """
        Parameters
        ----------
        name : str or Parameter
            Parameter name, optionally namespace-prefixed (e.g. 'galaxy.omega_m').
            If a Parameter, copy-construct from it (all other args ignored).
        value : float, optional
            Default value. Inferred from prior center when omitted and prior is proper.
        prior : ParameterPrior, dict, or None
            Prior distribution. Defaults to improper flat.
        ref : ParameterPrior, dict, or None
            Reference distribution (expected posterior). Defaults to copy of prior.
        latex : str, optional
            LaTeX string (without surrounding $).
        fixed : bool, optional
            Whether the parameter is fixed. Defaults to True when no prior/ref given.
        derived : bool or str
            False (default), True, 'best' or 'marg' (solved), or an expression string using
            {param_name} placeholders (e.g. '{omega_m} * {h}**2').
        shape : tuple
            Array shape; () for scalars.
        fd_eps : float, optional
            Finite-difference step. Defaults to ref.std().
        fd_acc : int, optional
            Finite-difference accuracy order (must be a positive even integer). Defaults to 2.
        namespace : str, optional
            Namespace prefix prepended to the parsed name.
        depends : dict, optional
            Maps {placeholder} names in the derived expression to Parameter objects.
        """
        if isinstance(name, Parameter):
            self.__dict__.update(name.__dict__)
            self.depends = dict(name.depends)
            if isinstance(self._derived, str) and self._derived not in self._solved_values:
                self._call_fn = self._build_call_fn()
            else:
                self._call_fn = None
            return

        # Parse full name: namespace kwarg (if any) prefixes the embedded namespace in name
        parts = str(name).split(NAMESPACE_SEP)
        basename = parts[-1]
        embedded_ns = NAMESPACE_SEP.join(parts[:-1])
        if namespace:
            ns = NAMESPACE_SEP.join(filter(None, [namespace, embedded_ns]))
        else:
            ns = embedded_ns
        self._name = NAMESPACE_SEP.join([ns, basename]) if ns else basename
        self._latex = latex
        self._call_fn = None  # set early so value setter guard works during __init__

        # Prior
        if prior is None:
            self.prior = ParameterPrior()
        elif isinstance(prior, ParameterPrior):
            self.prior = prior.copy()
        else:
            self.prior = ParameterPrior(**prior)

        self.shape = tuple(shape) if shape else ()

        # Value: explicit, or inferred from prior center
        if value is not None:
            v = np.asarray(value)
            self.value = float(v) if v.shape == () else v
        elif self.prior.is_proper():
            self._value = self.prior.center()
        else:
            self._value = None

        # Ref: defaults to copy of prior
        if ref is None:
            self.ref = self.prior.copy()
        elif isinstance(ref, ParameterPrior):
            self.ref = ref.copy()
        else:
            self.ref = ParameterPrior(**ref)

        # fixed: True when no prior/ref explicitly provided
        if fixed is None:
            fixed = prior is None and ref is None
        self.fixed = bool(fixed)

        # derived expression and depends mapping
        self.depends = dict(depends) if depends else {}
        if isinstance(derived, str):
            self._derived = derived
            self._call_fn = self._build_call_fn() if derived not in self._solved_values else None
        else:
            self._derived = bool(derived)
            self._call_fn = None
        for attr in ('prior', 'ref'):
            p = getattr(self, attr)
            if p.shape is None:
                setattr(self, attr, p.clone(shape=self.shape))
            elif p.shape != self.shape:
                raise ValueError(f'{attr} shape {p.shape} inconsistent with parameter shape {self.shape}')
        if fd_eps is not None:
            if hasattr(fd_eps, '__len__'):
                # 3-tuple (center, eps_below, eps_above) — same convention as desilike_bak delta.
                center_val, eps_below, eps_above = fd_eps
                self._fd_eps = (float(center_val), float(eps_below), float(eps_above))
            else:
                self._fd_eps = float(fd_eps)
        else:
            self._fd_eps = None
        self._fd_acc = int(fd_acc)

    # ── derived property (overrides Variable.derived; read-only after init) ──────

    @property
    def derived(self):
        return self._derived

    # ── value setter override: block direct assignment on derived-expression params ──

    @Variable.value.setter
    def value(self, v):
        if self._call_fn is not None:
            raise AttributeError(f"'{self._name}': cannot set value on a parameter with a derived expression; use update(value=...)")
        Variable.value.fset(self, v)

    def _build_call_fn(self):
        """Compile the derived expression once into a no-arg callable that reads dep.value."""
        code = compile(self._derived, '<derived>', 'eval')
        _ns = {'__builtins__': {}, 'np': np, 'jnp': jnp}
        _deps = dict(self.depends)

        def _fn():
            return eval(code, _ns, {k: dep.value for k, dep in _deps.items()})  # noqa: S307

        return _fn

    def __call__(self):
        """Evaluate derived expression (if any), set and return self.value."""
        if self._call_fn is not None:
            self._value = self._call_fn()
        return self._value

    # ── read-only properties ──────────────────────────────────────────────────
    # basename / namespace / latex() are inherited from Variable.

    @property
    def fd_eps(self):
        """Finite-difference step; falls back to ref.std()."""
        if self._fd_eps is not None:
            return self._fd_eps
        return self.ref.std()

    @property
    def fd_acc(self):
        """Finite-difference accuracy order."""
        return self._fd_acc

    def sample(self, key, shape=None):
        """Draw a sample from the prior; shape defaults to self.shape via prior.shape."""
        return self.prior.sample(key, shape=shape)

    @property
    def varied(self):
        return not self.fixed

    @property
    def solved(self):
        return self._derived in self._solved_values

    @property
    def input(self):
        """Whether parameter should be fed as input to calculator."""
        return ((self._derived is False) or isinstance(self._derived, str)) and not self.depends

    # latex() is inherited from Variable.

    # ── cloning / serialisation ───────────────────────────────────────────────

    def update(self, **kwargs):
        """Re-initialize in-place with overridden attributes; value= bypasses the derived-expression guard."""
        state = self.__getstate__()
        state.update(kwargs)
        self.__init__(**state)
        return self

    def clone(self, **kwargs):
        """Return a copy with selected attributes overridden.

        Passing ``namespace`` replaces the current namespace (rather than prepending).
        """
        state = self.__getstate__()
        if 'namespace' in kwargs:
            ns = kwargs.pop('namespace')
            state['name'] = self.basename  # strip current namespace
            kwargs['namespace'] = ns
        state.update(kwargs)
        return Parameter(**state)

    def __copy__(self):
        new = object.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.depends = dict(self.depends)
        new.prior = self.prior.copy()
        new.ref = self.ref.copy()
        return new

    def __getstate__(self, to_file=False):
        state = {
            'name': self._name,
            'prior': self.prior if to_file else self.prior.__getstate__(),
            'ref': self.ref if to_file else self.ref.__getstate__(),
            'latex': self._latex,
            'fixed': self.fixed,
            'derived': self._derived,
            'shape': list(self.shape) if to_file else self.shape,
            'fd_eps': self._fd_eps,
            'fd_acc': self._fd_acc,
            'depends': {k: dep.name for k, dep in self.depends.items()} if to_file else dict(self.depends),
        }
        if to_file:
            file_state = {'attrs': {'meta': json.dumps({'__class__': 'Parameter', **state}, cls=ParameterEncoder)}}
            if self._value is not None:
                file_state['value'] = np.asarray(self._value)
            return file_state
        state['value'] = self._value
        return state

    def __setstate__(self, state):
        if 'attrs' in state:
            # file format: metadata in JSON, value as numpy dataset
            raw = state['attrs'].get('meta', '{}')
            if isinstance(raw, bytes):
                raw = raw.decode()
            meta = json.loads(raw, object_hook=_parameter_object_hook)
            self.__init__(
                name=meta['name'], value=state.get('value'),
                prior=meta.get('prior'), ref=meta.get('ref'),
                latex=meta.get('latex'), fixed=meta.get('fixed', True),
                derived=meta.get('derived', False),
                shape=tuple(meta.get('shape', [])),
                fd_eps=meta.get('fd_eps'),
                fd_acc=meta.get('fd_acc', 2), depends={})
            # depends resolved later by VariableCollection.__setstate__
        else:
            # in-memory format: feed directly to __init__
            self.__init__(**state)

    def __repr__(self):
        return f'Parameter({self._name!r}, {"fixed" if self.fixed else "varied"})'


@register_type
class VariableCollection:
    """Ordered collection of Variable (or Parameter) instances.

    Accepts at construction:
    - dict-of-dicts: {'omega_m': {'value': 0.3, 'prior': {...}}, ...}
    - dict-of-scalars: {'omega_m': 0.3, ...}  (backward compat)
    - list of Variable/Parameter or dicts
    - another VariableCollection (shallow copy)
    """

    _name = 'VariableCollection'

    def __init__(self, data=None):
        self._data = []
        if data is None:
            return
        if isinstance(data, VariableCollection):
            self._data = list(data._data)
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, Variable):
                    self.set(item)
                elif isinstance(item, dict):
                    self.set(Parameter(**item))
                else:
                    raise ValueError(f'Cannot interpret {item!r} as Variable')
            return
        if isinstance(data, dict):
            for name, conf in data.items():
                if isinstance(conf, Variable):
                    self.set(conf)
                elif isinstance(conf, dict):
                    self.set(Parameter(name=name, **conf))
                else:
                    # scalar or array value: backward-compatible {'omega_m': 0.3}
                    self.set(Parameter(name=name, value=conf))
            return
        raise ValueError(f'Cannot construct VariableCollection from {type(data).__name__}')

    def set(self, param):
        """Insert or replace a variable by name."""
        if not isinstance(param, Variable):
            raise ValueError(f'{param!r} is not a Variable')
        for i, p in enumerate(self._data):
            if p.name == param.name:
                self._data[i] = param
                return
        self._data.append(param)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        for p in self._data:
            if p.name == key:
                return p
        raise KeyError(f'Variable {key!r} not found')

    def __contains__(self, item):
        name = item.name if isinstance(item, Variable) else str(item)
        return any(p.name == name for p in self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def names(self, **kwargs):
        """Return list of variable names, optionally after select(**kwargs)."""
        if kwargs:
            return [p.name for p in self.select(**kwargs)]
        return [p.name for p in self._data]

    def select(self, **kwargs):
        """Return new collection containing variables whose attributes match all kwargs.

        String-valued attributes ``name``, ``basename``, and ``namespace``
        support ``*`` wildcards and ``[start:stop:step]`` index range patterns
        (see :func:`find_names`).  All other attributes are matched with
        equality; if the supplied value is a sequence each element is tried.

        Examples
        --------
        >>> col.select(name='omega_*')           # wildcard
        >>> col.select(name='a_[0:3]')           # index range 0,1,2
        >>> col.select(derived=False)            # exact match
        """
        _name_attrs = {'name', 'basename', 'namespace'}
        result = type(self)()
        result._data = []
        allnames_cache = {}  # cached per string attribute key
        for p in self._data:
            match = True
            for key, value in kwargs.items():
                param_value = getattr(p, key, None)
                if key in _name_attrs:
                    # Build the per-key candidate list once
                    if key not in allnames_cache:
                        allnames_cache[key] = [getattr(q, key, None) for q in self._data]
                    candidates = allnames_cache[key]
                    matched_names = find_names(candidates, value)
                    key_match = param_value in matched_names
                else:
                    key_match = (value == param_value)
                    if not key_match:
                        try:
                            key_match = any(v == param_value for v in value)
                        except TypeError:
                            pass
                if not key_match:
                    match = False
                    break
            if match:
                result._data.append(p)
        return result

    def __add__(self, other):
        """Merge two collections; variables in other override those with the same name."""
        result = VariableCollection(self)
        for p in VariableCollection(other):
            result.set(p)
        return result

    def __sub__(self, other):
        """Return collection with variables from other removed."""
        other = VariableCollection(other)
        result = VariableCollection()
        for p in self._data:
            if p not in other:
                result._data.append(p)
        return result

    def __repr__(self):
        return f'VariableCollection({self.names()})'

    # ── serialisation ─────────────────────────────────────────────────────────

    def __getstate__(self, to_file=False):
        """Return a state dict for this collection.

        Parameters
        ----------
        to_file : bool
            When ``False`` (default) each entry is a plain Python dict suitable
            for in-memory copy/pickle.  When ``True`` each entry uses the
            file-ready format produced by :meth:`Variable.__getstate__` /
            :meth:`Parameter.__getstate__` (numpy value as a native dataset,
            all other metadata as a JSON string); a ``'__names__'`` array
            preserves insertion order and a root ``'attrs'`` carries the class
            tag used by :func:`~desilike.utils.read` for dispatch.
        """
        state = {}
        if to_file:
            state['attrs'] = {'__class__': self._name}
            state['__names__'] = np.array([p.name for p in self._data])
        for p in self._data:
            state[p.name] = p.__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        """Populate from a state dict produced by :meth:`__getstate__`."""
        self._data = []
        is_file = 'attrs' in state and '__class__' in state.get('attrs', {})

        # Determine ordered names
        if is_file:
            names = [str(n) for n in state.get('__names__', [])]
        else:
            names = [k for k in state if k not in ('__names__', 'attrs')]

        # Helper: determine Variable vs Parameter from a sub-state
        def _is_parameter(pstate):
            if 'attrs' in pstate:
                raw = pstate['attrs'].get('meta', '{}')
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw).get('__class__') == 'Parameter'
            return 'prior' in pstate

        # First pass: build objects (depends left empty for file format)
        params_by_name = {}
        for name in names:
            pstate = state[name]
            cls = Parameter if _is_parameter(pstate) else Variable
            p = cls.__new__(cls)
            p.__setstate__(pstate)
            params_by_name[p.name] = p

        # Second pass: resolve depends (file format stores them as name-strings)
        if is_file:
            for name in names:
                pstate = state[name]
                if not _is_parameter(pstate):
                    continue
                raw = pstate['attrs'].get('meta', '{}')
                if isinstance(raw, bytes):
                    raw = raw.decode()
                dep_names = json.loads(raw).get('depends', {})
                if dep_names:
                    resolved = {k: params_by_name[n] for k, n in dep_names.items() if n in params_by_name}
                    if resolved:
                        params_by_name[name].update(depends=resolved)

        self._data = [params_by_name[n] for n in names if n in params_by_name]

    def write(self, filename):
        """Write to an HDF5 or text file.

        The format is determined by the file extension:

        - ``.h5`` / ``.hdf5`` — HDF5 via h5py; each parameter's value is stored
          as a native dataset and all other metadata as a JSON string attribute.
        - ``.txt`` — directory of ``.txt`` / ``.json`` files (useful for
          human-readable inspection).

        Parameters
        ----------
        filename : str
        """
        _utils_write(filename, self)

    @classmethod
    def read(cls, filename):
        """Read from an HDF5 or text file written by :meth:`write`.

        Parameters
        ----------
        filename : str

        Returns
        -------
        VariableCollection
        """
        return _utils_read(filename)


def expand_dict(di, names):
    """Expand a (possibly wildcard) dict to cover all *names*.

    Parameters
    ----------
    di : dict, sequence, or scalar
        Input mapping.  A bare scalar is treated as ``{'*': di}`` (applied to
        all names).  A sequence is zipped with *names*.  Wildcard ``*`` in
        keys matches any suffix.
    names : list of str
        Target parameter names.

    Returns
    -------
    dict
        A dict with exactly the keys in *names*.

    Examples
    --------
    >>> expand_dict({'*': 2}, ['a', 'b'])
    {'a': 2, 'b': 2}
    >>> expand_dict({'a*': 2, 'b': 1}, ['a1', 'a2', 'b'])
    {'a1': 2, 'a2': 2, 'b': 1}
    """
    toret = dict.fromkeys(names)
    if isinstance(di, (list, tuple)):
        di = dict(zip(names, di))
    if not hasattr(di, 'items'):
        di = {'*': di}
    for template, value in di.items():
        for matched_name in find_names(names, template):
            toret[matched_name] = value
    return toret
