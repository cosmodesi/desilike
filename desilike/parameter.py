"""Parameter classes for desilike."""

import re
import copy
import threading
import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats as sp

_compile_context = threading.local()


class _CompileContext:
    def __init__(self):
        self.traced = set()           # id(node) seen during __post_init__ (phase 1)
        self.post_init_called = set() # id(node) whose __post_init__ has already run this compile
        self.stack = []               # currently-tracing Calculator stack
        self.node_deps = {}           # id(node) -> list[Node], in access order, deduplicated
        self.node_order = []          # topological order: leaves first, root last
        self.call_returns = {}        # id(node) -> return value of node.__call__()
        self.phase = 'post_init'      # 'post_init' or 'call'
        self.call_activated = set()   # id(Calculator) nodes accessed during __call__ (phase 2)

jax.config.update('jax_enable_x64', True)

NAMESPACE_SEP = '.'


class Node:
    """Common base for mutable objects traced in the pipeline."""

    _is_calculator = False
    _updated = False  # set to True by update(); reset to False by CompiledGraph.__init__

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if not name.startswith('_'):
            ctx = getattr(_compile_context, 'ctx', None)
            if ctx is not None and ctx.phase == 'call' and ctx.stack and ctx.stack[-1] is self:
                nodes = []
                if isinstance(value, Node):
                    nodes = [value]
                elif isinstance(value, (list, tuple)):
                    nodes = [v for v in value if isinstance(v, Node)]
                for node in nodes:
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

    def __init__(self, name, value=None, derived=False):
        self._name = str(name)
        self._derived = bool(derived)
        if value is not None:
            v = np.asarray(value)
            self.shape = v.shape
            self.value = float(v) if v.shape == () else v
        else:
            self.shape = ()
            self._value = None

    @property
    def name(self):
        return self._name

    @property
    def derived(self):
        return self._derived

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
        self.__init__(
            kwargs.get('name', self._name),
            value=kwargs.get('value', self._value),
            derived=kwargs.get('derived', self._derived),
        )
        if getattr(_compile_context, 'ctx', None) is None:
            self._updated = True
        return self

    # Minimal FD defaults so ExternalCalculator works when Variable is a dep.
    @property
    def fd_eps(self):
        return None

    @property
    def fd_acc(self):
        return 2

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
                finite = [l for l in (lo, hi) if np.isfinite(l)]
                self._center = float(np.mean(finite)) if finite else 0.
                self._std = None
            else:
                lo_, hi_ = lo, hi
                def _logpdf(x):
                    return jax.scipy.stats.uniform.logpdf(jnp.asarray(x), loc=lo_, scale=hi_ - lo_)
                def _sample(key, shape):
                    return jax.random.uniform(key, shape=shape, minval=lo_, maxval=hi_, dtype=jnp.float64)
                self._logpdf_fn = _logpdf
                self._sample_fn = _sample
                self._center = (lo + hi) / 2.
                self._std = (hi - lo) / float(np.sqrt(12.))
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
                self._center, self._std = float(rv.mean()), float(rv.std())
            self._logpdf_fn = _logpdf
            self._sample_fn = _sample
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

        # sampling via inverse CDF (scipy ppf, result converted to JAX)
        rv_sp = getattr(sp, dist)(**attrs)
        p_lo_ = float(rv_sp.cdf(lo) if np.isfinite(lo) else 0.)
        p_hi_ = float(rv_sp.cdf(hi) if np.isfinite(hi) else 1.)
        def _sample(key, shape, _rv=rv_sp, _plo=p_lo_, _phi=p_hi_):
            u = jax.random.uniform(key, shape=shape, minval=_plo, maxval=_phi, dtype=jnp.float64)
            return jnp.asarray(_rv.ppf(np.asarray(u)), dtype=jnp.float64)
        self._sample_fn = _sample

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
            self._std = s if np.isfinite(s) else None
        except Exception:
            self._center, self._std = 0., None

    def logpdf(self, x):
        """JAX-differentiable log probability density at x."""
        return self._logpdf_fn(jnp.asarray(x))

    def sample(self, key, shape=None):
        """Draw samples using JAX PRNG key; raises if prior is improper.

        shape defaults to self.shape (set by Parameter) when None; falls back to ().
        """
        if self._sample_fn is None:
            raise ValueError('Cannot sample from improper prior')
        if shape is None:
            shape = self.shape if self.shape is not None else ()
        if isinstance(shape, int):
            shape = (shape,)
        return self._sample_fn(key, shape)

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
        if self.shape is not None:
            parts.append(f'shape={self.shape}')
        parts += [f'{k}={v}' for k, v in self.attrs.items()]
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
                 derived=False, shape=(), proposal=None, fd_eps=None, fd_acc=2,
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
        proposal : float, optional
            MCMC proposal scale. Defaults to ref.std().
        fd_eps : float, optional
            Finite-difference step. Defaults to proposal.
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
        self._proposal = float(proposal) if proposal is not None else None
        self._fd_eps = float(fd_eps) if fd_eps is not None else None
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

    @property
    def basename(self):
        return self._name.split(NAMESPACE_SEP)[-1]

    @property
    def namespace(self):
        parts = self._name.split(NAMESPACE_SEP)
        return NAMESPACE_SEP.join(parts[:-1]) if len(parts) > 1 else ''

    @property
    def proposal(self):
        """MCMC proposal scale; falls back to ref.std()."""
        if self._proposal is not None:
            return self._proposal
        return self.ref.std()

    @property
    def fd_eps(self):
        """Finite-difference step; falls back to proposal."""
        if self._fd_eps is not None:
            return self._fd_eps
        return self.proposal

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

    # ── latex ─────────────────────────────────────────────────────────────────

    def latex(self, namespace=False, inline=False):
        """Return LaTeX representation, optionally with namespace subscript and $ delimiters."""
        if self._latex is not None:
            s = self._latex
            if namespace and self.namespace:
                ns = self.namespace
                m1 = re.match(r'(.*)_(.)$', s)
                m2 = re.match(r'(.*)_{(.*)}$', s)
                if m1:
                    s = r'%s_{%s,\mathrm{%s}}' % (m1.group(1), m1.group(2), ns)
                elif m2:
                    s = r'%s_{%s,\mathrm{%s}}' % (m2.group(1), m2.group(2), ns)
                else:
                    s = r'%s_{\mathrm{%s}}' % (s, ns)
            return f'${s}$' if inline else s
        return self._name

    # ── cloning / serialisation ───────────────────────────────────────────────

    def update(self, **kwargs):
        """Re-initialize in-place with overridden attributes; value= bypasses the derived-expression guard."""
        state = self.__getstate__()
        state.update(kwargs)
        self.__init__(**state)
        if getattr(_compile_context, 'ctx', None) is None:
            self._updated = True
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

    def __getstate__(self):
        return {
            'name': self._name,
            'value': self._value,
            'prior': self.prior.__getstate__(),
            'ref': self.ref.__getstate__(),
            'latex': self._latex,
            'fixed': self.fixed,
            'derived': self._derived,
            'shape': self.shape,
            'proposal': self._proposal,
            'fd_eps': self._fd_eps,
            'fd_acc': self._fd_acc,
            'depends': dict(self.depends),
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return f'Parameter({self._name!r}, {"fixed" if self.fixed else "varied"})'


class VariableCollection:
    """Ordered collection of Variable (or Parameter) instances.

    Accepts at construction:
    - dict-of-dicts: {'omega_m': {'value': 0.3, 'prior': {...}}, ...}
    - dict-of-scalars: {'omega_m': 0.3, ...}  (backward compat)
    - list of Variable/Parameter or dicts
    - another VariableCollection (shallow copy)
    """

    def __init__(self, data=None):
        self._data = []
        if data is None:
            return
        if isinstance(data, VariableCollection):
            self._data = [copy.copy(p) for p in data._data]
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, Variable):
                    self.set(copy.copy(item))
                elif isinstance(item, dict):
                    self.set(Parameter(**item))
                else:
                    raise ValueError(f'Cannot interpret {item!r} as Variable')
            return
        if isinstance(data, dict):
            for name, conf in data.items():
                if isinstance(conf, Variable):
                    self.set(copy.copy(conf))
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
        """Return new collection containing variables whose attributes match all kwargs."""
        result = VariableCollection()
        for p in self._data:
            if all(getattr(p, k, None) == v for k, v in kwargs.items()):
                result._data.append(p)
        return result

    def __add__(self, other):
        """Merge two collections; variables in other override those with the same name."""
        result = VariableCollection(self)
        for p in VariableCollection(other):
            result.set(copy.copy(p))
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
