"""
JAX-friendly calculator pipeline for desilike.

Two base classes:
- Calculator: pure JAX ops, fully traceable by jit/vmap/grad.
- ExternalCalculator: arbitrary Python/numpy, wrapped via pure_callback +
  custom_vjp (finite-difference backward pass) so that jit/vmap/grad all work.

Lazy initialization:
- Calculator(*args, **kwargs) stores args without calling __post_init__(). If a subclass defines __init__, args are saved automatically.
- CompiledGraph calls __post_init__() during graph construction, then scans
  instance attributes for Node objects to discover dependencies.

__call__() interface:
- __call__(self) reads own params and dep outputs directly from self (e.g. self.A, self.cosmo.growth_factor).
- The pipeline sets param attributes on self before invoking __call__().
- __call__() sets named output attributes on self (e.g. self.growth_factor).
- __call__() may return any value (including self); that value
  is forwarded as the pipeline output if this node is the root. tree_flatten()
  defines which outputs are passed to downstream nodes.

tree_flatten / tree_unflatten:
- Each calculator must define tree_flatten(self) -> (children, aux) and
  tree_unflatten(cls, aux, children) -> instance. children are the output
  arrays produced by __call__(). The framework uses these to pass outputs between
  calculators and to define the ExternalCalculator dep-passing interface.

Pipeline:
- CompiledGraph: builds a static DAG once, exposes a pure
  __call__(params) compatible with jax.jit / jax.vmap / jax.grad.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict

jax.config.update('jax_enable_x64', True)

from .parameter import Node, Variable, Parameter, VariableCollection, _compile_context, _CompileContext


# ── base classes ──────────────────────────────────────────────────────────────

class Calculator(Node):
    """
    Base class for calculators implemented with JAX ops.

    Subclasses define:
      __post_init__(*args, **kwargs): wire dependencies and parameters by setting
        Calculator instances (deps) and Variable/Parameter instances (params) as attributes.
        Any Node-typed attribute (or list/tuple of Nodes) accessed during __post_init__
        or __call__ is auto-registered as a dependency during compile().
      __call__(self): read params via self.param.value and dep outputs via self.dep.attr;
        compute and store output attributes; return the output value.
      tree_flatten(self) -> (children, aux): children = list of output arrays,
        aux = static data needed by tree_unflatten.
      tree_unflatten(cls, aux, children) -> instance: reconstruct an instance
        carrying only the output attrs (no dep refs, no init args).

    Instantiation is lazy: __init__ stores args/kwargs and does NOT call __post_init__().
    compile() calls __post_init__() then __call__() to discover dependencies.
    """

    _is_calculator = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(
            cls,
            lambda node: node.tree_flatten(),
            cls.tree_unflatten,
        )
        if '__init__' in cls.__dict__:
            _orig = cls.__dict__['__init__']
            @functools.wraps(_orig)
            def _wrapped(self, *args, _f=_orig, **kwargs):
                self._init = (args, kwargs)
                _f(self, *args, **kwargs)
            cls.__init__ = _wrapped
        if '__post_init__' in cls.__dict__:
            _orig_pi = cls.__dict__['__post_init__']
            @functools.wraps(_orig_pi)
            def _wrapped_pi(self, *args, _f=_orig_pi, **kwargs):
                _f(self, *args, **kwargs)
                ctx = getattr(_compile_context, 'ctx', None)
                if ctx is not None:
                    ctx.post_init_called.add(id(self))
            cls.__post_init__ = _wrapped_pi

    def __init__(self, *args, **kwargs):
        self._init = (args, kwargs)

    def update(self, *args, **kwargs):
        """Re-initialize in-place with overridden arguments; new kwargs override old ones."""
        old_args, old_kwargs = self._init
        merged_args = args if args else old_args
        merged_kwargs = {**old_kwargs, **kwargs}
        self.__init__(*merged_args, **merged_kwargs)
        if getattr(_compile_context, 'ctx', None) is None:
            self._updated = True

    def __post_init__(self, *args, **kwargs):
        pass

    def __call__(self):
        raise NotImplementedError

    def tree_flatten(self):
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux, children):
        raise NotImplementedError


class ExternalCalculator(Calculator):
    """
    Base class for non-JAX calculators.
    __call__() may use arbitrary Python/numpy, accessing dep outputs as self.dep.attr
    (concrete numpy arrays when the callback executes).

    Wrapped via jax.pure_callback (for jit/vmap) + jax.custom_jvp
    (finite-difference JVP for grad/jacfwd/hessian).

    Per-parameter FD step and accuracy are taken from param.fd_eps and param.fd_acc.
    param.fd_eps falls back through: explicit → param.proposal → param.ref.std() → 1e-5.
    param.fd_acc defaults to 2; set to 4, 6, ... for higher-accuracy stencils.
    """


class Likelihood(Calculator):
    """
    Base class for likelihood calculators.

    Subclasses implement __post_init__() and __call__(). __call__() must set self.logpdf.
    tree_flatten/tree_unflatten are provided here; subclasses need not repeat them.
    """

    def tree_flatten(self):
        return [self.logpdf], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf = children[0]
        return obj


class GaussianLikelihood(Likelihood):
    """
    Base class for Gaussian chi-squared likelihoods.

    Subclasses must implement:
      __post_init__(): set self.flatdata (1D array, the observations) and self.precision
        (2D array, C⁻¹), along with any Calculator deps and Variable/Parameter instances.
      __call__(): set self.flattheory (1D JAX array), then call super().__call__() to compute
        self.logpdf = -½ (flatdata - flattheory)ᵀ precision (flatdata - flattheory).

    tree_flatten exposes [logpdf, flattheory, precision] so downstream nodes can
    access them as dep outputs.
    """

    def __call__(self):
        r = self.flatdata - self.flattheory
        self.logpdf = -0.5 * r @ self.precision @ r
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf, self.flattheory, self.precision], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf, obj.flattheory, obj.precision = children
        return obj

    def _gaussians(self):
        return [self]


class SumLikelihood(Likelihood):
    """
    Sums logpdf from multiple Likelihood components.

    __post_init__(*likelihoods): each argument must be a Likelihood instance.
    __call__(): sums logpdf across components (deps are called before this node in the pipeline).
    _gaussians(): recursively collects GaussianLikelihood leaves so Posterior can
      perform per-component analytic marginalization, skipping jacfwd for components
      that don't depend on any solved parameter.
    """

    def __post_init__(self, *likelihoods):
        self.likelihoods = list(likelihoods)

    def __call__(self):
        self.logpdf = sum(like.logpdf for like in self.likelihoods)
        return self.logpdf

    def _gaussians(self):
        result = []
        for like in self.likelihoods:
            if hasattr(like, '_gaussians'):
                result.extend(like._gaussians())
        return result


class Prior(Calculator):
    """
    Sums log-prior probabilities over non-fixed parameters.

    __post_init__(*args, **kwargs) collects Parameter arguments into self.params (a list).
    Positional args may be VariableCollection instances; keyword args are individual
    Parameters. Fixed parameters are silently skipped (they contribute 0).

    __call__() returns a scalar: sum of param.prior.logpdf(param) over non-fixed params.
    Returns -inf when any parameter is outside its prior support.

    To include a Prior in a pipeline it must be a dependency of the root node
    (directly or transitively). Typically users create a thin Calculator that
    takes both likelihood and prior as deps and returns their sum.
    """

    def __post_init__(self, *args, **kwargs):
        seen_ids = set()
        params = []
        for arg in args:
            if isinstance(arg, VariableCollection):
                for p in arg:
                    if id(p) not in seen_ids:
                        seen_ids.add(id(p))
                        params.append(p)
        for p in kwargs.values():
            if isinstance(p, Parameter):
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    params.append(p)
        self.params = params

    def __call__(self):
        logprior = jnp.zeros(())
        for p in self.params:
            if not p.fixed:
                logprior = logprior + p.prior.logpdf(p)
        self.logpdf = logprior
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf = children[0]
        return obj


class Posterior(Calculator):
    """
    Log-posterior = log-likelihood + log-prior, with optional analytic treatment of solved params.

    __post_init__(likelihood, prior) compiles both into private internal pipelines. The
    likelihood's parameters are surfaced via inputs() so the outer compiled pipeline
    manages them uniformly.

    Solved parameters (derived='marg' or derived='best') are handled analytically for
    GaussianLikelihood / SumLikelihood theories that are linear in those params.

    derived='marg': Gaussian marginalization — the parameter is integrated out with its
      prior (param.prior.std() required). Produces the correct Bayesian evidence including
      the volume factor  + ½ log|P_α| − ½ log|F_eff|.

    derived='best': profile likelihood — the parameter is set to its best-fit (MLE) value.
      No prior required; no volume factor included.

    Mixed 'marg'/'best' is supported: the Schur-complement formula is used so the 'best'
    block contributes no volume correction.

    Marginalization is performed per Gaussian component: jacfwd is only computed for
    components whose theory actually depends on the solved parameters.

    __call__() evaluates the prior first (for non-solved params). In eager mode the
    likelihood is skipped when logprior == -inf. Under JAX tracing the full computation
    is always built.
    """

    def __post_init__(self, likelihood, prior):
        self._likelihood = compile(likelihood)
        self._solved_params = self._likelihood.params.select(solved=True)

        self.likelihood_params = list(self._likelihood.params)

        if self._solved_params:
            gaussians = likelihood._gaussians() if hasattr(likelihood, '_gaussians') else []
            if not gaussians:
                raise ValueError('Analytic marginalization requires a GaussianLikelihood (or SumLikelihood of them)')

            alpha_names = [p.name for p in self._solved_params]
            n_alpha = len(alpha_names)
            marg_global = {i for i, p in enumerate(self._solved_params) if p.derived == 'marg'}
            best_global = {i for i, p in enumerate(self._solved_params) if p.derived == 'best'}

            # Validate 'marg' prior scales and collect them.
            marg_scales = {}
            for i, p in enumerate(self._solved_params):
                if p.derived == 'marg':
                    s = p.prior.std() if p.prior is not None else None
                    if s is None:
                        raise ValueError(f'Parameter {p.name!r} has derived="marg" but its prior has no finite std')
                    marg_scales[i] = s

            # Build per-gaussian-component list: (theory_pipe, precision, flatdata, alpha_idx).
            components = []
            for g in gaussians:
                theory = compile(g, output=lambda g=g: g.flattheory)
                comp_param_names = set(theory.params.names())
                alpha_idx = [i for i, p in enumerate(self._solved_params) if p.name in comp_param_names]
                components.append((theory, g.precision, g.flatdata, alpha_idx))

            # Components with no solved-param dependence are handled separately.
            self._no_alpha_components = [(t, p, d) for t, p, d, ai in components if not ai]
            alpha_components = [(t, p, d, ai) for t, p, d, ai in components if ai]

            # Union-find: group alpha indices that appear together in any component.
            parent = list(range(n_alpha))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            for _, _, _, alpha_idx in alpha_components:
                for i in range(1, len(alpha_idx)):
                    union(alpha_idx[0], alpha_idx[i])

            # Collect and sort global alpha indices per group root.
            root_globals = defaultdict(set)
            for _, _, _, alpha_idx in alpha_components:
                root_globals[find(alpha_idx[0])].update(alpha_idx)
            root_sorted = {r: sorted(gs) for r, gs in root_globals.items()}

            # Remap each component's alpha_idx to local (group-relative) indices.
            root_comps = defaultdict(list)
            for theory_pipe, precision, flatdata, alpha_idx in alpha_components:
                root = find(alpha_idx[0])
                global_idx = root_sorted[root]
                g_to_l = {g: l for l, g in enumerate(global_idx)}
                root_comps[root].append((theory_pipe, precision, flatdata, [g_to_l[g] for g in alpha_idx]))

            # Build one descriptor per independent group.
            self._groups = []
            for root, global_idx in root_sorted.items():
                comps = root_comps[root]
                marg_local = np.array([j for j, g in enumerate(global_idx) if g in marg_global], dtype=int)
                best_local = np.array([j for j, g in enumerate(global_idx) if g in best_global], dtype=int)
                pp_g = jnp.diag(jnp.array([1.0 / marg_scales[global_idx[j]] ** 2 for j in marg_local])) if marg_local.size else None
                group_alpha_names = [alpha_names[g] for g in global_idx]
                self._groups.append((group_alpha_names, comps, marg_local, best_local, pp_g))

            prior.update(self._likelihood.params.select(solved=False))
        else:
            prior.update(self._likelihood.params)

        self._prior = compile(prior)

    def _marg_loglik(self, params):
        """Profile/marginalize over solved params, one independent group at a time."""
        logL = jnp.zeros(())

        # Components with no solved-param dependence: contribute only to logL.
        for theory_pipe, precision, flatdata in self._no_alpha_components:
            comp_params = {p.name: jnp.asarray(params[p.name]) for p in theory_pipe.params}
            theory = theory_pipe(comp_params)[0]
            r = flatdata - theory
            logL = logL - 0.5 * r @ (precision @ r)

        # Per-group: independent block solve of size n_g × n_g.
        for group_alpha_names, comps, marg_local, best_local, pp_g in self._groups:
            n_g = len(group_alpha_names)
            F_g = jnp.zeros((n_g, n_g))
            b_g = jnp.zeros(n_g)

            for theory_pipe, precision, flatdata, local_idx in comps:
                comp_params = {p.name: jnp.asarray(params[p.name]) for p in theory_pipe.params}
                comp_alpha_names = [group_alpha_names[j] for j in local_idx]
                comp_alpha_vals = jnp.stack([comp_params[name] for name in comp_alpha_names])

                def theory_fn(alpha_vec, _pipe=theory_pipe, _cp=comp_params, _names=comp_alpha_names):
                    p = {**_cp, **{name: alpha_vec[i] for i, name in enumerate(_names)}}
                    return _pipe(p)[0]

                theory = theory_fn(comp_alpha_vals)
                B = jax.jacfwd(theory_fn)(comp_alpha_vals)      # (n_data, len(local_idx))
                r = flatdata - theory
                BtP = B.T @ precision
                ix = np.array(local_idx)
                F_g = F_g.at[ix[:, None], ix[None, :]].add(BtP @ B)
                b_g = b_g.at[ix].add(BtP @ r)
                logL = logL - 0.5 * r @ (precision @ r)

            # Add prior precision for 'marg' params in this group.
            if marg_local.size:
                F_g = F_g.at[marg_local[:, None], marg_local[None, :]].add(pp_g)

            logL = logL + 0.5 * b_g @ jnp.linalg.solve(F_g, b_g)

            # Volume factor: only 'marg' params contribute.
            # Mixed case uses Schur complement: + ½ log|P_α| − ½ log|F_g| + ½ log|F_g[best,best]|.
            if marg_local.size:
                _, logdet_Pa = jnp.linalg.slogdet(pp_g)
                _, logdet_F = jnp.linalg.slogdet(F_g)
                if best_local.size:
                    _, logdet_F_bb = jnp.linalg.slogdet(F_g[best_local[:, None], best_local[None, :]])
                    logL = logL + 0.5 * logdet_Pa - 0.5 * logdet_F + 0.5 * logdet_F_bb
                else:
                    logL = logL + 0.5 * logdet_Pa - 0.5 * logdet_F

        return logL

    def __call__(self):
        params = {p.name: p.value for p in self.likelihood_params}
        logprior = self._prior(params)[0]
        is_tracing = isinstance(logprior, jax.core.Tracer)
        if is_tracing or not bool(jnp.isneginf(logprior)):
            loglik = self._marg_loglik(params) if self._solved_params else self._likelihood(params)[0]
            self.logpdf = logprior + loglik
        else:
            self.logpdf = jnp.full((), -jnp.inf)
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf = children[0]
        return obj


def _fd_stencil(order, acc):
    """
    Uniform centered finite-difference stencil for the order-th derivative at accuracy acc.

    Returns (offsets, coeffs): integer offsets and weights such that
      f^(order)(x) ≈ sum(coeffs[i] * f(x + offsets[i] * h)) / h^order.
    Zero-weight points (e.g. center for odd-order derivatives) are omitted.

    Adapted from the coefficient construction in the original desilike/differentiation.py.
    """
    import math
    nside = (order + acc - 1) // 2
    offsets = np.arange(-nside, nside + 1)
    # Vandermonde system: sum_j c_j * j^k = order! * delta(k, order)
    A = np.array([[float(o) ** k for o in offsets] for k in range(len(offsets))])
    b = np.zeros(len(offsets))
    b[order] = float(math.factorial(order))
    coeffs = np.linalg.solve(A, b)
    mask = np.abs(coeffs) > 1e-12
    return offsets[mask], coeffs[mask]


# ── external function factory ─────────────────────────────────────────────────

def _make_external_fn(node: ExternalCalculator, params_list: list, calc_deps: list, call_return, node_state: dict, dep_states: list):
    """
    Return (fn_dep, fn_call) — two pure_callback-wrapped callables for node().

    fn_dep  -> tree_flatten children (used to pass outputs downstream).
    fn_call -> __call__() return value (used when this node is the pipeline root).

    Differentiation is handled at the CompiledGraph level via _build_graph_call_fn;
    these functions carry no custom_jvp of their own.
    node() is skipped when own params and all dep outputs are unchanged.
    """
    dep_schema = []
    for dep in calc_deps:
        dep_children, dep_aux = dep.tree_flatten()
        dep_schema.append((dep, len(dep_children), dep_aux))

    own_children, _ = node.tree_flatten()
    dep_sdt = tuple(jax.ShapeDtypeStruct(np.asarray(c).shape, np.asarray(c).dtype) for c in own_children)

    if isinstance(call_return, tuple):
        call_sdt = tuple(jax.ShapeDtypeStruct(np.asarray(r).shape, np.asarray(r).dtype) for r in call_return)
    else:
        call_sdt = (jax.ShapeDtypeStruct(np.asarray(call_return).shape, np.asarray(call_return).dtype),)
    call_result_sdt = call_sdt[0] if len(call_sdt) == 1 else call_sdt

    def _run_or_cache(own_params_tuple, dep_args):
        dep_was_called = any(s['was_called'] for s in dep_states)
        last = node_state['last_params']
        params_changed = last is None or any(not np.array_equal(a, b) for a, b in zip(own_params_tuple, last))
        if dep_was_called or params_changed:
            node_state['last_params'] = tuple(np.asarray(a) for a in own_params_tuple)
            for i, param in enumerate(params_list):
                param.value = np.asarray(own_params_tuple[i])
            offset = 0
            for dep, n_children, dep_aux in dep_schema:
                proxy = dep.__class__.tree_unflatten(dep_aux, list(dep_args[offset:offset + n_children]))
                dep.__dict__.update(proxy.__dict__)
                offset += n_children
            call_result = node()
            node_state['dep_result'] = tuple(np.asarray(c) for c in node.tree_flatten()[0])
            node_state['call_result'] = call_result
            node_state['was_called'] = True
        else:
            node_state['was_called'] = False

    def _inject_and_call_dep(own_params_tuple, *dep_args):
        _run_or_cache(own_params_tuple, dep_args)
        return node_state['dep_result']

    def _inject_and_call_call(own_params_tuple, *dep_args):
        _run_or_cache(own_params_tuple, dep_args)
        call_result = node_state['call_result']
        if isinstance(call_result, tuple):
            return tuple(np.asarray(r) for r in call_result)
        return np.asarray(call_result)

    def _make_fn(callback, result_sdt):
        def fn(own_params_tuple, dep_attr_flat):
            return jax.pure_callback(callback, result_sdt, own_params_tuple, *dep_attr_flat, vmap_method='sequential')
        return fn

    return _make_fn(_inject_and_call_dep, dep_sdt), _make_fn(_inject_and_call_call, call_result_sdt)


# ── graph-level custom JVP ────────────────────────────────────────────────────

def _build_graph_call_fn(pipeline):
    """
    Build and return a jax.custom_jvp-wrapped call function for the CompiledGraph.

    The returned function has signature (fd_params_tuple, jax_params_tuple) ->
    (return_val, derived_dict, ext_outputs_flat).

    JVP strategy:
      fd_params (feed any ExternalCalculator directly or transitively): finite differences
        over the full graph — O(n_fd_params × n_stencil) full-graph evaluations.
      jax_params (only feed JAX calculators): exact forward-mode AD through the JAX
        sub-graph, with External outputs frozen at their primal values.
    """
    nodes = pipeline.nodes
    node_var_deps = pipeline._node_var_deps
    node_calc_deps = pipeline._node_calc_deps
    node_states = pipeline._node_states
    tree_own_aux = pipeline._tree_own_aux
    fn_dep = pipeline._fn_dep
    fn_call = pipeline._fn_call
    ext_n_children = pipeline._ext_n_children
    root = pipeline.root
    output = pipeline.output
    all_params = list(pipeline.params)
    derived_params = pipeline._derived_params
    fd_params = pipeline._fd_params
    jax_params = pipeline._jax_params
    fd_names = [p.name for p in fd_params]
    jax_names = [p.name for p in jax_params]

    def _run_graph(params, ext_flat=None):
        """
        Execute the graph.
        ext_flat=None  : full run — External nodes execute via pure_callback.
        ext_flat=tuple : JAX sub-graph run — External outputs taken from ext_flat,
                         External nodes are not called. Used in the JAX-params JVP.
        Returns (return_val, derived_dict, ext_outputs_flat).
        """
        is_tracing = any(isinstance(params[p.name], jax.core.Tracer) for p in all_params)

        if is_tracing:
            saved_param_values = {p.name: p._value for p in all_params}
            saved_node_dicts = {id(n): dict(n.__dict__) for n in nodes}

        ext_collected = []
        result = None
        ext_flat_offset = 0

        for i, node in enumerate(nodes):
            nvd = node_var_deps[id(node)]
            ncd = node_calc_deps[id(node)]

            if isinstance(node, ExternalCalculator):
                n_ch = ext_n_children[id(node)]
                if ext_flat is not None:
                    # JAX sub-graph mode: unpack frozen External outputs.
                    children = list(ext_flat[ext_flat_offset:ext_flat_offset + n_ch])
                    proxy = node.__class__.tree_unflatten(tree_own_aux[i], children)
                    node.__dict__.update(proxy.__dict__)
                    ext_collected.extend(children)
                else:
                    own_params_tuple = tuple(jnp.asarray(params[p.name]) for p in nvd)
                    dep_attr_flat = tuple(v for dep in ncd for v in dep.tree_flatten()[0])
                    # Always call fn_dep to capture tree_flatten children into ext_collected.
                    raw_dep = fn_dep[i](own_params_tuple, dep_attr_flat)
                    proxy = node.__class__.tree_unflatten(tree_own_aux[i], list(raw_dep))
                    node.__dict__.update(proxy.__dict__)
                    ext_collected.extend(raw_dep)
                    if node is root and output is None:
                        # fn_call reuses the cached computation from fn_dep above.
                        result = fn_call[i](own_params_tuple, dep_attr_flat)
                ext_flat_offset += n_ch
            else:
                node_state = node_states[id(node)]
                if is_tracing:
                    for param in nvd:
                        param.value = params[param.name]
                    result = node()
                    node_state['was_called'] = True
                else:
                    own_params_np = np.concatenate([np.ravel(np.asarray(params[p.name])) for p in nvd]) if nvd else np.array([])
                    dep_states_list = [node_states[id(dep)] for dep in ncd]
                    dep_was_called = any(s['was_called'] for s in dep_states_list)
                    params_changed = node_state['last_params'] is None or not np.array_equal(own_params_np, node_state['last_params'])
                    if dep_was_called or params_changed:
                        for param in nvd:
                            param.value = params[param.name]
                        result = node()
                        node_state['last_params'] = own_params_np
                        node_state['last_result'] = result
                        node_state['was_called'] = True
                    else:
                        result = node_state['last_result']
                        node_state['was_called'] = False

        return_val = output() if output is not None else result
        derived_dict = {p.name: p._value for p in derived_params}
        ext_flat_out = tuple(ext_collected)

        if is_tracing:
            for node in nodes:
                node.__dict__.clear()
                node.__dict__.update(saved_node_dicts[id(node)])
            for p in all_params:
                p._value = saved_param_values[p.name]

        return return_val, derived_dict, ext_flat_out

    @jax.custom_jvp
    def call_fn(fd_params_tuple, jax_params_tuple):
        params = {**dict(zip(fd_names, fd_params_tuple)), **dict(zip(jax_names, jax_params_tuple))}
        return _run_graph(params)

    @call_fn.defjvp
    def call_fn_jvp(primals, tangents):
        fd_p, jax_p = primals
        vfd, vjax = tangents

        primal_val, primal_derived, primal_ext = call_fn(fd_p, jax_p)

        # ── FD tangent for fd_params ──────────────────────────────────────────
        # One full-graph call per stencil point per scalar element of each fd_param.
        tangent_val = jnp.zeros_like(primal_val)
        for i, param in enumerate(fd_params):
            eps = param.fd_eps if param.fd_eps is not None else 1e-5
            offsets, coeffs = _fd_stencil(1, param.fd_acc)
            param_arr = jnp.asarray(fd_p[i])
            for idx in np.ndindex(param_arr.shape):
                v_ij = vfd[i][idx] if param_arr.ndim > 0 else vfd[i]
                df = jnp.zeros_like(primal_val)
                for off, coeff in zip(offsets, coeffs):
                    shifted = list(fd_p)
                    shifted[i] = param_arr.at[idx].add(off * eps) if param_arr.ndim > 0 else param_arr + off * eps
                    fi = call_fn(tuple(shifted), jax_p)[0]
                    df = df + coeff * fi
                tangent_val = tangent_val + v_ij * df / eps

        # ── JAX tangent for jax_params ────────────────────────────────────────
        # Forward-mode AD through the JAX sub-graph; External outputs frozen at
        # primal_ext so their contribution is zero (treated as constants).
        if jax_names:
            def _jax_sub(jp):
                params = {**dict(zip(fd_names, fd_p)), **dict(zip(jax_names, jp))}
                return _run_graph(params, ext_flat=primal_ext)
            _, (jax_tv, _, _) = jax.jvp(_jax_sub, (jax_p,), (vjax,))
            tangent_val = tangent_val + jax_tv

        tangent_out = (tangent_val, jax.tree_util.tree_map(jnp.zeros_like, primal_derived), tuple(jnp.zeros_like(e) for e in primal_ext))
        return (primal_val, primal_derived, primal_ext), tangent_out

    return call_fn


# ── compiled pipeline ─────────────────────────────────────────────────────────

class CompiledGraph:
    """
    Static computation graph compiled from a root calculator.

    __call__(params_flat) is a pure function fully compatible with
    jax.jit, jax.vmap, and jax.grad.

    Parameters
    ----------
    root : Calculator
        Terminal node (e.g. a likelihood) whose output is returned.
    """

    def __init__(self, root: Calculator, ctx: '_CompileContext', output=None):
        self.root = root
        self.output = output
        self.nodes = ctx.node_order

        # Per-node dep lists split by type.
        self._node_var_deps = {}
        self._node_calc_deps = {}
        for node in self.nodes:
            deps = ctx.node_deps.get(id(node), [])
            self._node_var_deps[id(node)] = [d for d in deps if isinstance(d, Variable)]
            self._node_calc_deps[id(node)] = [d for d in deps if isinstance(d, Calculator)]

        # Collect unique Variable objects preserving DAG order, deduplicated by identity.
        # Two distinct Variable objects with the same name is an error — share the instance.
        self.params = VariableCollection()
        seen_ids = set()
        for node in self.nodes:
            for param in self._node_var_deps[id(node)]:
                if id(param) not in seen_ids:
                    if param.name in self.params:
                        raise ValueError(f'Variable {param.name!r} is owned by multiple nodes as distinct objects; pass the same Variable instance to both')
                    seen_ids.add(id(param))
                    self.params.set(param)

        self._derived_params = [p for p in self.params if p.derived is True]

        # Clear stale flags on all nodes seen during compilation.
        for node in self.nodes:
            node._updated = False
        for p in self.params:
            p._updated = False

        # Per-node cache state for all nodes (keyed by node id).
        self._node_states = {id(node): {'last_params': None, 'was_called': False, 'last_result': None, 'dep_result': None, 'call_result': None}
                             for node in self.nodes}

        # Build ExternalCalculator callables; record tree_flatten child counts.
        self._tree_own_aux = []
        self._fn_dep = []
        self._fn_call = []
        self._ext_n_children = {}
        for node in self.nodes:
            children, aux = node.tree_flatten()
            self._tree_own_aux.append(aux)
            if isinstance(node, ExternalCalculator):
                self._ext_n_children[id(node)] = len(children)
                node_state = self._node_states[id(node)]
                calc_deps = self._node_calc_deps[id(node)]
                dep_states = [self._node_states[id(dep)] for dep in calc_deps]
                call_return = ctx.call_returns[id(node)]
                fn_dep, fn_call = _make_external_fn(node, self._node_var_deps[id(node)], calc_deps, call_return, node_state, dep_states)
                self._fn_dep.append(fn_dep)
                self._fn_call.append(fn_call)
            else:
                self._fn_dep.append(None)
                self._fn_call.append(None)

        # Compute which params must be FD-differentiated (feed any ExternalCalculator
        # directly or transitively) vs which can use exact JAX auto-diff.
        downstream_of = defaultdict(list)
        for node in self.nodes:
            for dep in self._node_calc_deps[id(node)]:
                downstream_of[id(dep)].append(node)

        ext_reach = set()
        for node in reversed(self.nodes):
            if isinstance(node, ExternalCalculator):
                ext_reach.add(id(node))
            elif any(id(ds) in ext_reach for ds in downstream_of[id(node)]):
                ext_reach.add(id(node))

        fd_param_names = {p.name for node in self.nodes if id(node) in ext_reach for p in self._node_var_deps[id(node)]}
        self._fd_params = [p for p in self.params if p.name in fd_param_names]
        self._jax_params = [p for p in self.params if p.name not in fd_param_names]
        self._fd_names = [p.name for p in self._fd_params]
        self._jax_names = [p.name for p in self._jax_params]

        self._call_fn = _build_graph_call_fn(self)

    def __call__(self, params=None, **kwargs):
        """
        Pure function: dict of params → (return_val, derived_dict).
        Compatible with jax.jit, jax.vmap, jax.grad (dict form only for JAX transforms).
        Kwargs form (pipeline(omega_m=0.3, ...)) is a convenience for eager calls;
        missing params are filled from defaults.
        """
        stale = [n for n in self.nodes if n._updated] + [p for p in self.params if p._updated]
        if stale:
            names = ', '.join(getattr(n, 'name', type(n).__name__) for n in stale)
            raise RuntimeError(f"Pipeline is stale — {names} updated since compile(); call compile() again")
        if params is None:
            params = {p.name: p.value for p in self.params}
            params.update(kwargs)
        else:
            missing = {p.name: p.value for p in self.params if p.name not in params}
            if missing:
                params = {**params, **missing}
        fd_params_tuple = tuple(jnp.asarray(params[n]) for n in self._fd_names)
        jax_params_tuple = tuple(jnp.asarray(params[n]) for n in self._jax_names)
        return_val, derived_dict, _ = self._call_fn(fd_params_tuple, jax_params_tuple)
        return return_val, derived_dict


def _trace_node(node: Calculator, ctx: _CompileContext) -> None:
    """DFS helper for build_graph: run __post_init__ and scan __dict__ for deps."""
    ctx.traced.add(id(node))
    if id(node) not in ctx.post_init_called:
        args, kwargs = node._init
        node.__post_init__(*args, **kwargs)
    # Discover deps from public attributes set during __post_init__
    for key, val in node.__dict__.items():
        if key.startswith('_'):
            continue
        candidates = [val] if isinstance(val, Node) else [v for v in val if isinstance(v, Node)] if isinstance(val, (list, tuple)) else []
        for dep in candidates:
            deps = ctx.node_deps.setdefault(id(node), [])
            if id(dep) not in {id(d) for d in deps}:
                deps.append(dep)
            if id(dep) not in ctx.traced and isinstance(dep, Calculator):
                _trace_node(dep, ctx)
    ctx.node_order.append(node)


def build_graph(root: Calculator) -> _CompileContext:
    """Run __post_init__ on root and all reachable Calculators; return the compilation context.

    Discovers all node dependencies (Variable, Parameter, Calculator) by scanning public
    attributes set during __post_init__, without executing any __call__.
    Use params() or inspect ctx.node_deps / ctx.node_order to examine the graph.
    """
    outer_ctx = getattr(_compile_context, 'ctx', None)
    ctx = _CompileContext()
    _compile_context.ctx = ctx
    try:
        _trace_node(root, ctx)
    finally:
        _compile_context.ctx = outer_ctx
    return ctx


def params(node_or_graph) -> VariableCollection:
    """Return the Variable/Parameter collection for a Calculator or CompiledGraph.

    For a CompiledGraph, returns the already-collected params. For a Calculator,
    builds the graph via __post_init__ only (no __call__) and collects Variable deps.
    """
    if isinstance(node_or_graph, CompiledGraph):
        return node_or_graph.params
    ctx = build_graph(node_or_graph)
    result = VariableCollection()
    seen_ids = set()
    for node in ctx.node_order:
        for dep in ctx.node_deps.get(id(node), []):
            if isinstance(dep, Variable) and id(dep) not in seen_ids:
                if dep.name in result:
                    raise ValueError(f'Variable {dep.name!r} appears as distinct objects in multiple nodes; pass the same instance')
                seen_ids.add(id(dep))
                result.set(dep)
    return result


def compile(root: Calculator, output=None) -> CompiledGraph:
    """Trace root's dependency graph and return a CompiledGraph.

    Phase 1 (build_graph): runs __post_init__ on all reachable Calculators; discovers deps
    by scanning public attributes. Phase 2: runs __call__ on each node in topological order;
    raises if __call__ introduces a new Calculator not declared in __post_init__; prunes
    nodes not activated during __call__.
    """
    outer_ctx = getattr(_compile_context, 'ctx', None)
    ctx = build_graph(root)
    _compile_context.ctx = ctx
    try:
        ctx.phase = 'call'
        ctx.call_activated.add(id(root))
        for node in ctx.node_order:
            ctx.stack.append(node)
            ret = node.__call__()
            ctx.call_returns[id(node)] = ret
            ctx.stack.pop()
        # Prune Calculator nodes not accessed during __call__
        ctx.node_order = [n for n in ctx.node_order if id(n) in ctx.call_activated]
        for nid in list(ctx.node_deps.keys()):
            ctx.node_deps[nid] = [d for d in ctx.node_deps[nid] if not isinstance(d, Calculator) or id(d) in ctx.call_activated]
    finally:
        _compile_context.ctx = outer_ctx
    return CompiledGraph(root, ctx, output=output)
