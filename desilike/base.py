"""
JAX-friendly calculator pipeline for desilike.

Two base classes:
- JAXCalculator: pure JAX ops, fully traceable by jit/vmap/grad.
- ExternalCalculator: arbitrary Python/numpy, wrapped via pure_callback +
  custom_vjp (finite-difference backward pass) so that jit/vmap/grad all work.

Lazy initialization:
- Calculator(*args, **kwargs) stores args without calling init().
- CompiledPipeline calls init() during graph construction, then scans
  instance attributes for JAXCalculator objects to discover dependencies.

call() interface:
- call(self) reads own params and dep outputs directly from self (e.g. self.A, self.cosmo.growth_factor).
- The pipeline sets param attributes on self before invoking call().
- call() sets named output attributes on self (e.g. self.growth_factor).
- call() may return any value (including self); that value
  is forwarded as the pipeline output if this node is the root. tree_flatten()
  defines which outputs are passed to downstream nodes.

tree_flatten / tree_unflatten:
- Each calculator must define tree_flatten(self) -> (children, aux) and
  tree_unflatten(cls, aux, children) -> instance. children are the output
  arrays produced by call(). The framework uses these to pass outputs between
  calculators and to define the ExternalCalculator dep-passing interface.

Pipeline:
- CompiledPipeline: builds a static DAG once, exposes a pure
  __call__(params) compatible with jax.jit / jax.vmap / jax.grad.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from .parameter import Parameter, ParameterCollection




# ── base classes ──────────────────────────────────────────────────────────────

class JAXCalculator:
    """
    Base class for calculators implemented with JAX ops.

    Subclasses define:
      init(*args, **kwargs): wire dependencies and parameters by setting
        JAXCalculator instances (deps) and Parameter instances (params) as attributes.
        Both are auto-detected from these attributes after init().
      call(self): read params via self.param.value and dep outputs via self.dep.attr;
        compute and store output attributes; return the output value.
      tree_flatten(self) -> (children, aux): children = list of output arrays,
        aux = static data needed by tree_unflatten.
      tree_unflatten(cls, aux, children) -> instance: reconstruct an instance
        carrying only the output attrs (no dep refs, no init args).

    Instantiation is lazy: __init__ stores args/kwargs and does NOT call init().
    CompiledPipeline calls init() during graph construction.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(
            cls,
            lambda node: node.tree_flatten(),
            cls.tree_unflatten,
        )

    def __init__(self, *args, **kwargs):
        self._init = (args, kwargs)

    def update(self, *args, **kwargs):
        """Update stored init arguments; new kwargs override old ones."""
        old_args, old_kwargs = self._init
        self._init = (args if args else old_args, {**old_kwargs, **kwargs})

    def init(self, *args, **kwargs):
        pass

    def call(self):
        raise NotImplementedError

    def tree_flatten(self):
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux, children):
        raise NotImplementedError


class ExternalCalculator(JAXCalculator):
    """
    Base class for non-JAX calculators.
    call() may use arbitrary Python/numpy, accessing dep outputs as self.dep.attr
    (concrete numpy arrays when the callback executes).

    Wrapped via jax.pure_callback (for jit/vmap) + jax.custom_jvp
    (finite-difference JVP for grad/jacfwd/hessian).

    Per-parameter FD step and accuracy are taken from param.fd_eps and param.fd_acc.
    param.fd_eps falls back through: explicit → param.proposal → param.ref.std() → 1e-5.
    param.fd_acc defaults to 2; set to 4, 6, ... for higher-accuracy stencils.
    """


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


# ── dependency resolution ─────────────────────────────────────────────────────

def _resolve_deps(node: JAXCalculator) -> list:
    """Return JAXCalculator attributes set on node by init(), in definition order."""
    return [v for v in vars(node).values() if isinstance(v, JAXCalculator)]


def _resolve_params(node: JAXCalculator) -> list:
    """Return Parameter attributes set on node by init(), in definition order."""
    return [v for v in vars(node).values() if isinstance(v, Parameter)]


# ── graph construction (init + topological sort) ──────────────────────────────

def _init_and_sort(root: JAXCalculator) -> list:
    """
    Recursively call init() on each node and return them in topological order
    (leaves first, root last). Dependencies are discovered by scanning instance
    attributes for JAXCalculator objects after init() runs.
    """
    order, visited = [], set()

    def visit(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        args, kwargs = node._init
        node.init(*args, **kwargs)
        for dep in _resolve_deps(node):
            visit(dep)
        order.append(node)

    visit(root)
    return order


# ── external function factory ─────────────────────────────────────────────────

def _make_external_fn(node: ExternalCalculator, call_return, node_state: dict, dep_states: list):
    """
    Return (fn_dep, fn_call) — two JAX-compatible callables wrapping node.call().

    fn_dep(own_params_flat, dep_attr_flat) -> tree_flatten children
        Used for intermediate nodes: result drives tree_unflatten for dep passing.

    fn_call(own_params_flat, dep_attr_flat) -> call() return value
        Used when this node is the pipeline root.

    Both support jit, vmap, and grad (finite differences via _fd_stencil).
    node.call() is skipped (cached result reused) when own params and all
    dep outputs are unchanged since the last invocation.
    node_state: mutable dict {'last_params', 'was_called', 'dep_result', 'call_result'}.
    dep_states: list of node_state dicts for all dependencies (JAX and External).
    """
    params_list = _resolve_params(node)
    deps = _resolve_deps(node)
    dep_schema = []
    for dep in deps:
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
        """Call node.call() only if own params or any dep has changed."""
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
            call_result = node.call()
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
        @jax.custom_jvp
        def fn(own_params_tuple, dep_attr_flat):
            return jax.pure_callback(callback, result_sdt,
                                     own_params_tuple, *dep_attr_flat,
                                     vmap_method='sequential')

        @fn.defjvp
        def fn_jvp(primals, tangents):
            own_params_tuple, dep_attr_flat = primals
            own_tangents, dep_tangents = tangents

            # Calling fn (not pure_callback) with primal-only inputs: since the
            # primal values extracted from `primals` are at a lower trace level
            # than the current JVP trace, JAX does not intercept them with fn_jvp
            # recursively — they fall through to pure_callback.
            primal_out = fn(own_params_tuple, dep_attr_flat)
            tangent_out = jax.tree_util.tree_map(jnp.zeros_like, primal_out)

            for i, (param_arr, param) in enumerate(zip(own_params_tuple, params_list)):
                v_i = own_tangents[i]
                eps = param.fd_eps if param.fd_eps is not None else 1e-5
                offsets, coeffs = _fd_stencil(1, param.fd_acc)
                for idx in np.ndindex(param_arr.shape):
                    v_ij = v_i[idx] if param_arr.ndim > 0 else v_i
                    df = jax.tree_util.tree_map(jnp.zeros_like, primal_out)
                    for off, coeff in zip(offsets, coeffs):
                        shifted = list(own_params_tuple)
                        shifted[i] = param_arr.at[idx].add(off * eps)
                        fi = fn(tuple(shifted), dep_attr_flat)
                        df = jax.tree_util.tree_map(lambda d, f, c=coeff: d + c * f, df, fi)
                    tangent_out = jax.tree_util.tree_map(
                        lambda t, d, v=v_ij, e=eps: t + v * d / e, tangent_out, df)

            dep_offsets, dep_coeffs = _fd_stencil(1, 2)
            dep_eps = 1e-5
            for j, (dep_val, dep_t) in enumerate(zip(dep_attr_flat, dep_tangents)):
                for idx in np.ndindex(dep_val.shape):
                    v_ij = dep_t[idx] if dep_val.ndim > 0 else dep_t
                    df = jax.tree_util.tree_map(jnp.zeros_like, primal_out)
                    for off, coeff in zip(dep_offsets, dep_coeffs):
                        shifted_dep = list(dep_attr_flat)
                        shifted_dep[j] = dep_val.at[idx].add(off * dep_eps)
                        fi = fn(own_params_tuple, tuple(shifted_dep))
                        df = jax.tree_util.tree_map(lambda d, f, c=coeff: d + c * f, df, fi)
                    tangent_out = jax.tree_util.tree_map(
                        lambda t, d, v=v_ij, e=dep_eps: t + v * d / e, tangent_out, df)

            return primal_out, tangent_out

        return fn

    return _make_fn(_inject_and_call_dep, dep_sdt), _make_fn(_inject_and_call_call, call_result_sdt)


# ── compiled pipeline ─────────────────────────────────────────────────────────

class CompiledPipeline:
    """
    Static computation graph compiled from a root calculator.

    __call__(params_flat) is a pure function fully compatible with
    jax.jit, jax.vmap, and jax.grad.

    Parameters
    ----------
    root : JAXCalculator
        Terminal node (e.g. a likelihood) whose output is returned.
    """

    def __init__(self, root: JAXCalculator, output=None):
        self.root = root
        self.output = output
        self.nodes = _init_and_sort(root)

        # Collect unique Parameter objects preserving DAG order, deduplicated by identity.
        # Two distinct Parameter objects with the same name is an error — share the instance.
        self.params = ParameterCollection()
        seen_ids = set()
        for node in self.nodes:
            for param in _resolve_params(node):
                if id(param) not in seen_ids:
                    if param.name in self.params:
                        raise ValueError(f'Parameter {param.name!r} is owned by multiple nodes as distinct objects; pass the same Parameter instance to both')
                    seen_ids.add(id(param))
                    self.params.set(param)

        # Per-node cache state for all nodes (keyed by node id).
        self._node_states = {id(node): {'last_params': None, 'was_called': False, 'last_result': None, 'dep_result': None, 'call_result': None}
                             for node in self.nodes}

        # Dry run: call each node with default params, capture tree_flatten aux,
        # and build ExternalCalculator callables.
        self._tree_own_aux = []
        self._fn_dep = []
        self._fn_call = []
        for node in self.nodes:
            call_return = node.call()
            _, aux = node.tree_flatten()
            self._tree_own_aux.append(aux)
            if isinstance(node, ExternalCalculator):
                node_state = self._node_states[id(node)]
                dep_states = [self._node_states[id(dep)] for dep in _resolve_deps(node)]
                fn_dep, fn_call = _make_external_fn(node, call_return, node_state, dep_states)
                self._fn_dep.append(fn_dep)
                self._fn_call.append(fn_call)
            else:
                self._fn_dep.append(None)
                self._fn_call.append(None)

    def __call__(self, params=None, **kwargs):
        """
        Pure function: dict of params → scalar (or array).
        Compatible with jax.jit, jax.vmap, jax.grad (dict form only for JAX transforms).
        Kwargs form (pipeline(omega_m=0.3, ...)) is a convenience for eager calls;
        missing params are filled from defaults.
        """
        if params is None:
            params = {p.name: p.value for p in self.params}
            params.update(kwargs)
        is_tracing = any(isinstance(v, jax.core.Tracer) for v in params.values())

        if is_tracing:
            # Snapshot concrete state before tracing. JAX mutates param._value and node
            # output attrs to abstract tracers during the trace; restoring afterwards keeps
            # everything concrete when __call__ returns, preventing UnexpectedTracerError.
            saved_param_values = {p.name: p._value for p in self.params}
            saved_node_dicts = {id(n): dict(n.__dict__) for n in self.nodes}

        result = None
        for i, node in enumerate(self.nodes):
            if isinstance(node, ExternalCalculator):
                own_params_tuple = tuple(jnp.asarray(params[p.name]) for p in _resolve_params(node))
                dep_attr_flat = tuple(
                    v for dep in _resolve_deps(node)
                    for v in dep.tree_flatten()[0]
                )
                if node is self.root and self.output is None:
                    result = self._fn_call[i](own_params_tuple, dep_attr_flat)
                else:
                    raw = self._fn_dep[i](own_params_tuple, dep_attr_flat)
                    proxy = node.__class__.tree_unflatten(self._tree_own_aux[i], list(raw))
                    node.__dict__.update(proxy.__dict__)
            else:
                node_params = _resolve_params(node)
                node_state = self._node_states[id(node)]
                if is_tracing:
                    for param in node_params:
                        param.value = params[param.name]
                    result = node.call()
                    node_state['was_called'] = True
                else:
                    own_params_np = np.concatenate([np.ravel(np.asarray(params[p.name])) for p in node_params]) if node_params else np.array([])
                    dep_states = [self._node_states[id(dep)] for dep in _resolve_deps(node)]
                    dep_was_called = any(s['was_called'] for s in dep_states)
                    params_changed = node_state['last_params'] is None or not np.array_equal(own_params_np, node_state['last_params'])
                    if dep_was_called or params_changed:
                        for param in node_params:
                            param.value = params[param.name]
                        result = node.call()
                        node_state['last_params'] = own_params_np
                        node_state['last_result'] = result
                        node_state['was_called'] = True
                    else:
                        result = node_state['last_result']
                        node_state['was_called'] = False

        return_val = self.output() if self.output is not None else result

        if is_tracing:
            # Restore: undo all tracing-time side effects so node attrs and param values
            # remain concrete after __call__ returns. return_val already holds the traced
            # computation and is unaffected by this restoration.
            for node in self.nodes:
                node.__dict__.clear()
                node.__dict__.update(saved_node_dicts[id(node)])
            for p in self.params:
                p._value = saved_param_values[p.name]

        return return_val


def compile(root: JAXCalculator, output=None) -> CompiledPipeline:
    """Build and return a CompiledPipeline from root."""
    return CompiledPipeline(root, output=output)


