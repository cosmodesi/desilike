"""Taylor polynomial emulator for desilike calculator graphs."""

import copy
import functools
import itertools
import math
import logging
import pickle

import jax
import numpy as np
import jax.numpy as jnp

from ..base import Calculator, CompiledGraph, compile, differentiate
from ..parameter import VariableCollection
from ..utils import register_type


def _iter_multi_indices(input_param_names, order_dict, total_order):
    """Yield all valid multi-indices ``(k_1, …, k_n)`` for the Taylor expansion.

    A multi-index is included when:
      - ``0 ≤ kᵢ ≤ order_dict[input_param_names[i]]`` for every *i*, and
      - ``Σkᵢ ≤ total_order``.

    The all-zero index (constant term) is always yielded first.
    """
    for mi in itertools.product(*(range(order_dict[n] + 1) for n in input_param_names)):
        if sum(mi) <= total_order:
            yield mi


@register_type
class TaylorEmulator:
    """
    Taylor polynomial emulator for a compiled calculator graph.

    Computes a total-degree Taylor expansion of the graph's outputs around a
    center point in the space of *input* (non-derived) parameters.  The
    resulting polynomial is pure JAX and can be ``jax.jit``-compiled or
    ``jax.vmap``-ed.

    Parameters
    ----------
    graph : CompiledGraph
        The compiled graph to emulate.
    order : int or dict[str | Variable, int]
        Maximum derivative order per *input* parameter.  A scalar applies to
        all input parameters; a dict gives per-parameter control (derived
        parameters are silently ignored).
    fd_acc : int or dict, optional
        FD accuracy override forwarded to :func:`~desilike.base.differentiate`.
    fd_eps : float or dict, optional
        FD step-size override forwarded to :func:`~desilike.base.differentiate`.

    Examples
    --------
    Build, fit and predict::

        emulator = TaylorEmulator(graph, order=3)
        emulator.fit()
        return_val, derived = emulator.predict({'omega_m': 0.31, 'sigma8': 0.82})

    Drop-in replacement in a new pipeline::

        calc_emu = emulator.to_calculator()
        g2 = compile(calc_emu)
        result = g2({'omega_m': 0.31, 'sigma8': 0.82})

    Save / load::

        emulator.write('emu.h5')
        emulator2 = TaylorEmulator.read('emu.h5')
    """

    _name = 'TaylorEmulator'

    logger = logging.getLogger('TaylorEmulator')

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, graph: CompiledGraph, order, fd_acc=None, fd_eps=None):
        self._graph = graph

        input_param_names = graph.params.select(varied=True).names()

        # Normalise *order* to a per-input-parameter dict.
        if isinstance(order, int):
            order_dict = {n: order for n in input_param_names}
        else:
            order_dict = {(k.name if hasattr(k, 'name') else str(k)): v
                          for k, v in order.items()}
            for n in input_param_names:
                order_dict.setdefault(n, 0)

        self._order = {n: order_dict.get(n, 0) for n in input_param_names}
        self._fd_acc = fd_acc
        self._fd_eps = fd_eps
        self._input_param_names = input_param_names    # expansion variables
        self._total_order = max(self._order.values()) if self._order else 0

        # Preserve the full VariableCollection (including derived Variables) so
        # to_calculator() reconstructs the complete parameter interface.
        self._params = graph.params

        # Populated by fit(); None until then.
        self._center = None
        self._powers = None              # (n_terms, n_input_params)  int32
        self._coeffs_children = None     # list (per tree_flatten child) of (n_terms, *child_shape)
        self._coeffs_derived = None      # {name: (n_terms, *shape)}
        self._coeffs_returned = None     # (n_terms, *rv_shape); only when return kind == 'value'
        self._derived_names = None       # output quantities tracked in derived_dict
        self._root_cls = type(graph.root)
        self._treedef = None
        self._n_children = None          # number of tree_flatten leaves
        # How the root's __call__ return value relates to its tree_flatten children:
        #   'none'  → returns None (outputs live entirely in attributes/children)
        #   'self'  → returns the node itself
        #   'child' → returns children[_return_index] (the common single-array case)
        #   'value' → returns an array not among the children (emulated separately)
        self._return_kind = None
        self._return_index = None

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, center=None):
        """Compute Taylor expansion coefficients around *center*.

        Parameters
        ----------
        center : dict[str, float | array], optional
            Values of the *input* parameters at the expansion point.  Defaults
            to the graph's compile-time parameter values.

        Returns
        -------
        self
        """
        graph = self._graph
        if graph is None:
            raise RuntimeError('The graph is not available (loaded from file?); cannot refit')

        self.logger.info('Varied parameters: %s', self._input_param_names)

        # Resolve expansion center (input params only; derived handled separately).
        p0 = {p.name: p._value for p in graph.params}
        if center is not None:
            p0.update(center)
        self._center = {n: np.asarray(p0[n]) for n in self._input_param_names}

        # Primal evaluation at center on the original graph — gives the return value
        # (rv0, used only to classify how it relates to the tree children) and derived dict.
        def _primal(g):
            fd_t = tuple(jnp.asarray(p0[n]) for n in g._fd_names)
            jax_t = tuple(jnp.asarray(p0[n]) for n in g._jax_names)
            input_saved = {p.name: p._value for p in g.params if g.params[p.name].varied}
            try:
                return g._call_fn(fd_t, jax_t)
            finally:
                for p in g.params:
                    if p.name in input_saved:
                        p._value = input_saved[p.name]

        rv0, derived0, _ = _primal(graph)
        self._derived_names = list(derived0.keys())

        # After an eager graph call the root's attributes are set; capture tree structure.
        children0, self._treedef = jax.tree_util.tree_flatten(graph.root)
        self._n_children = len(children0)

        # Classify the return value w.r.t. the tree_flatten children.  Many calculators
        # set their outputs as attributes and return None (or self): in that case the
        # meaningful, parameter-dependent state is the *children*, which we Taylor-expand.
        self._return_index = None
        if rv0 is None:
            self._return_kind = 'none'
        elif rv0 is graph.root:
            self._return_kind = 'self'
        else:
            idx = next((i for i, c in enumerate(children0) if c is rv0), None)
            if idx is not None:
                self._return_kind, self._return_index = 'child', idx
            else:
                self._return_kind = 'value'

        # Build a graph whose output is the tuple of tree_flatten children, so that
        # differentiate() yields the derivative of every child (not just the return value).
        root = graph.root
        child_graph = compile(root, output=lambda root=root: tuple(jnp.asarray(c) for c in jax.tree_util.tree_leaves(root)))
        children0_t, _, _ = _primal(child_graph)

        # Enumerate multi-indices once.
        order_dict = self._order
        input_param_names = self._input_param_names
        total_order = self._total_order
        multi_indices = list(_iter_multi_indices(input_param_names, order_dict, total_order))

        coeffs_children = [[] for _ in range(self._n_children)]
        coeffs_derived = {n: [] for n in self._derived_names}
        coeffs_returned = [] if self._return_kind == 'value' else None

        for mi in multi_indices:
            order_arg = {n: k for n, k in zip(input_param_names, mi) if k > 0}
            prefactor = 1.0 / math.prod(math.factorial(k) for k in mi)

            if order_arg:
                d_fn = differentiate(child_graph, order_arg, fd_acc=self._fd_acc, fd_eps=self._fd_eps)
                deriv_children, deriv_derived = d_fn(p0, return_derived=True)
                if self._return_kind == 'value':
                    deriv_rv = differentiate(graph, order_arg, fd_acc=self._fd_acc, fd_eps=self._fd_eps)(p0)
            else:
                deriv_children = children0_t
                deriv_derived = dict(derived0)
                deriv_rv = rv0

            for ci in range(self._n_children):
                coeffs_children[ci].append(np.asarray(deriv_children[ci]) * prefactor)
            for n in self._derived_names:
                v_zero = np.zeros_like(np.asarray(derived0.get(n, 0.0)))
                coeffs_derived[n].append(np.asarray(deriv_derived.get(n, v_zero)) * prefactor)
            if self._return_kind == 'value':
                coeffs_returned.append(np.asarray(deriv_rv) * prefactor)

        self._powers = np.array(multi_indices, dtype=np.int32)
        self._coeffs_children = [np.stack(coeffs_children[ci], axis=0) for ci in range(self._n_children)]
        self._coeffs_derived = {n: np.stack(coeffs_derived[n], axis=0)
                                 for n in self._derived_names}
        self._coeffs_returned = (np.stack(coeffs_returned, axis=0)
                                 if coeffs_returned is not None else None)
        self._jitted_predict_children = jax.jit(self._predict_children)
        return self

    # ── prediction ───────────────────────────────────────────────────────────

    def _monomials(self, params):
        """Return the per-term monomial values ``∏ᵢ (paramsᵢ − centerᵢ)^kᵢ``  (shape ``(n_terms,)``)."""
        center = self._center
        p_full = dict(center)
        p_full.update({n: params[n] for n in self._input_param_names if n in params})
        # diffs[i] = params[i] − center[i],  shape (n_input_params,)
        diffs = jnp.array([jnp.asarray(p_full[n]) - jnp.asarray(center[n])
                           for n in self._input_param_names])
        powers_j = jnp.asarray(self._powers)   # (n_terms, n_input_params)
        # Avoid NaN when diff=0 and k=0.
        return jnp.prod(jnp.where(powers_j > 0, diffs[None, :] ** powers_j, 1.0), axis=-1)

    def _predict_children(self, params):
        """Evaluate the Taylor polynomial; return ``(monomials, children_list, derived_dict)``."""
        if self._powers is None:
            raise RuntimeError('Call fit() before predict()')
        monomials = self._monomials(params)
        children = [jnp.tensordot(monomials, jnp.asarray(c), axes=([0], [0]))
                    for c in self._coeffs_children]
        derived = {n: jnp.tensordot(monomials, jnp.asarray(self._coeffs_derived[n]), axes=([0], [0]))
                   for n in self._derived_names}
        return monomials, children, derived

    def _return_value(self, monomials, children):
        """Reconstruct the emulated graph return value from the emulated children."""
        kind = self._return_kind
        if kind == 'none':
            return None
        if kind == 'child':
            return children[self._return_index]
        if kind == 'self':
            return jax.tree_util.tree_unflatten(self._treedef, children)
        # 'value': return value not among the children — emulated separately.
        return jnp.tensordot(monomials, jnp.asarray(self._coeffs_returned), axes=([0], [0]))

    def predict(self, params):
        """Evaluate the Taylor polynomial at *params*.

        Parameters
        ----------
        params : dict[str, float | array]
            Values of the *input* parameters.  Missing entries are filled from
            the expansion center.

        Returns
        -------
        return_val : jax array or None
            Emulated main return value of the graph (``None`` when the root's
            ``__call__`` returns ``None`` — its outputs live in the tree children).
        derived : dict[str, jax array]
            Emulated derived quantities (empty dict when none are tracked).
        """
        monomials, children, derived = self._jitted_predict_children(params)
        return self._return_value(monomials, children), derived

    # ── calculator factory ───────────────────────────────────────────────────

    @functools.cached_property
    def _emulated_cls(self):
        """Dynamically build a Calculator subclass backed by this emulator."""
        emulator = self
        root_cls = self._root_cls
        input_param_names = self._input_param_names
        derived_names = self._derived_names or []

        class TaylorEmulatedCalculator(root_cls):

            def __init__(self, *args, **kwargs):
                # Register the same Variable/Parameter objects as the original graph,
                # preserving the full interface (including derived Variables).  These
                # are set in __init__ so build_graph discovers them as dependencies.
                # Extra kwargs (e.g. k=, ells=) passed by a downstream __post_init__
                # via update() are accepted and ignored — the emulator is fixed at
                # training time.
                self.params = {param.name: param.clone() for param in emulator._params}

            def __post_init__(self, *args, **kwargs):
                # Skip root_cls.__post_init__ — emulator needs no heavy setup.
                # Data (table, kap, …) is populated by __call__ via tree_unflatten.
                pass

            def __call__(self):
                # Collect input parameter values and reconstruct the full child state.
                p = {n: self.params[n].value for n in input_param_names}
                monomials, children, derived = emulator._jitted_predict_children(p)

                # Reconstruct the root's attribute state via tree_unflatten so
                # downstream calculators can access e.g. self.theory.table or
                # underscore-prefixed config attrs like self.pt._A_full.
                # The proxy is a bare object.__new__ instance with no Node lifecycle
                # attributes, so copying its entire __dict__ is safe.
                proxy = jax.tree_util.tree_unflatten(emulator._treedef, children)
                self.__dict__.update(proxy.__dict__)

                # Update derived-Variable values *in-place* rather than replacing the
                # object, so that the pipeline's reference in _node_var_deps remains valid.
                # derived_names is always a subset of self.params' names (both ultimately
                # come from the same compiled graph.params at fit() time), so the lookup
                # is guaranteed to succeed.
                for name in derived_names:
                    self.params[name].value = derived[name]

                rv = emulator._return_value(monomials, children)
                # 'self'-returning roots: mirror that by returning this instance.
                return self if emulator._return_kind == 'self' else rv

            def tree_flatten(self):
                return root_cls.tree_flatten(self)

            @classmethod
            def tree_unflatten(cls, aux, children):
                obj = object.__new__(cls)
                proxy = root_cls.tree_unflatten(aux, children)
                obj.__dict__.update(proxy.__dict__)
                return obj

        TaylorEmulatedCalculator._is_external = False
        TaylorEmulatedCalculator.__name__ = root_cls.__name__
        TaylorEmulatedCalculator.__qualname__ = root_cls.__qualname__
        TaylorEmulatedCalculator.__module__ = root_cls.__module__
        return TaylorEmulatedCalculator

    def to_calculator(self):
        """Return a Calculator instance that is a drop-in replacement for the
        emulated graph's root calculator.

        The instance has the same parameters and ``tree_flatten`` interface as
        the original, but evaluates the fitted Taylor polynomial.

        Returns
        -------
        TaylorEmulatedCalculator instance
        """
        if self._powers is None:
            raise RuntimeError('Call fit() before to_calculator()')
        return self._emulated_cls()

    # ── serialisation ────────────────────────────────────────────────────────

    def __getstate__(self, to_file=False):
        """Return a serialisable state dict.

        Parameters
        ----------
        to_file : bool
            When ``True``, adds ``'attrs'`` with ``'__class__'`` so that
            :func:`~desilike.utils.read` can dispatch on load.
        """
        if self._powers is None:
            raise RuntimeError('Call fit() before serializing')

        state = {
            'attrs': {
                '__class__': self._name,
                'root_cls_module': self._root_cls.__module__,
                'root_cls_name': self._root_cls.__name__,
                'total_order': int(self._total_order),
                'n_children': int(self._n_children),
                'return_kind': self._return_kind,
                'return_index': int(self._return_index) if self._return_index is not None else -1,
            },
            'input_param_names': np.array(self._input_param_names, dtype='U'),
            'derived_names': np.array(self._derived_names if self._derived_names else [], dtype='U'),
            'order': {n: np.int32(v) for n, v in self._order.items()},
            'center': self._center,
            'powers': self._powers,
            # One stacked (n_terms, *child_shape) coefficient array per tree_flatten child.
            'coeffs_children': {str(i): np.asarray(c) for i, c in enumerate(self._coeffs_children)},
        }

        if self._coeffs_derived:
            state['coeffs_derived'] = dict(self._coeffs_derived)
        if self._coeffs_returned is not None:
            state['coeffs_returned'] = self._coeffs_returned

        # treedef: serialise with pickle → hex string stored in attrs.
        if self._treedef is not None:
            state['attrs']['treedef_hex'] = pickle.dumps(self._treedef).hex()

        # Preserve the full VariableCollection so to_calculator() works after loading.
        state['params'] = self._params.__getstate__(to_file=to_file)

        return state

    def __setstate__(self, state):
        """Populate from a state dict produced by :meth:`__getstate__`."""
        attrs = state.get('attrs', {})

        self._total_order = int(attrs.get('total_order', 0))
        self._input_param_names = [str(n) for n in state['input_param_names']]
        self._derived_names = ([str(n) for n in state['derived_names']]
                               if state['derived_names'].size else [])
        self._order = {n: int(state['order'][n]) for n in self._input_param_names}
        self._center = {n: state['center'][n] for n in self._input_param_names}
        self._powers = state['powers']
        self._coeffs_derived = dict(state.get('coeffs_derived', {}))
        self._coeffs_returned = state.get('coeffs_returned')
        self._fd_acc = None
        self._fd_eps = None
        self._graph = None   # not serialised

        # Reconstruct return-value classification and the per-child coefficient list.
        self._n_children = int(attrs.get('n_children', 0))
        self._return_kind = attrs.get('return_kind', 'child')
        ret_idx = int(attrs.get('return_index', -1))
        self._return_index = ret_idx if ret_idx >= 0 else None
        coeffs_children_state = state.get('coeffs_children', {})
        self._coeffs_children = [coeffs_children_state[str(i)] for i in range(self._n_children)]

        # Reconstruct the VariableCollection.
        vc = VariableCollection.__new__(VariableCollection)
        vc.__setstate__(state['params'])
        self._params = vc

        # Reconstruct the root Calculator class by module + name.
        import importlib
        mod = importlib.import_module(attrs['root_cls_module'])
        self._root_cls = getattr(mod, attrs['root_cls_name'])

        # Reconstruct treedef (None when not serialised).
        treedef_hex = attrs.get('treedef_hex')
        self._treedef = pickle.loads(bytes.fromhex(treedef_hex)) if treedef_hex else None

        self._jitted_predict_children = jax.jit(self._predict_children)

    # ── convenience I/O ──────────────────────────────────────────────────────

    def write(self, filename):
        """Write this emulator to *filename* (``.h5``, ``.hdf5``, or ``.txt``)."""
        from ..utils import write
        write(filename, self)

    @classmethod
    def read(cls, filename):
        """Load a :class:`TaylorEmulator` from *filename*."""
        from ..utils import read
        return read(filename)
