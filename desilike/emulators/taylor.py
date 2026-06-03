"""Taylor polynomial emulator for desilike calculator graphs."""

import copy
import functools
import itertools
import math
import pickle

import numpy as np
import jax.numpy as jnp

from ..base import Calculator, CompiledGraph, differentiate
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

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, graph: CompiledGraph, order, fd_acc=None, fd_eps=None):
        self._graph = graph

        # Split params into inputs (non-derived) and outputs (derived Variables).
        all_param_names = list(graph.params.names())
        input_param_names = [n for n in all_param_names if not graph.params[n].derived]

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
        self._params_vc = graph.params

        # Populated by fit(); None until then.
        self._center = None
        self._powers = None              # (n_terms, n_input_params)  int32
        self._coeffs_returned = None     # (n_terms, *rv_shape)
        self._coeffs_derived = None      # {name: (n_terms, *shape)}
        self._derived_names = None       # output quantities tracked in derived_dict
        self._root_cls = type(graph.root)
        self._tree_aux = None
        self._children_center = None     # primal tree_flatten children

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

        # Resolve expansion center (input params only; derived handled separately).
        p0 = {p.name: p._value for p in graph.params}
        if center is not None:
            p0.update(center)
        self._center = {n: np.asarray(p0[n]) for n in self._input_param_names}

        # Primal evaluation at center — call _call_fn directly to get derived_dict.
        fd_t = tuple(jnp.asarray(p0[n]) for n in graph._fd_names)
        jax_t = tuple(jnp.asarray(p0[n]) for n in graph._jax_names)
        input_saved = {p.name: p._value for p in graph.params if not graph.params[p.name].derived}
        try:
            rv0, derived0, _ = graph._call_fn(fd_t, jax_t)
        finally:
            for p in graph.params:
                if p.name in input_saved:
                    p._value = input_saved[p.name]
        self._derived_names = list(derived0.keys())

        # After an eager graph call the root's attrs are set; capture tree structure.
        children0, aux0 = graph.root.tree_flatten()
        self._tree_aux = aux0
        self._children_center = [np.asarray(c) for c in children0]

        # Enumerate multi-indices once.
        order_dict = self._order
        input_param_names = self._input_param_names
        total_order = self._total_order
        multi_indices = list(_iter_multi_indices(input_param_names, order_dict, total_order))

        coeffs_returned = []
        coeffs_derived = {n: [] for n in self._derived_names}

        for mi in multi_indices:
            order_arg = {n: k for n, k in zip(input_param_names, mi) if k > 0}
            prefactor = 1.0 / math.prod(math.factorial(k) for k in mi)

            if order_arg:
                d_fn = differentiate(
                    graph, order_arg,
                    fd_acc=self._fd_acc, fd_eps=self._fd_eps,
                )
                deriv_rv, deriv_derived = d_fn(p0, return_derived=True)
            else:
                deriv_rv = rv0
                deriv_derived = dict(derived0)

            coeffs_returned.append(np.asarray(deriv_rv) * prefactor)
            for n in self._derived_names:
                v_zero = np.zeros_like(np.asarray(derived0.get(n, 0.0)))
                coeffs_derived[n].append(
                    np.asarray(deriv_derived.get(n, v_zero)) * prefactor
                )

        self._powers = np.array(multi_indices, dtype=np.int32)
        self._coeffs_returned = np.stack(coeffs_returned, axis=0)
        self._coeffs_derived = {n: np.stack(coeffs_derived[n], axis=0)
                                 for n in self._derived_names}
        return self

    # ── prediction ───────────────────────────────────────────────────────────

    def predict(self, params):
        """Evaluate the Taylor polynomial at *params*.

        Parameters
        ----------
        params : dict[str, float | array]
            Values of the *input* parameters.  Missing entries are filled from
            the expansion center.

        Returns
        -------
        return_val : jax array
            Emulated main return value of the graph.
        derived : dict[str, jax array]
            Emulated derived quantities (empty dict when none are tracked).
        """
        if self._powers is None:
            raise RuntimeError('Call fit() before predict()')

        center = self._center
        p_full = dict(center)
        p_full.update({n: params[n] for n in self._input_param_names if n in params})

        # diffs[i] = params[i] − center[i],  shape (n_input_params,)
        diffs = jnp.array([jnp.asarray(p_full[n]) - jnp.asarray(center[n])
                           for n in self._input_param_names])

        powers_j = jnp.asarray(self._powers)   # (n_terms, n_input_params)

        # Monomial values: ∏ᵢ diffsᵢ^kᵢ — avoid NaN when diff=0 and k=0.
        monomials = jnp.prod(
            jnp.where(powers_j > 0, diffs[None, :] ** powers_j, 1.0),
            axis=-1,
        )  # (n_terms,)

        return_val = jnp.tensordot(monomials, jnp.asarray(self._coeffs_returned), axes=([0], [0]))
        derived = {n: jnp.tensordot(monomials, jnp.asarray(self._coeffs_derived[n]), axes=([0], [0]))
                   for n in self._derived_names}

        return return_val, derived

    # ── calculator factory ───────────────────────────────────────────────────

    @functools.cached_property
    def _emulated_cls(self):
        """Dynamically build a Calculator subclass backed by this emulator."""
        emulator = self
        root_cls = self._root_cls
        tree_aux = self._tree_aux
        # The extra children (beyond children[0]) are constants taken from the primal.
        extra_children = [jnp.asarray(c) for c in (self._children_center or [])[1:]]
        input_param_names = self._input_param_names
        derived_names = self._derived_names or []

        class TaylorEmulatedCalculator(Calculator):

            def __post_init__(self, **kwargs):
                # Register the same Variable/Parameter objects as the original
                # graph, preserving the full interface (including derived Variables).
                # Extra kwargs (e.g. k=, ells=) passed by a downstream __post_init__
                # via update() are accepted and ignored — the emulator is fixed at
                # training time.
                for param in emulator._params_vc:
                    setattr(self, param.name, copy.copy(param))

            def __call__(self):
                # Collect input parameter values.
                p = {n: getattr(self, n).value for n in input_param_names}
                rv, derived = emulator.predict(p)

                # Reconstruct the root's attribute state via tree_unflatten so
                # downstream calculators can access e.g. self.theory.power.
                children = [rv] + extra_children
                proxy = root_cls.tree_unflatten(tree_aux, children)
                for k, v in proxy.__dict__.items():
                    if not k.startswith('_'):
                        setattr(self, k, v)

                # For derived-Variable attributes, update the Variable's value
                # *in-place* rather than replacing the object, so that the
                # pipeline's reference in _node_var_deps remains valid.
                for n in derived_names:
                    attr = getattr(self, n, None)
                    if attr is not None and hasattr(attr, '_value'):
                        attr.value = derived[n]
                    else:
                        setattr(self, n, derived[n])

                return rv

            def tree_flatten(self):
                return root_cls.tree_flatten(self)

            @classmethod
            def tree_unflatten(cls, aux, children):
                obj = object.__new__(cls)
                proxy = root_cls.tree_unflatten(aux, children)
                obj.__dict__.update(proxy.__dict__)
                return obj

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
            },
            'input_param_names': np.array(self._input_param_names, dtype='U'),
            'derived_names': np.array(self._derived_names if self._derived_names else [], dtype='U'),
            'order': {n: np.int32(v) for n, v in self._order.items()},
            'center': self._center,
            'powers': self._powers,
            'coeffs_returned': self._coeffs_returned,
        }

        if self._coeffs_derived:
            state['coeffs_derived'] = dict(self._coeffs_derived)

        # tree_aux: serialise with pickle → hex string stored in attrs.
        if self._tree_aux is not None:
            state['attrs']['tree_aux_hex'] = pickle.dumps(self._tree_aux).hex()

        # Extra tree_flatten children beyond children[0] (stored as constants).
        if self._children_center and len(self._children_center) > 1:
            state['children_extra'] = {
                str(i): np.asarray(c) for i, c in enumerate(self._children_center[1:])
            }

        # Preserve the full VariableCollection so to_calculator() works after loading.
        state['params'] = self._params_vc.__getstate__(to_file=to_file)

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
        self._coeffs_returned = state['coeffs_returned']
        self._coeffs_derived = dict(state.get('coeffs_derived', {}))
        self._fd_acc = None
        self._fd_eps = None
        self._graph = None   # not serialised

        # Reconstruct the VariableCollection.
        vc = VariableCollection.__new__(VariableCollection)
        vc.__setstate__(state['params'])
        self._params_vc = vc

        # Reconstruct the root Calculator class by module + name.
        import importlib
        mod = importlib.import_module(attrs['root_cls_module'])
        self._root_cls = getattr(mod, attrs['root_cls_name'])

        # Reconstruct tree_aux (None when not serialised).
        tree_aux_hex = attrs.get('tree_aux_hex')
        self._tree_aux = pickle.loads(bytes.fromhex(tree_aux_hex)) if tree_aux_hex else None

        # Reconstruct the primal tree_flatten children list.
        # Only the *extra* children (beyond children[0]) are stored.
        children_extra_state = state.get('children_extra', {})
        n_extra = len(children_extra_state)
        extra = [children_extra_state[str(i)] for i in range(n_extra)]
        # Prepend None as placeholder for children[0] (the return value).
        self._children_center = [None] + extra

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
