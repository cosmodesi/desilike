"""Shared helper for multitracer theory construction.

Tracer theories accept a ``tracers`` constructor argument and namespace their own
bias parameters accordingly via :func:`propose_params_multitracer` and
:func:`assign_params`:

- ``None``: a single auto-spectrum with unnamespaced parameters.
- ``'LRG'`` (str): an auto-spectrum with parameters namespaced ``LRG.b1`` etc.
- ``('LRG', 'QSO')`` (tuple): the cross-spectrum.  Deterministic bias parameters
  become per-tracer 2-tuples ``(LRG.b1, QSO.b1)``; *stochastic* parameters get the
  ``LRGxQSO`` namespace; *shared* (e.g. cosmological) parameters stay unnamespaced.
  Only theories whose ``__call__`` handles tuple-valued parameters (``cross=True``)
  support this; others raise.

For a multitracer analysis, build one instance per spectrum and unify shared
parameters across instances with :func:`desilike.base.share_params`.
"""


def propose_params_multitracer(auto_params, tracers, stochastic=(), shared=(), cross=False):
    """Build a :class:`~desilike.parameter.VariableCollection` from *auto_params* with tracer namespacing applied.

    Same namespacing logic as :func:`apply_tracers`, but operates on a plain list of
    Parameters and returns a VariableCollection rather than mutating an instance.

    Parameters
    ----------
    auto_params : list of Parameter
        Unnamespaced default parameters.
    tracers : str, (str, str), or None
        Tracer spec (see module docstring).
    stochastic : sequence of str
        Parameter basenames treated as stochastic: in the cross case they take the
        ``XxY`` namespace (a single parameter) rather than becoming a 2-tuple.
    shared : sequence of str
        Parameter basenames that are never namespaced (e.g. ``fnl_loc``).
    cross : bool
        Whether cross-correlations are supported.  When ``False`` a tuple *tracers* raises.

    Returns
    -------
    VariableCollection
    """
    from ...parameter import VariableCollection
    vc = VariableCollection()
    shared_set, stochastic_set = set(shared), set(stochastic)

    if tracers is None:
        for param in auto_params:
            vc.set(param.clone())
        return vc

    if not isinstance(tracers, str) and len(tracers) == 1:
        tracers = tracers[0]

    if isinstance(tracers, str):
        for param in auto_params:
            if param.basename in shared_set:
                vc.set(param.clone())
            else:
                vc.set(param.clone(namespace=tracers))
        return vc

    if not cross:
        raise NotImplementedError(f'cross-correlations require cross=True (got tracers={tracers!r})')
    name_X, name_Y = tracers
    cross_ns = f'{name_X}x{name_Y}'
    for param in auto_params:
        basename = param.basename
        if basename in shared_set:
            vc.set(param.clone())
        elif basename in stochastic_set:
            vc.set(param.clone(namespace=cross_ns))
        else:
            vc.set(param.clone(namespace=name_X))
            vc.set(param.clone(namespace=name_Y))
    return vc


def assign_params(inst, vc, tracers, mapping=None):
    """Set Parameter attributes on *inst* from *vc*.

    Groups the VC entries by basename and sets the corresponding instance attributes:

    - One entry per basename (no tracers, single tracer, stochastic/shared cross): scalar.
    - Two entries per basename (deterministic cross): 2-tuple ordered ``(X, Y)``.
    - Non-identifier basenames (e.g. ``al0_-3``): appended to ``inst.bb_params`` in
      VC order, preserving ``zip(inst._bb_ell_pow, inst.bb_params)`` in ``__call__``.

    Parameters
    ----------
    inst : Calculator
        Instance to update in place.
    vc : VariableCollection
        Typically built by the class's :meth:`propose_params` then merged with
        caller-supplied overrides via ``vc = vc + VariableCollection(params)``.
    tracers : str, (str, str), or None
        Used to resolve (X, Y) ordering for deterministic cross parameters.
    mapping : callable or None, default=None
        Optional ``basename -> attribute_name`` transform applied before ``setattr``.
        Use this to give parameters a different attribute name than their basename —
        e.g. ``lambda name: name[:-1]`` strips a trailing ``p`` so that physical-basis
        params (``b1p``, ``b2p``, …) are stored as ``self.b1``, ``self.b2``, etc.
    """
    from collections import defaultdict
    by_basename = defaultdict(list)
    for p in vc:
        by_basename[p.basename].append(p)

    list_params = []
    for bn, params_group in by_basename.items():
        if len(params_group) == 1:
            value = params_group[0]
        else:
            # Deterministic cross: two params — order as (X, Y).
            name_X, name_Y = tracers
            value = tuple(sorted(params_group, key=lambda p: 0 if name_X in p.namespace else 1))
        attr_name = mapping(bn) if mapping is not None else bn
        if attr_name.isidentifier():
            setattr(inst, attr_name, value)
        else:
            list_params.append(value)

    if list_params:
        inst.bb_params = list_params
