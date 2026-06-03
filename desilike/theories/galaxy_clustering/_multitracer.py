"""Shared helper for multitracer theory construction.

Tracer theories accept a ``tracers`` constructor argument and namespace their own
bias parameters accordingly via :func:`apply_tracers`:

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


def apply_tracers(inst, tracers, stochastic=(), shared=(), cross=False):
    """Namespace *inst*'s own bias parameters in place for a tracer spec.

    Parameters
    ----------
    inst : Calculator
        Instance whose own ``Parameter`` attributes (and lists of Parameters) are
        rewritten.  Sub-calculator dependencies are left untouched.
    tracers : str, (str, str), or None
        Tracer spec (see module docstring).
    stochastic : sequence of str
        Basenames of stochastic parameters: in the cross case these take the
        ``XxY`` namespace (a single parameter) rather than becoming a 2-tuple.
    shared : sequence of str
        Basenames of parameters that are *never* namespaced (e.g. cosmological
        parameters shared across all tracers, such as ``fnl_loc``).
    cross : bool
        Whether the calculator's ``__call__`` handles tuple-valued (cross) bias
        parameters.  When ``False`` a tuple *tracers* raises.

    Returns
    -------
    inst (modified in place).
    """
    if tracers is None:
        return inst
    from ...parameter import Parameter
    shared, stochastic = set(shared), set(stochastic)

    def _param_attrs():
        return [(key, val) for key, val in list(inst.__dict__.items()) if isinstance(val, Parameter)]

    def _list_attrs():
        return [(key, val) for key, val in list(inst.__dict__.items())
                if isinstance(val, list) and val and all(isinstance(v, Parameter) for v in val)]

    if isinstance(tracers, str):
        for key, val in _param_attrs():
            if val.basename not in shared:
                setattr(inst, key, val.clone(namespace=tracers))
        for key, val in _list_attrs():
            setattr(inst, key, [v.clone(namespace=tracers) if v.basename not in shared else v for v in val])
        return inst

    # cross spectrum: tracers is a (X, Y) pair
    if not cross:
        raise NotImplementedError(f'{type(inst).__name__} does not support cross-correlations '
                                  f'(got {tracers!r}); pass a single tracer name.')
    name_X, name_Y = tracers
    cross_ns = f'{name_X}x{name_Y}'
    for key, val in _param_attrs():
        bn = val.basename
        if bn in shared:
            continue
        if bn in stochastic:
            setattr(inst, key, val.clone(namespace=cross_ns))
        else:  # deterministic: one parameter per tracer
            setattr(inst, key, (val.clone(namespace=name_X), val.clone(namespace=name_Y)))
    for key, val in _list_attrs():
        if all(v.basename in shared for v in val):
            continue
        setattr(inst, key, [(v.clone(namespace=name_X), v.clone(namespace=name_Y)) for v in val])
    return inst
