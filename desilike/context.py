"""Local, reversible settings on built graphs: parameter values, priors, refs, and how constraints
enter the posterior -- one context manager, :func:`override`, undone on exit (also on error), with no
new :func:`~desilike.base.build`::

    with desilike.override(posterior, value={'h': 0.68}, prior={'h': {'limits': [0.6, 0.8]}},
                           ref={'b1': {'dist': 'norm', 'loc': 2., 'scale': 0.1}},
                           constraints={'mode': 'soft', 'scale': {'LRG1.training_range': 0.01}}):
        profiler = Profiler(posterior, ...)
        profiler.maximize()

Settings are read when a graph is traced.  A graph's own jitted function is cached per settings
version, so a call under new settings retraces rather than reusing the old ones; a kernel
(profiler, sampler) captures the constraint settings active when it is constructed and keeps them.
"""

from __future__ import annotations

import contextlib
import threading

import numpy as np
import jax
import jax.numpy as jnp

_local = threading.local()
_MODES = ('hard', 'soft', None)


def _stack():
    if not hasattr(_local, 'stack'):
        _local.stack = []
    return _local.stack


def version():
    """Counter bumped by every :func:`constraints` / :func:`override` entry and exit; keys jit caches."""
    return getattr(_local, 'version', 0)


def _bump():
    _local.version = version() + 1


def _normalize_constraints(settings):
    """``'hard'`` / ``'soft'`` / ``None`` / dict → ``{'mode', 'scale', 'grad'}``."""
    if isinstance(settings, dict):
        settings = dict(settings)
    else:
        settings = {'mode': settings}
    unknown = set(settings) - {'mode', 'scale', 'grad'}
    if unknown:
        raise ValueError(f'unknown constraint settings {sorted(unknown)}; expected mode, scale, grad')
    settings.setdefault('mode', 'hard')
    settings.setdefault('scale', None)
    settings.setdefault('grad', None)
    if settings['mode'] not in _MODES:
        raise ValueError(f"constraint mode must be one of {_MODES}, not {settings['mode']!r}")
    if settings['grad'] not in (None, 'soft'):
        raise ValueError(f"constraint grad must be None or 'soft', not {settings['grad']!r}")
    return settings


def current_constraints():
    """The innermost active constraint settings, or ``None`` outside any context."""
    stack = _stack()
    return stack[-1] if stack else None


def resolve_constraints(default):
    """The active settings if any, else *default* (a mode or settings dict): what a kernel captures at construction."""
    current = current_constraints()
    return dict(current) if current is not None else _normalize_constraints(default)


@contextlib.contextmanager
def use_constraints(settings, bump=False):
    """Make *settings* active.  Used inside traced kernel objectives (``bump=False``: no jit-cache churn)."""
    settings = _normalize_constraints(settings)
    _stack().append(settings)
    if bump:
        _bump()
    try:
        yield settings
    finally:
        _stack().pop()
        if bump:
            _bump()


def constraint_logpdf(constraints, values, settings=None):
    """Log-density term of *constraints* (list of Constraint) at *values* (``{name: value}``).

    *settings* default to the active ones, or ``'hard'`` outside any context.
    """
    if not constraints:
        return jnp.zeros(())
    settings = _normalize_constraints(settings if settings is not None else (current_constraints() or 'hard'))
    mode, scale_setting = settings['mode'], settings['scale']
    if mode is None:
        return jnp.zeros(())
    violations, scales = [], []
    for constraint in constraints:
        value = jnp.maximum(jnp.ravel(jnp.asarray(values[constraint.name])), 0.)
        if isinstance(scale_setting, dict):
            scale = scale_setting.get(constraint.name, constraint.scale)
        elif scale_setting is not None:
            scale = scale_setting
        else:
            scale = constraint.scale
        violations.append(value)
        scales.append(np.full(value.shape, float(scale)))
    violations = jnp.concatenate(violations)
    scales = jnp.asarray(np.concatenate(scales))

    def soft(violations):
        return -0.5 * jnp.sum((violations / scales) ** 2)

    def hard(violations):
        return jnp.where(jnp.any(violations > 0.), -jnp.inf, 0.)

    if mode == 'soft':
        return soft(violations)
    if settings['grad'] == 'soft':

        @jax.custom_jvp
        def hard_soft_grad(violations):
            return hard(violations)

        @hard_soft_grad.defjvp
        def hard_soft_grad_jvp(primals, tangents):
            (violations,), (dviolations,) = primals, tangents
            return hard(violations), jnp.sum(-(violations / scales ** 2) * dviolations)

        return hard_soft_grad(violations)
    return hard(violations)


_PRIOR_LIKE = ('prior', 'ref')


def _merged_distribution(current, update):
    """A new ParameterPrior: *update* replaces *current* when it names a ``dist``, else is merged into it."""
    from .parameter import ParameterPrior
    if 'dist' in update:
        state = dict(update)
        state.setdefault('shape', current.shape)
        return ParameterPrior(**state)
    state = current.__getstate__()
    attrs = state.pop('attrs', {}) if isinstance(state.get('attrs', None), dict) else {}
    state = {**state, **attrs, **update}
    return ParameterPrior(**state)


#: ``override(..., constraints=)`` not given: keep the active constraint settings.
_UNSET = object()


@contextlib.contextmanager
def override(graph, value=None, prior=None, ref=None, constraints=_UNSET):
    """Temporarily change parameter values, priors, refs and constraint settings of a built graph.

    Everything is restored on exit, also on error.  Structural changes (``fixed``, ``solved``,
    ``derived``, adding or removing parameters) change the graph itself and need a new
    :func:`~desilike.base.build`; they cannot be overridden.

    Parameters
    ----------
    graph : CompiledGraph or Calculator
        Graph whose parameters to override.
    value : dict, optional
        ``{name: value}``.
    prior, ref : dict, optional
        ``{name: {field: value}}``, merged into the current distribution (``{'limits': [0.6, 0.8]}``
        keeps the distribution and narrows its limits); a dict with ``dist`` replaces it.
    constraints : str, dict or None, optional
        How :class:`~desilike.parameter.Constraint` values enter the posterior: a mode, or
        ``{'mode': ..., 'scale': ..., 'grad': ...}``.  Not given: the active settings stay.

        - ``mode``: ``'hard'`` (the default outside any override: ``-inf`` wherever a constraint
          is violated, the exact truncated posterior), ``'soft'`` (add ``-1/2 sum (value / scale)^2``,
          continuous at the boundary) or ``None`` (ignore constraints).
        - ``scale``: soft-penalty widths, one for all or ``{constraint name: scale}``; default each
          constraint's own.
        - ``grad``: ``'soft'`` takes the derivative from the soft penalty while the value stays
          hard (with ``mode='hard'``).  No effect in NUTS (a step into the wall ends the trajectory
          before its gradient is used); meant for fixed-length HMC.
    """
    from .base import get_params
    params = get_params(graph)
    requested = {'value': value or {}, 'prior': prior or {}, 'ref': ref or {}}
    for field, entries in requested.items():
        unknown = [name for name in entries if name not in params]
        if unknown:
            raise ValueError(f'override {field}: unknown parameters {unknown}')
        if field in _PRIOR_LIKE:
            for name, update in entries.items():
                if not isinstance(update, dict):
                    raise TypeError(f'override {field}[{name!r}] must be a dict of distribution fields, not {type(update).__name__}')
                if not hasattr(params[name], field):
                    raise ValueError(f'override {field}: {name!r} is a {type(params[name]).__name__}, which has no {field}')
    saved = []
    try:
        for name, new in requested['value'].items():
            param = params[name]
            saved.append((param, '_value', param._value))
            param.value = new
        for field in _PRIOR_LIKE:
            for name, update in requested[field].items():
                param = params[name]
                saved.append((param, field, getattr(param, field)))
                setattr(param, field, _merged_distribution(getattr(param, field), update))
        settings_context = use_constraints(constraints if constraints is not _UNSET else (current_constraints() or 'hard'))
        with settings_context:
            _bump()
            yield params
    finally:
        for param, attribute, old in reversed(saved):
            setattr(param, attribute, old)
        _bump()
