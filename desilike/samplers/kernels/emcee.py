"""Emcee ensemble MCMC kernel."""

import logging

import numpy as np
import jax
import jax.numpy as jnp

try:
    import emcee as _emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False

from .base import Kernel


def _build_flat_layout(params, param_shapes):
    """Return ``(name, size, shape, col_start)`` tuples in params order."""
    layout = []
    col = 0
    for name in params:
        shape = param_shapes[name]
        size = int(np.prod(shape)) if shape else 1
        layout.append((name, size, shape, col))
        col += size
    return layout


def _make_vmapped_log_prob(logposterior, layout):
    """Return a JAX-vmapped ``(N, ndim) → (N,)`` log-prob function."""
    def _one(flat):
        pos_dict = {name: flat[col:col + size].reshape(shape) if shape else flat[col]
                    for name, size, shape, col in layout}
        return logposterior(pos_dict)

    _vmapped = jax.jit(jax.vmap(_one))

    def log_prob_fn(positions):
        return np.asarray(_vmapped(jnp.asarray(positions)))

    return log_prob_fn


def _make_scalar_log_prob(logposterior, layout):
    """Return a per-walker ``(ndim,) → float`` log-prob function."""
    _jit = jax.jit(logposterior)

    def log_prob_fn(flat):
        pos_dict = {name: flat[col:col + size].reshape(shape) if shape else flat[col]
                    for name, size, shape, col in layout}
        return float(_jit(pos_dict))

    return log_prob_fn


class Emcee(Kernel):
    """Affine-invariant ensemble sampler (``emcee``).

    .. rubric:: References
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665
    """

    logger = logging.getLogger('Emcee')
    _sampler_cls = 'EnsembleKernelSampler'

    def __init__(self, nwalkers=None, **kwargs):
        """
        Parameters
        ----------
        nwalkers : int or None
            Number of walkers.  ``None`` defers to ``4 * ndim`` (set by
            :class:`~desilike.samplers.base.EnsembleKernelSampler` before :meth:`init`
            is called).
        **kwargs
            Extra keyword arguments forwarded to ``emcee.EnsembleSampler``.
        """
        self.nwalkers = nwalkers
        self._kwargs = kwargs

    @property
    def walker_shape(self):
        return (self.nwalkers,) if self.nwalkers is not None else ()

    def init(self, logposterior, params, rng, **context):
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not installed.")

        ndim = context['ndim']
        param_shapes = context['param_shapes']
        initial_position = context['initial_position']   # {name: (nwalkers, *shape)}

        layout = _build_flat_layout(params, param_shapes)
        log_prob_fn = _make_vmapped_log_prob(logposterior, layout)

        self._sampler = _emcee.EnsembleSampler(
            nwalkers=self.nwalkers, ndim=ndim,
            log_prob_fn=log_prob_fn, vectorize=True,
            **self._kwargs)

        # Build (nwalkers, ndim) initial positions and evaluate log-prob.
        positions = np.column_stack([
            np.asarray(initial_position[name]).reshape(self.nwalkers, -1)
            for name in params
        ])
        log_post = log_prob_fn(positions)
        rng_state = np.random.RandomState(int(rng.integers(2**32 - 1))).get_state()
        self._emcee_state = _emcee.State(positions, log_prob=log_post, random_state=rng_state)
        self._ndim = ndim
        self._total_likelihood_evaluations = 0

    def run(self, n_steps):
        samples  = np.zeros((n_steps, self.nwalkers, self._ndim))
        log_post = np.zeros((n_steps, self.nwalkers))
        for step_idx, state in enumerate(self._sampler.sample(
                self._emcee_state, iterations=n_steps, store=False)):
            samples[step_idx, :, :] = state.coords
            log_post[step_idx, :] = state.log_prob
        self._emcee_state = state
        self._total_likelihood_evaluations += n_steps * self.nwalkers
        self.logger.info('total likelihood evaluations: %d', self._total_likelihood_evaluations)
        return samples, log_post
