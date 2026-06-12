"""Metropolis-Hastings kernel."""

import logging

import numpy as np
import jax

from .base import Kernel


class MetropolisHastings(Kernel):
    """Metropolis-Hastings sampler with fast-slow decomposition.

    .. rubric:: References
    - https://arxiv.org/abs/1304.4473
    """

    logger = logging.getLogger('MetropolisHastings')
    _sampler_cls = 'MCMCSampler'

    def __init__(self, f_fast=1, f_drag=0, fast=None, covariance=None):
        """
        Parameters
        ----------
        f_fast : int
            Fast-parameter oversampling factor.  Default is 1 (no oversampling).
        f_drag : int
            Dragging factor for fast parameters.  Default is 0 (no dragging).
        fast : list of str or None
            Names of parameters considered "fast".  Default is ``None`` (none).
        covariance : array_like or None
            Initial proposal covariance in *rescaled* parameter space.  When
            ``None``, the identity matrix is used (suitable when the sampler
            uses ``rescale=True`` or the posterior is already ~unit-variance).
        """
        self.f_fast = f_fast
        self.f_drag = f_drag
        self.fast = list(fast) if fast is not None else []
        self.covariance = covariance

    def init(self, logposterior, params, rng, **context):
        from desilike.samplers.mhmcmc import StandAloneMetropolisHastingsSampler

        ndim = context['ndim']
        param_shapes = context['param_shapes']
        initial_position = context['initial_position']   # {name: shaped_array}, rescaled space

        # Map fast parameter names to flat column indices.
        flat_fast_indices = []
        col = 0
        for name in params:
            shape = param_shapes[name]
            size = int(np.prod(shape)) if shape else 1
            if name in self.fast:
                flat_fast_indices.extend(range(col, col + size))
            col += size

        # Build layout for flat ↔ dict conversion.
        layout = []
        col = 0
        for name in params:
            shape = param_shapes[name]
            size = int(np.prod(shape)) if shape else 1
            layout.append((name, size, shape, col))
            col += size

        _jit_logpost = jax.jit(logposterior)

        def _posterior(flat):
            # Called per-walker by StandAloneMetropolisHastingsSampler.
            pos_dict = {name: flat[c:c + sz].reshape(sh) if sh else flat[c]
                        for name, sz, sh, c in layout}
            return float(_jit_logpost(pos_dict))

        self._standalone = StandAloneMetropolisHastingsSampler(
            _posterior, fast=flat_fast_indices,
            f_fast=self.f_fast, f_drag=self.f_drag, rng=rng)

        # Build initial flat position vector.
        initial_flat = np.concatenate([
            np.asarray(initial_position[name]).ravel() for name in params
        ])
        initial_log_p = _posterior(initial_flat)

        cov = np.asarray(self.covariance) if self.covariance is not None else np.eye(ndim)
        self._standalone.update(
            pos=initial_flat[None, :],
            log_p=np.array([initial_log_p]),
            blobs=np.zeros((1, 0)),
            cov=cov)

        self._ndim = ndim
        self._adaptation_steps = 0
        self._accumulated_samples = []
        self._total_steps = 0

    def adapt(self, **kwargs):
        """Store the adaptation horizon; proposal covariance is updated inline during :meth:`run`."""
        self._adaptation_steps = int(kwargs.get('steps', 0))

    def run(self, n_steps):
        chains, _blobs, log_p = self._standalone.make_n_steps(n_steps)
        # chains: (1, n_steps, ndim); log_p: (1, n_steps)
        samples  = chains[0]    # (n_steps, ndim)
        log_post = log_p[0]     # (n_steps,)

        self._total_steps += n_steps

        # Adapt the proposal covariance from accumulated samples while within
        # the adaptation window.
        if self._total_steps < self._adaptation_steps:
            self._accumulated_samples.append(samples)
            all_samps = np.concatenate(self._accumulated_samples, axis=0)
            if len(all_samps) > self._ndim:
                try:
                    self._standalone.update(cov=np.cov(all_samps.T))
                except np.linalg.LinAlgError:
                    pass

        return samples, log_post
