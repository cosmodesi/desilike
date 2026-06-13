"""Emcee ensemble MCMC kernel."""

import logging

import numpy as np
import jax.numpy as jnp

try:
    import emcee as _emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False

from .base import Kernel


class Emcee(Kernel):
    """Affine-invariant ensemble sampler (``emcee``).

    .. rubric:: References
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665
    """

    logger = logging.getLogger('Emcee')
    _sampler_cls = 'EnsembleSampler'

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

    def init(self, posterior_logpdf, rng, **context):
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not installed.")

        if self.nwalkers is None:
            self.nwalkers = 4 * context['ndim']

        self._nderived = context.get('nderived', 0)
        self._rng = rng
        self._ndim = context['ndim']
        self._sampler = None
        self._emcee_state = None
        self._total_likelihood_evaluations = 0

        if self._nderived:
            _posterior_with_derived = context['posterior_with_derived']

            def log_prob_fn(positions):
                # emcee expects per-walker (log_prob, blob) even with vectorize=True
                log_post_batch, derived_batch = _posterior_with_derived(jnp.asarray(positions))
                log_post_np = np.asarray(log_post_batch)
                derived_np  = np.asarray(derived_batch)
                return [(float(log_post_np[walker_idx]), derived_np[walker_idx])
                        for walker_idx in range(len(log_post_np))]
        else:
            def log_prob_fn(positions):
                return np.asarray(posterior_logpdf(jnp.asarray(positions)))

        self._log_prob_fn = log_prob_fn

    def run(self, n_steps, initial_position=None):
        if self._sampler is None:
            self._sampler = _emcee.EnsembleSampler(
                nwalkers=self.nwalkers, ndim=self._ndim,
                log_prob_fn=self._log_prob_fn, vectorize=True,
                **self._kwargs)
            rng_state = np.random.RandomState(int(self._rng.integers(2**32 - 1))).get_state()
            if self._nderived:
                walker_results = self._log_prob_fn(initial_position)
                log_post_init = np.array([result[0] for result in walker_results])
                blobs_init    = np.array([result[1] for result in walker_results])
                self._emcee_state = _emcee.State(initial_position, log_prob=log_post_init,
                                                 blobs=blobs_init, random_state=rng_state)
            else:
                log_post_init = self._log_prob_fn(initial_position)
                self._emcee_state = _emcee.State(initial_position, log_prob=log_post_init,
                                                 random_state=rng_state)

        samples  = np.zeros((n_steps, self.nwalkers, self._ndim))
        log_post = np.zeros((n_steps, self.nwalkers))
        if self._nderived:
            derived = np.zeros((n_steps, self.nwalkers, self._nderived))
        for step_idx, state in enumerate(self._sampler.sample(
                self._emcee_state, iterations=n_steps, store=False)):
            samples[step_idx, :, :] = state.coords
            log_post[step_idx, :] = state.log_prob
            if self._nderived:
                derived[step_idx, :, :] = np.asarray(state.blobs).reshape(self.nwalkers, -1)
        self._emcee_state = state
        self._total_likelihood_evaluations += n_steps * self.nwalkers
        self.logger.info('total likelihood evaluations: %d', self._total_likelihood_evaluations)
        if self._nderived:
            return samples, derived, {'logposterior': log_post}
        return samples, None, {'logposterior': log_post}
