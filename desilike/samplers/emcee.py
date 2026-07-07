"""Emcee ensemble MCMC kernel."""

import logging

import numpy as np

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

    @classmethod
    def install(cls, installer):
        installer.pip('emcee')

    def init(self, posterior, rng, **context):
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not installed.")

        plain_log_prob_fn, with_derived_log_prob_fn = posterior
        # emcee auto-detects "blobs" from whether log_prob_fn returns a tuple; the
        # with-derived core always returns a (logpost, derived) tuple even when there
        # are no derived params (derived is then a zero-width array), which desyncs
        # from the blobs=None initial state below. Use the plain scalar-returning core
        # in that case so emcee's blob detection matches what we actually pass around.
        self._log_prob_fn = with_derived_log_prob_fn if context.get('nderived', 0) else plain_log_prob_fn
        ndim = context['ndim']

        if self.nwalkers is None:
            self.nwalkers = 4 * ndim

        self._rng = rng
        self._ndim = ndim
        self._pool = context['pool']
        self._sampler = None
        self._total_likelihood_evaluations = 0

    def run(self, n_steps, state):
        position, derived, logposterior = state
        nderived = derived.shape[-1]

        if self._sampler is None:
            self._sampler = _emcee.EnsembleSampler(
                nwalkers=self.nwalkers, ndim=self._ndim,
                log_prob_fn=self._log_prob_fn, pool=self._pool,
                vectorize=False, **self._kwargs)

        rng_state = np.random.RandomState(int(self._rng.integers(2**32 - 1))).get_state()
        emcee_state = _emcee.State(
            position,
            blobs=derived if nderived else None,
            log_prob=logposterior,
            random_state=rng_state)

        samples      = np.zeros((n_steps, self.nwalkers, self._ndim))
        log_post     = np.zeros((n_steps, self.nwalkers))
        if nderived:
            derived_out = np.zeros((n_steps, self.nwalkers, nderived))
        for step_idx, step_state in enumerate(self._sampler.sample(
                emcee_state, iterations=n_steps, store=False)):
            samples[step_idx, :, :] = step_state.coords
            log_post[step_idx, :] = step_state.log_prob
            if nderived:
                derived_out[step_idx, :, :] = np.array(step_state.blobs).reshape(self.nwalkers, -1)
        self._total_likelihood_evaluations += n_steps * self.nwalkers
        self.logger.info('total likelihood evaluations: %d', self._total_likelihood_evaluations)

        return samples, (derived_out if nderived else None), {'logposterior': log_post}
