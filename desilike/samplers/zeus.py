"""Zeus ensemble slice sampler kernel."""

import logging
import warnings

import numpy as np
import jax

try:
    import zeus as _zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import Kernel


class Zeus(Kernel):
    """Ensemble slice sampler (``zeus``).

    .. rubric:: References
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468
    """

    logger = logging.getLogger('Zeus')
    _sampler_cls = 'EnsembleSampler'

    def __init__(self, nwalkers=None, **kwargs):
        """
        Parameters
        ----------
        nwalkers : int or None
            Number of walkers.  ``None`` defers to ``4 * ndim``.
        **kwargs
            Extra keyword arguments forwarded to ``zeus.EnsembleSampler``.
        """
        self.nwalkers = nwalkers
        self._kwargs = kwargs

    def init(self, posterior_logpdf, rng, **context):
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not installed.")

        if self.nwalkers is None:
            self.nwalkers = 4 * context['ndim']

        if rng is not None:
            warnings.warn('Zeus does not support random seeds. Results are not deterministic.')

        import jax.numpy as jnp
        self._log_prob_fn = lambda flat: float(posterior_logpdf(jnp.asarray(flat)[None])[0])
        self._ndim = context['ndim']
        self._sampler = None
        self._current_positions = None
        self._current_log_post = None

    def run(self, n_steps, initial_position=None):
        if self._sampler is None:
            import logging as _logging
            handlers = _logging.root.handlers.copy()
            level = _logging.root.level
            self._sampler = _zeus.EnsembleSampler(
                nwalkers=self.nwalkers, ndim=self._ndim,
                logprob_fn=self._log_prob_fn, **self._kwargs)
            _logging.root.handlers = handlers
            _logging.root.level = level

            # initial_position is (nwalkers, ndim)
            self._current_positions = initial_position
            self._current_log_post = np.array([self._log_prob_fn(pos) for pos in initial_position])

        samples  = np.zeros((n_steps, self.nwalkers, self._ndim))
        log_post = np.zeros((n_steps, self.nwalkers))
        for step_idx, state in enumerate(self._sampler.sample(
                self._current_positions, log_prob0=self._current_log_post,
                iterations=n_steps, progress=False)):
            coords, log_prob, _ = state
            samples[step_idx, :, :] = coords
            log_post[step_idx, :] = log_prob
        self._current_positions = samples[-1]
        self._current_log_post  = log_post[-1]
        return samples, log_post
