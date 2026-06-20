"""Zeus ensemble slice sampler kernel."""

import logging
import warnings

import numpy as np

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

    @classmethod
    def install(cls, installer):
        installer.pip('zeus-mcmc')

    def init(self, posterior, rng, **context):
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not installed.")

        _, self._log_prob_fn = posterior
        ndim = context['ndim']

        if self.nwalkers is None:
            self.nwalkers = 4 * ndim

        if rng is not None:
            warnings.warn('Zeus does not support random seeds. Results are not deterministic.')

        self._ndim = ndim
        self._pool = context['pool']
        self._sampler = None

    def run(self, n_steps, state):
        position, derived, logposterior = state
        nderived = derived.shape[-1]

        if self._sampler is None:
            import logging as _logging
            handlers = _logging.root.handlers.copy()
            level = _logging.root.level
            self._sampler = _zeus.EnsembleSampler(
                nwalkers=self.nwalkers, ndim=self._ndim,
                logprob_fn=self._log_prob_fn, pool=self._pool, **self._kwargs)
            _logging.root.handlers = handlers
            _logging.root.level = level

        samples = np.zeros((n_steps, self.nwalkers, self._ndim))
        log_post = np.zeros((n_steps, self.nwalkers))
        derived_out = np.zeros((n_steps, self.nwalkers, nderived))
        for step_idx, step_state in enumerate(self._sampler.sample(
                position, log_prob0=logposterior,
                blobs0=np.array(derived),
                iterations=n_steps, progress=False)):
            coords, log_prob_step, blobs = step_state
            samples[step_idx, :, :] = coords
            log_post[step_idx, :] = log_prob_step
            derived_out[step_idx, :, :] = np.array(blobs).reshape(self.nwalkers, nderived)

        return samples, derived_out if nderived else None, {'logposterior': log_post}
