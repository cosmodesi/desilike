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
from .emcee import _build_flat_layout, _make_scalar_log_prob


class Zeus(Kernel):
    """Ensemble slice sampler (``zeus``).

    .. rubric:: References
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468
    """

    logger = logging.getLogger('Zeus')
    _sampler_cls = 'EnsembleKernelSampler'

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

    @property
    def walker_shape(self):
        return (self.nwalkers,) if self.nwalkers is not None else ()

    def init(self, logposterior, params, rng, **context):
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not installed.")

        ndim = context['ndim']
        param_shapes = context['param_shapes']
        initial_position = context['initial_position']

        layout = _build_flat_layout(params, param_shapes)
        log_prob_fn = _make_scalar_log_prob(logposterior, layout)

        # Zeus modifies the root logging configuration on initialisation.
        import logging as _logging
        handlers = _logging.root.handlers.copy()
        level = _logging.root.level
        self._sampler = _zeus.EnsembleSampler(
            nwalkers=self.nwalkers, ndim=ndim, logprob_fn=log_prob_fn, **self._kwargs)
        _logging.root.handlers = handlers
        _logging.root.level = level

        if rng is not None:
            warnings.warn('Zeus does not support random seeds. Results are not deterministic.')

        positions = np.column_stack([
            np.asarray(initial_position[name]).reshape(self.nwalkers, -1)
            for name in params
        ])
        log_post_init = np.array([log_prob_fn(pos) for pos in positions])
        self._current_positions = positions
        self._current_log_post  = log_post_init
        self._ndim = ndim

    def run(self, n_steps):
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
