"""Importance sampling kernel."""

import logging

import numpy as np
from scipy.special import logsumexp

from .base import StaticKernel


class Importance(StaticKernel):
    """Reweight an existing sample under a new posterior via importance sampling.

    Pass an :class:`~desilike.samples.MCSamples` object as the ``samples``
    keyword argument to :meth:`~desilike.samplers.base.StaticSampler.run`:

    .. code-block:: python

        sampler = Sampler(new_posterior, kernel=Importance())
        new_samples = sampler.run(samples=old_samples, resample=True)
    """

    logger = logging.getLogger('Importance')

    def get_samples(self, varied_params, samples=None, **kwargs):
        """Extract parameter columns from the input samples.

        Parameters
        ----------
        varied_params : VariableCollection
        samples : MCSamples
            Input samples from the old posterior.

        Returns
        -------
        numpy.ndarray, shape ``(n_samples, ndim)``
        """
        return np.column_stack([samples[key].value for key in varied_params])

    def post_process(self, results, samples=None, resample=True, **kwargs):
        """Reweight ``results`` relative to the original ``samples``.

        Parameters
        ----------
        results : MCSamples or None
            Newly evaluated samples on the main rank; ``None`` on workers.
        samples : MCSamples
            Original samples carrying the old log-posterior.
        resample : bool, optional
            If ``True`` (default), weights are ``exp(new_log_post - old_log_post)``.
            If ``False``, weights are ``exp(new_log_post - new_log_prior + old_log_post)``,
            combining the old posterior with the new likelihood.

        Returns
        -------
        MCSamples or None
        """
        if results is None:
            return None
        if resample:
            log_w = results.logposterior - samples.logposterior
        else:
            log_w = results.logposterior - results.logprior + samples.logposterior
        results.aweight = np.exp(log_w - logsumexp(log_w))
        return results
