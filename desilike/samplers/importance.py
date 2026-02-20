"""Module implementing an importance sampler."""

import numpy as np
from scipy.special import logsumexp

from .base import StaticSampler


class ImportanceSampler(StaticSampler):
    """An importance sampler.

    This class can be used to transform samples from one posterior to another.
    Alternatively, it can also be used to combine likelihoods from two
    experiments.
    """

    def get_samples(self, samples=None):
        """Get samples on the grid.

        Parameters
        ----------
        chain : desilike.samples.Chain, optional
            Input chain that defines the samples.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_dim)
            Grid to be evaluated.

        """
        return np.column_stack([
            samples[key].value for key in self.likelihood.varied_params])

    def run(self, samples, resample=True):
        """Reweight a sample using importance sampling.

        Parameters
        ----------
        samples : desilike.samples.Chain
            Input samples with a corresponding posterior.
        resample : bool, optional
            If True, the new weights for the chain will be the ratio of the new
            and old posterior. Effectively, the new chain will sample the new
            posterior. If False, the new weights are the product of the old
            posterior and the new likelihood. Default is True.

        Returns
        -------
        desilike.samples. Chain
            Sampler results.

        """
        results = super().run(samples=samples)

        if resample:
            log_w = results.logposterior - samples.logposterior
        else:
            log_w = (results.logposterior - results[results._logprior] +
                     samples.logposterior)

        results.aweight = np.exp(log_w - logsumexp(log_w))
        return results
