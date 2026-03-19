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
        chain : desilike.Samples, optional
            Input chain that defines the samples.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_dim)
            Grid to be evaluated.

        """
        return np.column_stack([
            samples[key] for key in self.likelihood.varied_params.names()])

    def run(self, samples, resample=True):
        """Reweight a sample using importance sampling.

        Parameters
        ----------
        samples : desilike.samples.Chain
            Input samples with a corresponding posterior.
        resample : bool, optional
            If True, the weights for the chain will be multiplied by the ratio
            of the new and old posterior. Effectively, the new chain will
            sample the new posterior (if it previously sampled the old). If
            False, the weights will be multiplied by the the new likelihood.
            This can be useful when combining observations. Default is True.

        Returns
        -------
        desilike.Samples

        """
        results = super().run(samples=samples)

        if resample:
            log_weight = results['log_posterior'] - samples['log_posterior']
        else:
            log_weight = results['log_posterior'] - results['log_prior']

        results['log_weight'] = np.log(samples.weight) + log_weight
        return results
