"""Module implementing an importance sampler."""

import numpy as np
from scipy.special import logsumexp

from .base import StaticSampler


class ImportanceSampler(StaticSampler):
    """An importance sampler.

    This class can be used to transform points from one posterior to another.
    Alternatively, it can also be used to combine likelihoods from two
    experiments.
    """

    def get_points(self, chain=None):
        """Get points on the grid.

        Parameters
        ----------
        chain : desilike.samples.Chain, optional
            Input chain that defines the points. Default is None.

        Returns
        -------
        numpy.ndarray of shape (n_points, n_dim)
            Grid to be evaluated.
        """
        return np.array([chain[key].value for key in
                         self.likelihood.varied_params.names()])

    def run(self, chain, mode='resample'):
        """Importance sample a chain.

        Parameters
        ----------
        chain : desilike.samples.Chain
            Input chain that samples a posterior.
        mode : str, optional
            If 'resample', the new weights for the chain will be the ratio
            of the new and old posterior. Effectively, the new chain will
            sample the new posterior. If 'combine', the new weights are the
            product of the old posterior and the new likelihood. Default is
            'resample'.

        Returns
        -------
        desilike.samples. Chain
            Sampler results.
        """
        if mode not in ['resample', 'combine']:
            raise ValueError(
                f"Unkown mode '{mode}'. Choose 'resample' or 'combine'.")

        chain_new = super().run(chain=chain)
        chain_old = chain
        if mode == 'resample':
            chain = chain_new
            log_w = chain_new.logposterior - chain_old.logposterior
        else:
            chain = chain_old.copy()
            chain.loglikelihood += chain_new.loglikelihood
            chain.logposterior += chain_new.loglikelihood
            log_w = chain_new.loglikelihood
        chain.aweight *= np.exp(log_w - logsumexp(log_w))
        return chain
