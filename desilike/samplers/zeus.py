"""Module implementing the zeus sampler."""

import warnings

import numpy as np
try:
    import zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import update_parameters, MarkovChainSampler


class ZeusSampler(MarkovChainSampler):
    """Wrapper for the ensemble slice sampler ``zeus``.

    .. rubric:: References
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468

    """

    def __init__(self, likelihood, n_chains=4, chains=None, rng=None,
                 directory=None, **kwargs):
        """Initialize the ``zeus`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        chains : list of desilike.samples.Chain, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``zeus`` during initialization.

        """
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, chains=chains, rng=rng,
                         directory=directory)

        if self.pool.main:
            kwargs = update_parameters(
                kwargs, 'zeus', nwalkers=self.n_chains, ndim=self.n_dim,
                logprob_fn=self.compute_posterior, pool=self.pool, args=None,
                kwargs=None, vectorize=False)
            self.sampler = zeus.EnsembleSampler(**kwargs)

            if rng is not None:
                warnings.warn("Zeus does not support random seeds. Results "
                              "are not deterministic.")

    def run_sampler(self, n_steps):
        """Run the ``zeus`` sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.

        """
        start, blobs0, log_prob0 = self.state
        samples = np.zeros((self.n_chains, n_steps, self.n_dim))
        derived = np.zeros((self.n_chains, n_steps, self.n_derived))
        log_post = np.zeros((self.n_chains, n_steps))
        for i, state in enumerate(self.sampler.sample(
                start, log_prob0=log_prob0, blobs0=np.squeeze(blobs0),
                iterations=n_steps, progress=False)):
            samples[:, i, :] = state[0]
            derived[:, i, :] = state[2].reshape(self.n_chains, -1)
            log_post[:, i] = state[1]

        self.extend(samples, derived, log_post)
