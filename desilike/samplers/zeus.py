"""Module implementing the zeus sampler."""

import warnings

import numpy as np
try:
    import zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import update_parameters, EnsembleSampler


class ZeusSampler(EnsembleSampler):
    """Wrapper for the ensemble slice sampler ``zeus``.

    .. rubric:: References
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468

    """

    def __init__(self, likelihood, n_chains=1, chains=None, rng=None,
                 directory=None, nwalkers=None, **kwargs):
        """Initialize the ``zeus`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 1.
        chains : list of desilike.samples.Chain, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        nwalkers : int, optional
            Number of walkers, defaults to :attr:`Chain.shape[1]` of input chains, if any,
            else ``2 * ndim``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``zeus`` during initialization.

        """
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, chains=chains, rng=rng,
                         directory=directory, nwalkers=nwalkers)
        if self.nwalkers is None:
            self.nwalkers = 2 * self.n_dim
        if self.pool.main:
            kwargs = update_parameters(
                kwargs, 'zeus', nwalkers=self.nwalkers, ndim=self.n_dim,
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
        if self.pool.main:
            start, blobs0, log_prob0 = self.state
            samples = np.zeros((n_steps, self.nwalkers, self.n_dim))
            derived = np.zeros((n_steps, self.nwalkers, self.n_derived))
            log_post = np.zeros((n_steps, self.nwalkers))
            for i, state in enumerate(self.sampler.sample(
                    start, log_prob0=log_prob0, blobs0=np.squeeze(blobs0),
                    iterations=n_steps, progress=False)):
                samples[i, :, :] = state[0]
                derived[i, :, :] = state[2]
                log_post[i, :] = state[1]

            self.extend(samples, derived, log_post)
            self.pool.stop_wait()
        else:
            self.pool.wait()
