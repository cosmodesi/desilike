"""Module implementing the zeus sampler."""

import warnings
import logging

import numpy as np
try:
    import zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import update_kwargs, EnsembleSampler


class ZeusSampler(EnsembleSampler):
    """Wrapper for the ensemble slice sampler ``zeus``.

    .. rubric:: References
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468

    """

    def __init__(self, posterior, nchains=1, chains=None, rng=None,
                 directory=None, nwalkers=None, rescale=False, covariance=None, **kwargs):
        """Initialize the ``zeus`` sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        nwalkers : int, optional
            Number of walkers, defaults to :attr:`MCSamples.shape[1]` of input chains, if any,
            else ``2 * ndim``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``zeus`` during initialization.

        """
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not "
                              "installed.")

        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, nwalkers=nwalkers,
                         rescale=rescale, covariance=covariance)
        if self.nwalkers is None:
            self.nwalkers = 2 * self.ndim
        if self.pool.main:
            # zeus treats tuple returns as (log_prob, blobs).  When there are
            # no derived parameters, return plain scalars to avoid a zero-size
            # blob array that breaks zeus's internal dtype inference.
            # compute_posterior is batched: it takes an (N, ndim) batch and
            # returns a list of N (log_post, derived) tuples.
            if self.n_derived:
                logprob_fn = self.compute_posterior
            else:
                logprob_fn = lambda batch: [result[0] for result in self.compute_posterior(batch)]
            kwargs = update_kwargs(
                kwargs, 'zeus', nwalkers=self.nwalkers, ndim=self.ndim,
                logprob_fn=logprob_fn, pool=self.pool, args=None,
                kwargs=None, vectorize=False)
            # Zeus modifies the logging handler
            handlers = logging.root.handlers.copy()
            level = logging.root.level
            self.sampler = zeus.EnsembleSampler(**kwargs)
            # Restore logging handler
            logging.root.handlers = handlers
            logging.root.level = level

            if rng is not None:
                warnings.warn("Zeus does not support random seeds. Results "
                              "are not deterministic.")

    def adapt_sampler(self, steps):
        """No-op: zeus does not support explicit adaptation."""

    def run_sampler(self, n_steps):
        """Run the ``zeus`` sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.

        """
        if self.pool.main:
            start, blobs0, log_prob0 = self.state
            samples  = np.zeros((n_steps, self.nwalkers, self.ndim))
            derived  = np.zeros((n_steps, self.nwalkers, self.n_derived))
            log_post = np.zeros((n_steps, self.nwalkers))
            for i, state in enumerate(self.sampler.sample(
                    start, log_prob0=log_prob0,
                    blobs0=np.squeeze(blobs0) if self.n_derived else None,
                    iterations=n_steps, progress=False)):
                samples[i, :, :] = state[0]
                if self.n_derived:
                    derived[i, :, :] = np.asarray(state[2]).reshape(self.nwalkers, -1)
                log_post[i, :] = state[1]

            self.extend(samples, derived, log_post)
            self.pool.stop_wait()
        else:
            self.pool.wait()
