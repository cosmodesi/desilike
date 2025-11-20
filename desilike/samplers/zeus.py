"""Module implementing the zeus sampler."""

import warnings

import numpy as np
try:
    import zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import update_kwargs, MarkovChainSampler


class ZeusSampler(MarkovChainSampler):
    """Wrapper for the ensemble slice sampler ``zeus``.

    Reference
    ---------
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468

    """

    def __init__(self, likelihood, n_chains=4, rng=None, directory=None,
                 **kwargs):
        """Initialize the ``zeus`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``zeus`` during initialization.

        """
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)

        kwargs = update_kwargs(kwargs, 'zeus', pool=self.pool, args=None,
                               kwargs=None, vectorize=False)

        if self.mpicomm.rank == 0:
            self.sampler = zeus.EnsembleSampler(
                self.n_chains, self.n_dim, self.compute_posterior, **kwargs)
        else:
            self.sampler = None

        if rng is not None:
            warnings.warn("Zeus does not support random seeds. Results are "
                          "not reproducible.")

    def run_sampler(self, n_steps, **kwargs):
        """Run the zeus sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.
        kwargs: dict, optional
            Extra keyword arguments passed to zeus's ``run_mcmc`` method.

        """
        kwargs = update_kwargs(kwargs, 'zeus', nsteps=n_steps, log_prob0=None)

        try:
            self.sampler.get_last_sample()
            start = None
        except AttributeError:
            start = self.chains[:, -1, :]
            kwargs['log_prob0'] = self.log_post[:, -1]

        self.sampler.run_mcmc(start, **kwargs)
        chains = np.transpose(self.sampler.get_chain()[-n_steps:],
                              (1, 0, 2))
        log_post = self.sampler.get_log_prob()[-n_steps:].T
        self.chains = np.concatenate([self.chains, chains], axis=1)
        self.log_post = np.concatenate([self.log_post, log_post], axis=1)
