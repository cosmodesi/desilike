"""Module implementing the zeus sampler."""

import warnings

import numpy as np
try:
    import zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import update_kwargs, MarkovChainSampler
from desilike.samples import Chain


class ZeusSampler(MarkovChainSampler):
    """Wrapper for the ensemble slice sampler ``zeus``.

    Reference
    ---------
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468

    """

    def __init__(self, likelihood, n_chains, rng=None, filepath=None,
                 **kwargs):
        """Initialize the ``zeus`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 10.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        filepath : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``zeus`` during initialization.

        """
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         filepath=filepath)

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
        kwargs: dict, optional
            Extra keyword arguments passed to zeus's ``run_mcmc`` method.

        Returns
        -------
        Chain
            Sampler results.

        """
        kwargs = update_kwargs(kwargs, 'zeus', nsteps=n_steps, log_prob0=None)

        try:
            self.sampler.get_last_sample()
            initialized = True
        except AttributeError:
            initialized = False

        if not initialized:
            start = np.zeros([self.n_chains, self.n_dim])
            log_post = np.zeros(self.n_chains)
            for i in range(self.n_chains):
                start[i] = [self.chains[i][param].value[-1] for param in
                            self.likelihood.varied_params.names()]
                log_post[i] = self.chains[i]['logposterior'].value[-1]
            kwargs['log_prob0'] = log_post

        else:
            start = None

        self.sampler.run_mcmc(start, **kwargs)

        chains = np.transpose(self.sampler.get_chain(), (1, 0, 2))
        log_post = self.sampler.get_log_prob().T
        for i in range(self.n_chains):
            chain = Chain(
                np.column_stack([chains[i], log_post[i]]).T,
                params=self.likelihood.varied_params + ['logposterior'])
            self.chains[i] = Chain.concatenate(
                self.chains[i], chain[-n_steps:])

    def reset_sampler(self):
        """Reset the emcee sampler."""
        self.sampler.reset()
