"""Module implementing the ``emcee`` sampler."""

try:
    import emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False
import numpy as np

from .base import update_kwargs, MarkovChainSampler


class EmceeSampler(MarkovChainSampler):
    """Wrapper for the affine-invariant ensemble sampler ``emcee``.

    Reference
    ---------
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665

    """

    def __init__(self, likelihood, n_chains=4, rng=None, directory=None,
                 **kwargs):
        """Initialize the emcee sampler.

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
            Extra keyword arguments passed to ``emcee`` during initialization.

        """
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)

        kwargs = update_kwargs(kwargs, 'emcee', pool=self.pool, args=None,
                               kwargs=None, vectorize=False)

        if self.mpicomm.rank == 0:
            self.sampler = emcee.EnsembleSampler(
                self.n_chains, self.n_dim, self.compute_posterior, **kwargs)
        else:
            self.sampler = None

    def run_sampler(self, n_steps, **kwargs):
        """Run the emcee sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.
        kwargs: dict, optional
            Extra keyword arguments passed to emcee's ``run_mcmc`` method.

        """
        kwargs = update_kwargs(kwargs, 'emcee', store=True)

        try:
            self.sampler.get_last_sample()
            initial_state = None
        except AttributeError:
            initial_state = emcee.State(
                self.chains[:, -1, :], log_prob=self.log_post[:, -1],
                random_state=np.random.RandomState(self.rng.integers(
                    2**32 - 1)).get_state())

        self.sampler.run_mcmc(initial_state, n_steps, **kwargs)
        chains = np.transpose(self.sampler.get_chain()[-n_steps:],
                              (1, 0, 2))
        log_post = self.sampler.get_log_prob()[-n_steps:].T
        self.chains = np.concatenate([self.chains, chains], axis=1)
        self.log_post = np.concatenate([self.log_post, log_post], axis=1)
