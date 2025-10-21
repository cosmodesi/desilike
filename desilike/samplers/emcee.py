"""Module implementing the ``emcee`` sampler."""

try:
    import emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False
import numpy as np

from .base import compute_posterior, update_kwargs, MarkovChainSampler
from desilike.samples import Chain


class EmceeSampler(MarkovChainSampler):
    """Wrapper for the affine-invariant ensemble sampler ``emcee``.

    Reference
    ---------
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665

    """

    def __init__(self, likelihood, n_chains=10, rng=None, filepath=None,
                 **kwargs):
        """Initialize the emcee sampler.

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
            Extra keyword arguments passed to ``emcee`` during initialization.

        """
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         filepath=filepath)

        kwargs = update_kwargs(kwargs, 'emcee', pool=self.pool, args=None,
                               kwargs=None, vectorize=False)

        if self.mpicomm.rank == 0:
            self.sampler = emcee.EnsembleSampler(
                self.n_chains, self.n_dim, compute_posterior, **kwargs)
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
            initialized = True
        except AttributeError:
            initialized = False

        if not initialized:
            coords = np.zeros([self.n_chains, self.n_dim])
            log_post = np.zeros(self.n_chains)
            for i in range(self.n_chains):
                coords[i] = [self.chains[i][param].value[-1] for param in
                             self.likelihood.varied_params.names()]
                log_post[i] = self.chains[i]['logposterior'].value[-1]
            initial_state = emcee.State(
                coords, log_prob=log_post,
                random_state=np.random.RandomState(self.rng.integers(
                    2**32 - 1)).get_state())
        else:
            initial_state = None

        self.sampler.run_mcmc(initial_state, n_steps, **kwargs)

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
