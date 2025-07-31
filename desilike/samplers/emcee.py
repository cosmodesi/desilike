"""Module implementing the emcee sampler."""

try:
    import emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False

from .base import compute_posterior, update_kwargs, MarkovChainSampler
from desilike.samples import Chain


class EmceeSampler(MarkovChainSampler):
    """Wrapper for the affine-invariant ensemble sampler emcee.

    Reference
    ---------
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665

    """

    def __init__(self, likelihood, n_chains, rng=None, save_fn=None,
                 mpicomm=None, **kwargs):
        """Initialize the emcee sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.

        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.

        save_fn : str, Path, optional
            Save samples to this location. Default is ``None``.

        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        kwargs: dict, optional
            Extra keyword arguments passed to dynesty during initialization.

        """
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains, rng=rng, save_fn=save_fn,
                         mpicomm=mpicomm)

        kwargs = update_kwargs(kwargs, 'emcee', pool=self.pool, args=None,
                               kwargs=None, vectorize=False)

        if self.mpicomm.rank == 0:
            self.sampler = emcee.EnsembleSampler(
                self.n_chains, self.n_dim, compute_posterior, **kwargs)
        else:
            self.sampler = None

    def run_sampler(self, start, n_steps, **kwargs):
        """Run the emcee sampler.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to emcee's ``run_mcmc`` method.

        Returns
        -------
        Chain
            Sampler results.

        """
        kwargs = update_kwargs(kwargs, 'emcee', rstate0=self.rng)

        self.sampler.run_mcmc(start, n_steps, **kwargs)

        chains_data = self.sampler.get_chain()
        log_p = self.sampler.get_log_prob()
        self.chains = []
        for i in range(self.n_chains):
            self.chains.append(Chain(
                [p for p in chains_data[:, i, :].T] + [log_p[:, i]],
                params=self.likelihood.varied_params + ['logposterior']))

    @classmethod
    def install(cls, config):
        """Install emcee."""
        config.pip('emcee')
