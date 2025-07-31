"""Module implementing the zeus sampler."""

try:
    import zeus
    ZEUS_INSTALLED = True
except ModuleNotFoundError:
    ZEUS_INSTALLED = False

from .base import compute_posterior, update_kwargs, MarkovChainSampler
from desilike.samples import Chain


class ZeusSampler(MarkovChainSampler):
    """Wrapper for the ensemble slice sampler zeus.

    Reference
    ---------
    - https://github.com/minaskar/zeus
    - https://arxiv.org/abs/2002.06212
    - https://arxiv.org/abs/2105.03468

    """

    def __init__(self, likelihood, n_chains, rng=None, save_fn=None,
                 mpicomm=None, **kwargs):
        """Initialize the zeus sampler.

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
        if not ZEUS_INSTALLED:
            raise ImportError("The 'zeus-mcmc' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains, rng=rng, save_fn=save_fn,
                         mpicomm=mpicomm)

        kwargs = update_kwargs(kwargs, 'zeus', pool=self.pool, args=None,
                               kwargs=None, vectorize=False)

        if self.mpicomm.rank == 0:
            self.sampler = zeus.EnsembleSampler(
                self.n_chains, self.n_dim, compute_posterior, **kwargs)
        else:
            self.sampler = None

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

        if self.chains is None:
            start = self.start
            kwargs['log_prob0'] = self.log_p_start
        else:
            start = None

        self.sampler.run_mcmc(start, **kwargs)

        chains_data = self.sampler.get_chain()
        log_p = self.sampler.get_log_prob()
        self.chains = []
        for i in range(self.n_chains):
            self.chains.append(Chain(
                [p for p in chains_data[:, i, :].T] + [log_p[:, i]],
                params=self.likelihood.varied_params + ['logposterior']))
