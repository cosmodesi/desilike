"""Module implementing the Hamiltonian Monte-Carlo (HMC) sampling."""

try:
    import jax
    import blackjax
    BLACKJAX_INSTALLED = True
except ModuleNotFoundError:
    BLACKJAX_INSTALLED = False
import numpy as np

from .base import compute_posterior, MarkovChainSampler
from desilike.samples import Chain


class HMCSampler(MarkovChainSampler):
    """Wrapper for Hamiltonian Monte-Carlo (HMC).

    Reference
    ---------
    - https://github.com/blackjax-devs/blackjax

    """

    save_derived = False

    def __init__(self, likelihood, n_chains, adaptation=True,
                 adaptation_kwargs=dict(n_iters=1000), step_size=1e-3,
                 inv_mass_matrix=None, num_integration_steps=60, rng=None,
                 save_fn=None, mpicomm=None, **kwargs):
        """Initialize the HMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.

        n_chains : int
            Number of chains.

        adaptation : bool, optional
            Determine the best inverse mass matrix and step size during an
            initial warm-up phase. Default is True.

        adaptation_kwargs : dict or None, optional
            Additional keyword arguments passed to
            ``blackjax.window_adaptation``. The key 'n_iters' is not passed to
            ``blackjax.window_adaptation`` and instead determines the length
            of the warm-up phase. Default is ``dict(n_iters=1000)``.

        step_size : float, optional
            Size of the integration step. Default is 1e-3.

        inv_mass_matrix : numpy.ndarray, optional
            The value to use for the inverse mass matrix when drawing a value
            for the momentum and computing the kinetic energy. If
            one-dimensional, a diagonal mass matrix is assumed. If None,
            a unity matrix is used. Default is None.

        num_integration_steps : int, optional
            Number of times we run the symplectic integrator to build the
            trajectory. Default is 60.

        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.

        save_fn : str, Path, optional
            Save samples to this location. Default is ``None``.

        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.hmc`` during
            initialization.

        """
        if not BLACKJAX_INSTALLED:
            raise ImportError("The 'blackjax' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains, rng=rng, save_fn=save_fn,
                         mpicomm=mpicomm)

        if inv_mass_matrix is None:
            inv_mass_matrix = np.ones(self.n_dim)

        self.sampler = blackjax.hmc(
            compute_posterior, step_size, inv_mass_matrix,
            num_integration_steps, **kwargs)

        self.adaptation = adaptation
        self.adaptation_kwargs = adaptation_kwargs
        self.states = None

    def run_sampler(self, n_steps):
        """Run the HMC sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.

        """
        if self.states is None:
            self.states = []
            for i in range(self.n_chains):
                self.states.append(self.sampler.init(
                    {param: self.chains[i][param].value[-1] for param in
                     self.likelihood.varied_params.names()}))

        rng_keys = jax.random.split(jax.random.PRNGKey(
            self.rng.integers(2**32)), self.n_chains)

        def one_step(state, rng_key):
            state, _ = self.sampler.step(rng_key, state)
            return state, state

        for i in range(self.n_chains):
            final_state, states = jax.lax.scan(
                one_step, self.states[i], jax.random.split(
                    rng_keys[i], n_steps))

            chain = states.position
            chain['logposterior'] = states.logdensity
            self.chains[i] = Chain.concatenate(self.chains[i], Chain(chain))
            self.states[i] = final_state
