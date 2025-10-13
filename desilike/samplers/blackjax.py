"""Module implementing the BlackJAX samplers."""

try:
    import jax
    import blackjax
    BLACKJAX_INSTALLED = True
except ModuleNotFoundError:
    BLACKJAX_INSTALLED = False
import numpy as np

from .base import compute_posterior, MarkovChainSampler
from desilike.samples import Chain, Samples


# TODO: Properly implement abstract classes and methods.


SAMPLER = None


def make_one_step(state, rng_key):
    """Advance the sampler by one step.

    Parameters
    ----------
    state : NamedTuple
        State of the sampler.
    rng_key : jax.Array
        Random state.

    Returns
    -------
    state : NamedTuple
        New state of the sampler.
    state : NamedTuple
        Returned again for use in `jax.lax.scan`.

    """
    state, _ = SAMPLER.step(rng_key, state)
    return state, state


def make_n_steps(args):
    """Advance the state by several steps.

    Parameters
    ----------
    args : tuple
        Blackjax state, random key, and number of steps.

    Returns
    -------
    final_state : NamedTuple
        Final state after all steps.
    states : NamedTuple
        All sampled states.

    """
    state, rng_key, n_steps = args
    return jax.lax.scan(make_one_step, state, jax.random.split(
        rng_key, n_steps))


class BlackJAXSampler(MarkovChainSampler):
    """Wrapper for BlackJAX samplers.

    Reference
    ---------
    - https://github.com/blackjax-devs/blackjax

    """

    def __init__(self, likelihood, n_chains, rng=None, save_fn=None,
                 mpicomm=None, **kwargs):
        """Initialize the HMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.

        n_chains : int
            Number of chains.

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

        global SAMPLER
        if SAMPLER is not None:
            raise RuntimeError("Cannot create multiple instances of BlackJAX "
                               "samplers simultaneously.")

        super().__init__(likelihood, n_chains, rng=rng, save_fn=save_fn,
                         mpicomm=mpicomm)
        self.states = None

    def run_sampler(self, n_steps):
        """Run the BlackJAX sampler.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.

        """
        self.save_derived = False

        if self.states is None:
            self.states = []
            for i in range(self.n_chains):
                self.states.append(SAMPLER.init(
                    {param: self.chains[i][param].value[-1] for param in
                     self.likelihood.varied_params.names()}))

        rng_keys = jax.random.split(jax.random.PRNGKey(
            self.rng.integers(2**32)), self.n_chains)

        inputs = [(self.states[i], rng_keys[i], n_steps) for i in
                  range(self.n_chains)]
        results = self.pool.map(make_n_steps, inputs)
        for i in range(self.n_chains):
            self.states[i] = results[i][0]
            positions = results[i][1].position
            chain = Samples(positions)
            # Recompute the derived parameters since they couldn't be saved
            # during the sampling.
            # TODO: Understand why the list() command is necessary.
            derived = jax.vmap(lambda point: self.likelihood(
                point, return_derived=True)[1])(positions)
            derived.data = list(derived.data)
            derived.update(chain)
            chain['logposterior'] = results[i][1].logdensity
            self.chains[i] = Chain.concatenate(self.chains[i], chain)
            self.derived = Chain.concatenate([self.derived, derived])

    def __del__(self):
        """Unset the global `SAMPLER` variable."""
        global SAMPLER
        SAMPLER = None


class HMCSampler(BlackJAXSampler):
    """Wrapper for Hamiltonian Monte-Carlo (HMC)."""

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
        super().__init__(likelihood, n_chains, rng=rng, save_fn=save_fn,
                         mpicomm=mpicomm)

        if inv_mass_matrix is None:
            inv_mass_matrix = np.ones(self.n_dim)

        global SAMPLER
        SAMPLER = blackjax.hmc(
            compute_posterior, step_size, inv_mass_matrix,
            num_integration_steps, **kwargs)

        self.adaptation = adaptation
        self.adaptation_kwargs = adaptation_kwargs


class NUTSSampler(BlackJAXSampler):
    """Wrapper for No-U-Turn Sampler (NUTS)."""

    def __init__(self, likelihood, n_chains, adaptation=True,
                 adaptation_kwargs=dict(n_iters=1000), step_size=1e-3,
                 inv_mass_matrix=None, num_integration_steps=60, rng=None,
                 save_fn=None, mpicomm=None, **kwargs):
        """Initialize the No-U-Turn Sampler (NUTS).

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
        super().__init__(likelihood, n_chains, rng=rng, save_fn=save_fn,
                         mpicomm=mpicomm)

        if inv_mass_matrix is None:
            inv_mass_matrix = np.ones(self.n_dim)

        global SAMPLER
        SAMPLER = blackjax.nuts(
            compute_posterior, step_size, inv_mass_matrix, **kwargs)

        self.adaptation = adaptation
        self.adaptation_kwargs = adaptation_kwargs
