"""Module implementing the BlackJAX samplers."""

from functools import partial

try:
    import jax
    import blackjax
    BLACKJAX_INSTALLED = True
except ModuleNotFoundError:
    BLACKJAX_INSTALLED = False
import numpy as np

from .base import MarkovChainSampler


# TODO: Properly implement abstract classes and methods.

def make_n_steps_factory(sampler):
    """Produce a JIT compiled version of the `make_n_steps` function.

    Parameters
    ----------
    sampler : object
        The BlackJAX sampler.

    Returns
    -------
    The `make_n_steps` function.
    """

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
        state, _ = sampler.step(rng_key, state)
        return state, state

    def make_n_steps(args):
        """Advance the state by several steps.

        Parameters
        ----------
        args : tuple
            Blackjax state and random keys. Each random key is used for one
            step.

        Returns
        -------
        final_state : NamedTuple
            Final state after all steps.
        states : NamedTuple
            All sampled states.

        """
        state, rng_keys = args
        return jax.lax.scan(make_one_step, state, rng_keys)

    return jax.jit(make_n_steps)


class BlackJAXSampler(MarkovChainSampler):
    """Wrapper for BlackJAX samplers.

    .. rubric:: References
    - https://github.com/blackjax-devs/blackjax

    """

    def __init__(self, likelihood, n_chains=4, rng=None, directory=None,
                 **kwargs):
        """Initialize the ``BlackJAX`` sampler.

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
            Extra keyword arguments passed to ``blackjax`` during
            initialization.

        """
        if not BLACKJAX_INSTALLED:
            raise ImportError("The 'blackjax' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains, rng=rng, directory=directory)
        self.compute_posterior = self.pool.save_function(
            partial(self.compute_posterior, save_derived=False),
            'compute_posterior')
        self.states = None

    def run_sampler(self, n_steps):
        """Run the ``BlackJAX`` sampler.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.

        """
        if self.states is None:
            self.states = []
            for i in range(self.n_chains):
                initial_position = dict(zip(self.params.keys()[:self.n_dim],
                                            self.chains[i][-1]))
                try:
                    self.states.append(self.sampler.init(initial_position))
                except TypeError:
                    rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
                    self.states.append(self.sampler.init(initial_position,
                                                         rng_key))

        rng_keys = jax.random.split(jax.random.PRNGKey(
            self.rng.integers(2**32)), self.n_chains)

        inputs = [(self.states[i], jax.random.split(rng_keys[i], n_steps)) for
                  i in range(self.n_chains)]
        results = self.pool.map(self.make_n_steps, inputs)
        self.states = [r[0] for r in results]
        chains = np.stack([np.column_stack([result[1].position[key] for key in
                                            self.params.keys()[:self.n_dim]])
                           for result in results])
        log_post = np.stack([result[1].logdensity for result in results])
        self.chains = np.concatenate([self.chains, chains], axis=1)
        self.log_post = np.concatenate([self.log_post, log_post], axis=1)

        for i in range(self.n_chains):
            # Recompute the derived parameters since they couldn't be saved
            # during the sampling.
            samples = results[i][1].position
            derived = jax.vmap(lambda sample: self.likelihood(
                sample, return_derived=True)[1])(samples)
            for i, key in enumerate(self.params.keys()):
                if i < self.n_dim:
                    self.derived[key].append(samples[key])
                else:
                    self.derived[key].append(derived[key])


class HMCSampler(BlackJAXSampler):
    """Wrapper for Hamiltonian Monte-Carlo (HMC)."""

    def __init__(self, likelihood, n_chains=4, adaptation=True,
                 adaptation_kwargs=dict(n_iters=1000), step_size=1e-3,
                 inv_mass_matrix=None, num_integration_steps=60, rng=None,
                 directory=None, **kwargs):
        """Initialize the HMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
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
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.hmc`` during
            initialization.

        """
        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)

        if inv_mass_matrix is None:
            inv_mass_matrix = np.ones(self.n_dim)

        self.sampler = blackjax.hmc(
            self.compute_posterior, step_size, inv_mass_matrix,
            num_integration_steps, **kwargs)
        self.make_n_steps = self.pool.save_function(
            make_n_steps_factory(self.sampler), "make_n_steps")

        self.adaptation = adaptation
        self.adaptation_kwargs = adaptation_kwargs


class NUTSSampler(BlackJAXSampler):
    """Wrapper for No-U-Turn Sampler (NUTS)."""

    def __init__(self, likelihood, n_chains=4, adaptation=True,
                 adaptation_kwargs=dict(n_iters=1000), step_size=1e-3,
                 inv_mass_matrix=None, rng=None, directory=None, **kwargs):
        """Initialize the No-U-Turn Sampler (NUTS).

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
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
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.hmc`` during
            initialization.

        """
        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)

        if inv_mass_matrix is None:
            inv_mass_matrix = np.ones(self.n_dim)

        self.sampler = blackjax.nuts(
            self.compute_posterior, step_size, inv_mass_matrix, **kwargs)
        self.make_n_steps = self.pool.save_function(
            make_n_steps_factory(self.sampler), "make_n_steps")

        self.adaptation = adaptation
        self.adaptation_kwargs = adaptation_kwargs


class MCLMCSampler(BlackJAXSampler):
    """Wrapper for the Microcanonical Langevin Monte Carlo (MCLMC) sampler.

    .. rubric:: References
    - https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    - https://arxiv.org/abs/2212.08549
    """

    def __init__(self, likelihood, n_chains=4, adaptation=True, L=1.,
                 step_size=0.1, integrator='isokinetic_mclachlan', rng=None,
                 directory=None, **kwargs):
        """Initialize the Microcanonical Langevin Monte Carlo (MCLMC) sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        adaptation : bool, dict, default=True
            Adapt momentum decoherence scale ``L`` and ``step_size``.
            Can be ``{'niterations': 1000, 'frac_tune1': 0.1, 'frac_tune1': 0.1, 'frac_tune2': 0.1, 'frac_tune3': 0.1,
            'desired_energy_var': 5e-4,, 'trust_in_estimate': 1.5, 'num_effective_samples': 150, 'diagonal_preconditioning': True}``
        L : float, default=1.
            Momentum decoherence scale.
        step_size : float, default=0.1
            The value to use for the step size in the integrator.
        integrator : str, default='isokinetic_mclachlan'
            Integrator, from :mod:`blackjax.mcmc.integrators`.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.hmc`` during
            initialization.

        """
        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)

        self.sampler = blackjax.mclmc(self.compute_posterior, L, step_size)
        self.make_n_steps = self.pool.save_function(
            make_n_steps_factory(self.sampler), "make_n_steps")

        self.adaptation = adaptation
