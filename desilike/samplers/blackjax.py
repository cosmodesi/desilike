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

def make_steps_factory(step):
    """Produce a JIT compiled version of the `make_steps` function.

    Parameters
    ----------
    step : function
        The BlackJAX kernel step function.

    Returns
    -------
    The `make_steps` function.

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
        state, _ = step(rng_key, state)
        return state, state

    def make_steps(args):
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

    return jax.jit(make_steps)


class BlackJAXSampler(MarkovChainSampler):
    """Wrapper for ``BlackJAX`` samplers.

    .. rubric:: References
    - https://github.com/blackjax-devs/blackjax

    """

    def __init__(self, likelihood, n_chains=4, rng=None, directory=None):
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

        """
        if not BLACKJAX_INSTALLED:
            raise ImportError("The 'blackjax' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains, rng=rng, directory=directory)
        self.compute_posterior = self.pool.save_function(
            partial(self.compute_posterior, save_derived=False),
            'compute_posterior')

        self.kernel_type = getattr(blackjax, self.kernel_type)
        self.kernel = self.kernel_type(
            self.compute_posterior, **self.kernel_args)
        self.adaptation_fn = getattr(blackjax, self.adaptation_fn)
        self.make_steps = self.pool.save_function(
            make_steps_factory(self.kernel.step), 'make_steps')

        self.states = None

    def run_sampler(self, steps):
        """Run the ``BlackJAX`` sampler.

        Parameters
        ----------
        steps : int
            Number of steps to take.

        """
        if self.states is None:
            self.states = []
            for i in range(self.n_chains):
                initial_position = dict(zip(self.params.keys()[:self.n_dim],
                                            self.chains[i][-1]))
                try:
                    self.states.append(self.kernel.init(initial_position))
                except TypeError:
                    rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
                    self.states.append(self.kernel.init(initial_position,
                                                        rng_key))

        rng_keys = jax.random.split(jax.random.PRNGKey(
            self.rng.integers(2**32)), self.n_chains)

        inputs = [(self.states[i], jax.random.split(rng_keys[i], steps)) for
                  i in range(self.n_chains)]
        results = self.pool.map(self.make_steps, inputs)
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

    def adapt_sampler(self, steps):
        """Adapt the step size and mass matrix.

        Parameters
        ----------
        steps : int
            How steps to run for the adaptation.

        """
        fixed_kernel_args = {
            key: value for key, value in self.kernel_args.items() if key not in
            self.adaptable_args}
        initial_position = dict(zip(self.params.keys()[:self.n_dim],
                                    self.chains[0][-1]))
        rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
        (state, parameters), _ = self.adaptation_fn(
            self.kernel_type, self.compute_posterior).run(
            rng_key, initial_position, num_steps=steps, **fixed_kernel_args)
        self.kernel_args.update(parameters)


class HMCSampler(BlackJAXSampler):
    """Wrapper for Hamiltonian Monte-Carlo (HMC)."""

    kernel_type = 'hmc'
    adaptable_args = ['step_size', 'inverse_mass_matrix']
    adaptation_fn = 'window_adaptation'

    def __init__(self, likelihood, n_chains=4, step_size=1e-3,
                 inverse_mass_matrix=None, num_integration_steps=60, rng=None,
                 directory=None, **kwargs):
        """Initialize the HMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        step_size : float, optional
            Size of the integration step. Default is 1e-3.
        inverse_mass_matrix : numpy.ndarray, optional
            The value to use for the inverse mass matrix when drawing a value
            for the momentum and computing the kinetic energy. If
            one-dimensional, a diagonal mass matrix is assumed. If ``None``,
            a unity matrix is used. Default is ``None``.
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
        if inverse_mass_matrix is None:
            inverse_mass_matrix = np.ones(len(likelihood.varied_params))

        self.kernel_args = dict(
            step_size=step_size, inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=num_integration_steps, **kwargs)

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)


class NUTSSampler(BlackJAXSampler):
    """Wrapper for No-U-Turn Sampler (NUTS)."""

    kernel_type = 'nuts'
    adaptable_args = ['step_size', 'inverse_mass_matrix']
    adaptation_fn = 'window_adaptation'

    def __init__(self, likelihood, n_chains=4, step_size=1e-3,
                 inverse_mass_matrix=None, rng=None, directory=None, **kwargs):
        """Initialize the No-U-Turn Sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        step_size : float, optional
            Size of the integration step. Default is 1e-3.
        inverse_mass_matrix : numpy.ndarray, optional
            The value to use for the inverse mass matrix when drawing a value
            for the momentum and computing the kinetic energy. If
            one-dimensional, a diagonal mass matrix is assumed. If ``None``,
            a unity matrix is used. Default is ``None``.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.nuts`` during
            initialization.

        """
        if inverse_mass_matrix is None:
            inverse_mass_matrix = np.ones(len(likelihood.varied_params))

        self.kernel_args = dict(
            step_size=step_size, inverse_mass_matrix=inverse_mass_matrix,
            **kwargs)

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)


class MCLMCSampler(BlackJAXSampler):
    """Wrapper for the Microcanonical Langevin Monte Carlo (MCLMC) sampler.

    .. rubric:: References
    - https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    - https://arxiv.org/abs/2212.08549

    """

    kernel_type = 'mclmc'
    adaptable_args = ['L', 'step_size']
    adaptation_fn = 'mclmc_find_L_and_step_size'

    def __init__(self, likelihood, n_chains=4, L=1., step_size=0.1, rng=None,
                 directory=None, **kwargs):
        """Initialize the Microcanonical Langevin Monte Carlo (MCLMC) sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        L : float, default=1.
            Momentum decoherence scale.
        step_size : float, default=0.1
            The value to use for the step size in the integrator.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.mclmc`` during
            initialization.

        """
        self.kernel_args = dict(L=L, step_size=step_size, **kwargs)

        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)
