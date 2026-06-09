"""Module implementing the BlackJAX samplers."""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
try:
    import blackjax
    BLACKJAX_INSTALLED = True
except ModuleNotFoundError:
    BLACKJAX_INSTALLED = False

from .base import MarkovChainSampler, _flat_to_dict


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
        (state, info) : tuple
            State and kernel info, accumulated by `jax.lax.scan`.

        """
        state, info = step(rng_key, state)
        return state, (state, info)

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
        (states, infos) : tuple
            All sampled states and per-step kernel info.

        """
        state, rng_keys = args
        return jax.lax.scan(make_one_step, state, rng_keys)

    return jax.jit(make_steps)


class BlackJAXSampler(MarkovChainSampler):
    """Wrapper for ``BlackJAX`` samplers.

    .. rubric:: References
    - https://github.com/blackjax-devs/blackjax

    """

    # Gradient-based kernels need their step size (and mass matrix / L) tuned before
    # sampling: window_adaptation for HMC/NUTS, mclmc_find_L_and_step_size for MCLMC.
    # Without it, the default step_size barely moves the chain (especially fixed-path
    # HMC).  Override via ``run(adaptation_steps=...)``; set 0 to disable.
    default_adaptation_steps = 500

    def __init__(self, posterior, nchains=1, chains=None, rng=None,
                 directory=None, rescale=False, covariance=None):
        """Initialize the ``BlackJAX`` sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        TypeError
            If called by this class.

        """
        if not BLACKJAX_INSTALLED:
            raise ImportError("The 'blackjax' package is required but not "
                              "installed.")

        if type(self) is BlackJAXSampler:
            raise TypeError("BlackJAXSampler cannot be iniated directly.")

        super().__init__(posterior, nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)

        _vfn = jax.jit(jax.vmap(self._compute_posterior_one))

        def _compute_derived(batch):
            _, derived = _vfn(jnp.asarray(batch))
            return np.asarray(derived)

        self.compute_derived = self.pool.save_function(_compute_derived, 'compute_derived')
        self.kernel_type = getattr(blackjax, self.kernel_type)
        self.kernel = self.kernel_type(self.compute_posterior_without_derived, **self.kernel_args)
        self.adaptation_fn = getattr(blackjax, self.adaptation_fn)
        self.make_steps = make_steps_factory(self.kernel.step)

    def compute_posterior_without_derived(self, sample):
        """Compute the natural logarithm of the posterior.

        Parameters
        ----------
        sample : dict
            Sample (in the sampler's rescaled working space) for which to compute
            the posterior; mapped back to original parameter values via
            :meth:`~desilike.samplers.base.BaseSampler._forward_dict`.

        Returns
        -------
        log_post : float
            Natural logarithm of the posterior.

        """
        return self.posterior(self._forward_dict(sample), return_derived=False)

    def run_sampler(self, n_steps):
        """Run the ``BlackJAX`` sampler.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.

        """
        if self.pool.main:
            if not hasattr(self, 'blackjax_state'):
                initial_position = dict(zip(self.varied_params.names(), self.state[0]))
                try:
                    self.blackjax_state = self.kernel.init(initial_position)
                except TypeError:
                    rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
                    self.blackjax_state = self.kernel.init(initial_position, rng_key)

            rng_keys = jax.random.split(jax.random.PRNGKey(
                self.rng.integers(2**32)), self.nchains)
            rng_key = rng_keys[self._ichain]

            # Make the steps
            inputs = (self.blackjax_state, jax.random.split(rng_key, n_steps))
            self.blackjax_state, (all_states, last_info) = self.make_steps(inputs)

            # Log last-step diagnostics when available (e.g. NUTS/HMC).
            parts = []
            if hasattr(last_info, 'num_integration_steps'):
                parts.append('num_integration_steps: %d' % int(np.asarray(last_info.num_integration_steps).ravel()[-1]))
            if hasattr(last_info, 'acceptance_rate'):
                parts.append('acceptance_rate: %.3f' % float(np.asarray(last_info.acceptance_rate).ravel()[-1]))
            if parts:
                self.logger.info(', '.join(parts))

            # Update the chains.
            samples = np.column_stack([all_states.position[key] for key in self.varied_params.names()])
            log_post = all_states.logdensity

            if len(self.derived_params):
                derived = np.array(self.pool.map(self.compute_derived, samples))
            else:
                derived = np.zeros((n_steps, 0))

            samples = samples.reshape((n_steps, -1))
            derived = derived.reshape((n_steps, -1))
            log_post = log_post.reshape(n_steps)
            self.extend(samples, derived, log_post)
            self.pool.stop_wait()
        else:
            self.pool.wait()

    def adapt_sampler(self, steps):
        """Adapt the step size and mass matrix.

        Parameters
        ----------
        steps : int
            How steps to run for the adaptation.

        """
        if self.pool.main:
            fixed_kernel_args = {
                key: value for key, value in self.kernel_args.items() if key not in
                self.adaptable_args}
            initial_position = _flat_to_dict(self.state[0], self.varied_params)
            rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
            # blackjax's window_adaptation takes the non-adaptable kernel parameters
            # (e.g. num_integration_steps) in its constructor, not in run().
            (state, parameters), _ = self.adaptation_fn(
                self.kernel_type, self.compute_posterior_without_derived, **fixed_kernel_args).run(
                rng_key, initial_position, num_steps=steps)
            self.kernel_args.update(parameters)
            # Rebuild the kernel so the tuned parameters (step size, mass matrix, ...) are
            # actually used during sampling, and warm-start from the adapted state.
            self.kernel = self.kernel_type(self.compute_posterior_without_derived, **self.kernel_args)
            self.make_steps = make_steps_factory(self.kernel.step)
            self.blackjax_state = state
            self.pool.stop_wait()
        else:
            self.pool.wait()
        if self.mpicomm.rank == 0:
            self.logger.info('Adaptation done.')


class HMCSampler(BlackJAXSampler):
    """Wrapper for Hamiltonian Monte-Carlo (HMC)."""

    kernel_type = 'hmc'
    adaptable_args = ['step_size', 'inverse_mass_matrix']
    adaptation_fn = 'window_adaptation'

    def __init__(self, posterior, nchains=1, chains=None, step_size=1e-3,
                 inverse_mass_matrix=None, num_integration_steps=60, rng=None,
                 directory=None, rescale=False, covariance=None, **kwargs):
        """Initialize the HMC sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
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
        **kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.hmc`` during
            initialization.

        """
        if inverse_mass_matrix is None:
            ndim = len(posterior.params.select(varied=True, solved=False))
            inverse_mass_matrix = np.ones(ndim)

        self.kernel_args = dict(
            step_size=step_size, inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=num_integration_steps, **kwargs)

        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)


class NoUTurnSampler(BlackJAXSampler):
    """Wrapper for No-U-Turn Sampler (NUTS)."""

    kernel_type = 'nuts'
    adaptable_args = ['step_size', 'inverse_mass_matrix']
    adaptation_fn = 'window_adaptation'

    def __init__(self, posterior, nchains=1, chains=None, step_size=1e-3,
                 inverse_mass_matrix=None, rng=None, directory=None,
                 rescale=False, covariance=None, **kwargs):
        """Initialize the No-U-Turn Sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
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
        **kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.nuts`` during
            initialization.

        """
        if inverse_mass_matrix is None:
            ndim = len(posterior.params.select(varied=True, solved=False))
            inverse_mass_matrix = np.ones(ndim)

        self.kernel_args = dict(
            step_size=step_size, inverse_mass_matrix=inverse_mass_matrix,
            **kwargs)

        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)


class MCLMCSampler(BlackJAXSampler):
    """Wrapper for the Microcanonical Langevin Monte Carlo (MCLMC) sampler.

    .. rubric:: References
    - https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    - https://arxiv.org/abs/2212.08549

    """

    kernel_type = 'mclmc'
    adaptable_args = ['L', 'step_size']
    adaptation_fn = 'mclmc_find_L_and_step_size'
    # MCLMC's adaptation (mclmc_find_L_and_step_size) uses a different API than the
    # window_adaptation-based adapt_sampler, so it is not run by default here.
    default_adaptation_steps = 0

    def __init__(self, posterior, nchains=1, chains=None, L=1., step_size=0.1, rng=None,
                 directory=None, rescale=False, covariance=None, **kwargs):
        """Initialize the Microcanonical Langevin Monte Carlo (MCLMC) sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        L : float, default=1.
            Momentum decoherence scale.
        step_size : float, default=0.1
            The value to use for the step size in the integrator.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``blackjax.mclmc`` during
            initialization.

        """
        self.kernel_args = dict(L=L, step_size=step_size, **kwargs)

        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)
