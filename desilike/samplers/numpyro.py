"""Module implementing NumPyro MCMC samplers."""

import numpy as np
import jax
import jax.numpy as jnp

try:
    import numpyro
    import numpyro.infer
    NUMPYRO_INSTALLED = True
except ModuleNotFoundError:
    NUMPYRO_INSTALLED = False

import inspect

from .base import MarkovChainSampler, _flat_to_dict, _param_sizes


class NumpyroSampler(MarkovChainSampler):
    """Base wrapper for ``NumPyro`` MCMC samplers.

    .. rubric:: References
    - https://github.com/pyro-ppl/numpyro

    """

    # Subclasses set these as class attributes before calling super().__init__().
    kernel_cls = None
    _extra_fields = ('accept_prob', 'potential_energy')

    def __init__(self, posterior, nchains=1, chains=None, rng=None,
                 directory=None, rescale=False, covariance=None):
        """Initialize the ``NumPyro`` sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        TypeError
            If called on this class directly.

        """
        if not NUMPYRO_INSTALLED:
            raise ImportError("The 'numpyro' package is required but not installed.")

        if type(self) is NumpyroSampler:
            raise TypeError("NumpyroSampler cannot be instantiated directly.")

        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)

        self._total_likelihood_evaluations = 0
        self._numpyro_position = None

        _vfn = jax.jit(jax.vmap(self._compute_posterior_one))

        def _compute_derived(batch):
            _, derived = _vfn(jnp.asarray(batch))
            return np.asarray(derived)

        self.compute_derived = self.pool.save_function(_compute_derived, 'compute_derived')
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        return getattr(numpyro.infer, self.kernel_cls)(
            potential_fn=self._potential_fn, **self.kernel_kwargs)

    def _potential_fn(self, params):
        """Return negative log-posterior (NumPyro minimizes the potential)."""
        return -self.posterior(self._forward_dict(params), return_derived=False)

    def _get_init_position(self):
        """Return current chain position as ``{name: rescaled_value}`` dict."""
        flat, _, _ = self.state
        return _flat_to_dict(jnp.array(flat), self.varied_params)

    def adapt_sampler(self, **kwargs):
        """Run NumPyro warmup and rebuild the kernel with adapted parameters.

        Parameters
        ----------
        steps : int
            Number of warmup steps.
        adapt_step_size : bool, optional
            Tune step size during warmup (if the kernel supports it). Default is ``True``.
        adapt_mass_matrix : bool, optional
            Tune mass matrix during warmup (if the kernel supports it). Default is ``True``.
        dense_mass : bool, optional
            Use a dense mass matrix (if the kernel supports it). Carried forward to the
            sampling kernel. Default is ``False`` (``True`` for :class:`NumpyroSASampler`).
        target_accept_prob : float, optional
            Target acceptance probability for dual averaging (if the kernel supports it).
        adapt_state_size : int or None, optional
            Number of past samples to adapt the mass matrix from (:class:`NumpyroSASampler`
            only). Carried forward to the sampling kernel.
        **kwargs : dict
            Extra keyword arguments forwarded to the warmup kernel constructor.

        """
        if self.pool.main:
            steps = kwargs.pop('steps')
            # Build warmup kernel with current sampling kwargs + all adaptation kwargs.
            warmup_kernel = getattr(numpyro.infer, self.kernel_cls)(
                potential_fn=self._potential_fn, **self.kernel_kwargs, **kwargs)
            init_position = self._numpyro_position or self._get_init_position()
            rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
            # num_samples=0 raises IndexError in numpyro ≤ 0.16; collect 1 and discard.
            warmup_mcmc = numpyro.infer.MCMC(
                warmup_kernel, num_warmup=steps, num_samples=1, progress_bar=False)
            warmup_mcmc.run(rng_key, init_params=init_position)
            self._numpyro_position = warmup_mcmc.last_state.z

            adapt_state = warmup_mcmc.last_state.adapt_state
            kernel_sig = inspect.signature(
                getattr(numpyro.infer, self.kernel_cls).__init__).parameters

            # Carry forward structure-defining kwargs (e.g. dense_mass, adapt_state_size)
            # but not warmup-only kwargs (adapt_step_size, adapt_mass_matrix, target_accept_prob).
            _warmup_only = {'adapt_step_size', 'adapt_mass_matrix', 'target_accept_prob'}
            for key, value in kwargs.items():
                if key not in _warmup_only and key in kernel_sig:
                    self.kernel_kwargs[key] = value

            # Override with adapted values and disable further adaptation.
            if hasattr(adapt_state, 'step_size') and 'step_size' in kernel_sig:
                self.kernel_kwargs['step_size'] = float(adapt_state.step_size)
                self.kernel_kwargs['adapt_step_size'] = False
            if hasattr(adapt_state, 'inverse_mass_matrix') and 'inverse_mass_matrix' in kernel_sig:
                self.kernel_kwargs['inverse_mass_matrix'] = np.asarray(adapt_state.inverse_mass_matrix)
                self.kernel_kwargs['adapt_mass_matrix'] = False
            self._kernel = self._make_kernel()

            self.logger.info('Adaptation done.')
            if 'step_size' in self.kernel_kwargs:
                self.logger.info('step_size: %.3g', self.kernel_kwargs['step_size'])
            if 'inverse_mass_matrix' in self.kernel_kwargs:
                imm = np.asarray(self.kernel_kwargs['inverse_mass_matrix'])
                if imm.ndim == 2:
                    eig = np.linalg.eigvalsh(imm)
                    self.logger.info(
                        'inverse_mass_matrix eigenvalues: min %.3g, max %.3g, cond %.3g, det^{1/n} %.3g'
                        % (eig.min(), eig.max(), eig.max() / eig.min(),
                           eig.prod() ** (1. / len(eig))))
                else:
                    imm_flat = imm.ravel()
                    self.logger.info(
                        'inverse_mass_matrix: min %.3g, max %.3g, det^{1/n} %.3g'
                        % (imm_flat.min(), imm_flat.max(),
                           imm_flat.prod() ** (1. / len(imm_flat))))
            self.pool.stop_wait()
        else:
            self.pool.wait()

    def run_sampler(self, n_steps):
        """Run the ``NumPyro`` sampler for ``n_steps`` samples.

        Parameters
        ----------
        n_steps : int
            Number of posterior samples to collect.

        """
        if self.pool.main:
            init_position = self._numpyro_position or self._get_init_position()
            rng_key = jax.random.PRNGKey(self.rng.integers(2**32))
            mcmc = numpyro.infer.MCMC(
                self._kernel, num_warmup=0, num_samples=n_steps, progress_bar=False)
            mcmc.run(rng_key, extra_fields=self._extra_fields, init_params=init_position)
            self._numpyro_position = mcmc.last_state.z

            samples_dict = mcmc.get_samples()
            extra = mcmc.get_extra_fields()

            # Build (n_steps, ndim) array in rescaled space.
            samples = np.column_stack([
                np.asarray(samples_dict[param.name]).reshape(n_steps, size)
                for param, size, col in _param_sizes(self.varied_params)
            ])
            log_post = -np.asarray(extra['potential_energy'])

            if len(self.derived_params):
                derived = np.array(self.pool.map(self.compute_derived, samples))
            else:
                derived = np.zeros((n_steps, 0))

            if 'num_steps' in extra:
                nsteps = np.asarray(extra['num_steps']).ravel()
                self._total_likelihood_evaluations += int(nsteps.sum())
                self.logger.info('number of integration steps: mean %.1f, max %d'
                                 % (nsteps.mean(), nsteps.max()))
            else:
                self._total_likelihood_evaluations += n_steps
            if 'accept_prob' in extra:
                self.logger.info('acceptance rate: mean %.3f'
                                 % float(np.asarray(extra['accept_prob']).mean()))
            if self._total_likelihood_evaluations:
                self.logger.info('total likelihood evaluations(~): %d',
                                 self._total_likelihood_evaluations)

            samples = samples.reshape((n_steps, -1))
            derived = derived.reshape((n_steps, -1))
            log_post = log_post.reshape(n_steps)
            self.extend(samples, derived, log_post)
            self.pool.stop_wait()
        else:
            self.pool.wait()


class NumpyroNUTSSampler(NumpyroSampler):
    """Wrapper for No-U-Turn Sampler (NUTS) via ``NumPyro``.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.NUTS

    """

    kernel_cls = 'NUTS'
    _extra_fields = ('accept_prob', 'potential_energy', 'num_steps')

    def __init__(self, posterior, nchains=1, chains=None,
                 step_size=1.0, inverse_mass_matrix=None,
                 max_tree_depth=10,
                 rng=None, directory=None, rescale=False, covariance=None, **kwargs):
        """Initialize the No-U-Turn Sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. Default is ``None``.
        step_size : float, optional
            Initial leapfrog step size. Default is 1.0.
        inverse_mass_matrix : numpy.ndarray, optional
            Initial inverse mass matrix.  1-D for diagonal, 2-D for dense.
            ``None`` uses unit diagonal. Default is ``None``.
        max_tree_depth : int, optional
            Maximum binary tree depth for NUTS. Default is 10.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        **kwargs : dict, optional
            Extra keyword arguments forwarded to ``numpyro.infer.NUTS``.

        Notes
        -----
        Adaptation parameters (``adapt_step_size``, ``adapt_mass_matrix``,
        ``dense_mass``, ``target_accept_prob``) are passed via
        ``run(adaptation=dict(steps=..., adapt_step_size=..., ...))``.

        """
        self.kernel_kwargs = dict(step_size=step_size, max_tree_depth=max_tree_depth, **kwargs)
        if inverse_mass_matrix is not None:
            self.kernel_kwargs['inverse_mass_matrix'] = inverse_mass_matrix
        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)


class NumpyroHMCSampler(NumpyroSampler):
    """Wrapper for Hamiltonian Monte Carlo (HMC) via ``NumPyro``.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.HMC

    """

    kernel_cls = 'HMC'
    _extra_fields = ('accept_prob', 'potential_energy', 'num_steps')

    def __init__(self, posterior, nchains=1, chains=None,
                 step_size=1.0, inverse_mass_matrix=None,
                 num_steps=None, trajectory_length=None,
                 rng=None, directory=None, rescale=False, covariance=None, **kwargs):
        """Initialize the Hamiltonian Monte Carlo sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. Default is ``None``.
        step_size : float, optional
            Initial leapfrog step size. Default is 1.0.
        inverse_mass_matrix : numpy.ndarray, optional
            Initial inverse mass matrix.  ``None`` uses unit diagonal. Default is ``None``.
        num_steps : int or None, optional
            Fixed number of leapfrog steps per proposal.  ``None`` derives it from
            ``trajectory_length / step_size``. Default is ``None``.
        trajectory_length : float or None, optional
            Total trajectory length.  ``None`` uses the NumPyro default (2π). Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        **kwargs : dict, optional
            Extra keyword arguments forwarded to ``numpyro.infer.HMC``.

        Notes
        -----
        Adaptation parameters (``adapt_step_size``, ``adapt_mass_matrix``,
        ``dense_mass``, ``target_accept_prob``) are passed via
        ``run(adaptation=dict(steps=..., adapt_step_size=..., ...))``.

        """
        self.kernel_kwargs = dict(step_size=step_size, **kwargs)
        if inverse_mass_matrix is not None:
            self.kernel_kwargs['inverse_mass_matrix'] = inverse_mass_matrix
        if num_steps is not None:
            self.kernel_kwargs['num_steps'] = num_steps
        if trajectory_length is not None:
            self.kernel_kwargs['trajectory_length'] = trajectory_length
        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)


class NumpyroBarkerMHSampler(NumpyroSampler):
    """Wrapper for the Barker Metropolis-Hastings sampler via ``NumPyro``.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.barker.BarkerMH

    """

    kernel_cls = 'BarkerMH'
    _extra_fields = ('accept_prob', 'potential_energy')

    def __init__(self, posterior, nchains=1, chains=None,
                 step_size=1.0,
                 rng=None, directory=None, rescale=False, covariance=None, **kwargs):
        """Initialize the Barker MH sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. Default is ``None``.
        step_size : float, optional
            Initial step size. Default is 1.0.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        **kwargs : dict, optional
            Extra keyword arguments forwarded to ``numpyro.infer.BarkerMH``.

        Notes
        -----
        Adaptation parameters (``adapt_step_size``, ``adapt_mass_matrix``,
        ``dense_mass``, ``target_accept_prob``) are passed via
        ``run(adaptation=dict(steps=..., adapt_step_size=..., ...))``.
        Note that ``target_accept_prob`` defaults to 0.4 for ``BarkerMH``.

        """
        self.kernel_kwargs = dict(step_size=step_size, **kwargs)
        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)


class NumpyroSASampler(NumpyroSampler):
    """Wrapper for the Sample Adaptive (SA) MCMC sampler via ``NumPyro``.

    SA is a gradient-free, sample-adaptive method that builds a mass matrix
    from past samples.  Adaptation happens implicitly during warmup steps.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.sa.SA

    """

    kernel_cls = 'SA'
    _extra_fields = ('accept_prob', 'potential_energy')

    def __init__(self, posterior, nchains=1, chains=None,
                 rng=None, directory=None, rescale=False, covariance=None, **kwargs):
        """Initialize the SA sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        **kwargs : dict, optional
            Extra keyword arguments forwarded to ``numpyro.infer.SA``.

        Notes
        -----
        Adaptation parameters (``dense_mass``, ``adapt_state_size``) are passed via
        ``run(adaptation=dict(steps=..., dense_mass=..., adapt_state_size=...))``.
        ``dense_mass`` defaults to ``True`` for SA.

        """
        self.kernel_kwargs = dict(**kwargs)
        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, rescale=rescale, covariance=covariance)
