"""NumPyro NUTS, HMC, BarkerMH, SA, AIES, and ESS kernels."""

import logging
import inspect

import numpy as np
import jax
import jax.numpy as jnp

try:
    import numpyro
    import numpyro.infer
    NUMPYRO_INSTALLED = True
except ModuleNotFoundError:
    NUMPYRO_INSTALLED = False

from .base import Kernel


def _log_adaptation(logger, kernel_kwargs):
    if 'step_size' in kernel_kwargs:
        logger.info('step_size: %.3g', kernel_kwargs['step_size'])
    if 'inverse_mass_matrix' in kernel_kwargs:
        imm = np.asarray(kernel_kwargs['inverse_mass_matrix'])
        if imm.ndim == 2:
            eig = np.linalg.eigvalsh(imm)
            logger.info('inverse_mass_matrix eigenvalues: min %.3g, max %.3g, cond %.3g, det^{1/n} %.3g',
                        eig.min(), eig.max(), eig.max() / eig.min(),
                        eig.prod() ** (1. / len(eig)))
        else:
            imm_flat = imm.ravel()
            logger.info('inverse_mass_matrix: min %.3g, max %.3g, det^{1/n} %.3g',
                        imm_flat.min(), imm_flat.max(),
                        imm_flat.prod() ** (1. / len(imm_flat)))


class _NumpyroKernel(Kernel):
    """Common base for NumPyro MCMC kernels."""

    logger = logging.getLogger('NumpyroKernel')
    _numpyro_cls = None
    _extra_fields = ('accept_prob', 'potential_energy')
    max_nparallel = None  # numpyro handles any number of chains via num_chains
    # Whether the kernel can be resumed from post_warmup_state instead of being handed
    # init_params on every call; see run().
    _supports_continuation = True

    @classmethod
    def install(cls, installer):
        installer.pip('numpyro')

    def init(self, posterior, rng, **context):
        if not NUMPYRO_INSTALLED:
            raise ImportError("The 'numpyro' package is required but not installed.")

        posterior_logpdf, _ = posterior

        self._rng = rng
        self._ndim = context['ndim']
        self._nsamples_parallel = context.get('nsamples_parallel', 1)

        import jax.numpy as _jnp
        def potential_fn(flat):
            return -posterior_logpdf(_jnp.asarray(flat)[None])[0]
        self._potential_fn = potential_fn

        self._numpyro_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=potential_fn, **self.kernel_kwargs)

        self._current_position = None
        self._total_likelihood_evaluations = 0
        # Persistent MCMC object, continued across run() calls -- see run() for why.
        self._mcmc = None
        self._mcmc_nsteps = 0
        self._mcmc_started = False

    def adapt(self, state, **kwargs):
        """Run NumPyro warmup and rebuild the kernel with adapted parameters.

        Adapts ``nsamples_parallel`` vectorized chains, reducing their per-chain step sizes
        and mass matrices to the single values the sampling kernel takes.  The warmed-up
        positions carry straight into :meth:`run`, so the chains start dispersed.

        Parameters
        ----------
        state : tuple
            ``(position, derived, logposterior)`` in rescaled space.
            For multi-chain state ``position`` has shape ``(nchains, ndim)``; a single
            position is broadcast across chains.
        steps : int
            Number of warmup steps.
        adapt_step_size : bool, optional
        adapt_mass_matrix : bool, optional
        dense_mass : bool, optional
        target_accept_prob : float, optional
        adapt_state_size : int or None, optional
        **kwargs
            Extra keyword arguments forwarded to the warmup kernel constructor.
        """
        position, _, _ = state
        # Warm up the same number of vectorized chains as sampling will use. Adapting a single
        # chain evaluates every gradient at batch size 1, where the GPU is essentially idle;
        # it also bases the step size and mass matrix on one trajectory, and leaves every chain
        # starting from that same point, so the early Gelman-Rubin reads as converged before the
        # chains have had time to separate.
        position = np.asarray(position)
        # Kernels that cannot be continued also reject chain_method='vectorized' (BarkerMH), so
        # they keep the single-chain warmup.
        nchains = self._nsamples_parallel if self._supports_continuation else 1
        if nchains > 1:
            if position.ndim == 1:
                position = np.broadcast_to(position, (nchains,) + position.shape)
            elif position.shape[0] != nchains:
                position = np.broadcast_to(position[:1], (nchains,) + position.shape[1:])
        elif position.ndim > 1:
            # Single chain: numpyro expects a plain (ndim,) vector, not (1, ndim).
            position = position[0]
        if self._current_position is None:
            self._current_position = position
        steps = kwargs.pop('steps')

        kernel_sig = inspect.signature(
            getattr(numpyro.infer, self._numpyro_cls).__init__).parameters

        # Kernels that adapt a mass matrix but take no inverse_mass_matrix (e.g. BarkerMH) sample
        # with an identity metric, so they must not adapt one either: the adapted step size below
        # is tuned in the whitened geometry, while sampling would then run unwhitened, and the
        # mismatch collapses the acceptance rate to zero. Kernels with neither knob (e.g. SA,
        # whose dense_mass drives its own proposal covariance) are left untouched.
        if 'adapt_mass_matrix' in kernel_sig and 'inverse_mass_matrix' not in kernel_sig:
            kwargs['adapt_mass_matrix'] = False
            kwargs.pop('dense_mass', None)

        # NumPyro adapts the mass matrix per chain, from that chain's own `steps` samples: its
        # Welford accumulator runs under vmap and never pools across chains. With `pool_mass_matrix`
        # the warmup samples of all chains are pooled instead, giving nchains * steps/2 samples to
        # estimate ndim * (ndim + 1) / 2 entries rather than steps per chain.
        pool_mass_matrix = kwargs.pop('pool_mass_matrix', False) and nchains > 1

        warmup_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=self._potential_fn, **self.kernel_kwargs, **kwargs)
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
        warmup_mcmc = numpyro.infer.MCMC(
            warmup_kernel, num_warmup=steps, num_samples=1, progress_bar=False,
            **(dict(num_chains=nchains, chain_method='vectorized') if nchains > 1 else {}))
        pooled_covariance = None
        if pool_mass_matrix:
            warmup_mcmc.warmup(rng_key, init_params=self._current_position, collect_warmup=True)
            warmup_samples = np.asarray(warmup_mcmc.get_samples(group_by_chain=True))
            # Second half only: the first is transient, as in NumPyro's own adaptation windows.
            warmup_samples = warmup_samples[:, warmup_samples.shape[1] // 2:]
            pooled = warmup_samples.reshape(-1, warmup_samples.shape[-1])
            npooled = pooled.shape[0]
            pooled_covariance = np.cov(pooled.T)
            # Same regularization as NumPyro's welford_covariance final_fn: shrink toward the
            # identity by a factor set by the sample count.
            pooled_covariance = ((npooled / (npooled + 5.)) * pooled_covariance
                                 + 1e-3 * (5. / (npooled + 5.)) * np.eye(pooled_covariance.shape[0]))
            self.logger.info('pooled mass matrix from %d warmup samples (%d chains x %d steps)',
                             npooled, nchains, warmup_samples.shape[1])
        else:
            warmup_mcmc.run(rng_key, init_params=self._current_position)
        self._current_position = warmup_mcmc.last_state.z

        adapt_state = warmup_mcmc.last_state.adapt_state

        _warmup_only = {'adapt_step_size', 'adapt_mass_matrix', 'target_accept_prob'}
        for key, value in kwargs.items():
            if key not in _warmup_only and key in kernel_sig:
                self.kernel_kwargs[key] = value

        # Vectorized warmup adapts one step size and one mass matrix per chain; reduce them to
        # the single values the sampling kernel takes. The median step size ignores a chain that
        # adapted badly, while the mass matrices are averaged.
        if hasattr(adapt_state, 'step_size') and 'step_size' in kernel_sig:
            self.kernel_kwargs['step_size'] = float(np.median(np.asarray(adapt_state.step_size)))
            self.kernel_kwargs['adapt_step_size'] = False
        if hasattr(adapt_state, 'inverse_mass_matrix') and 'inverse_mass_matrix' in kernel_sig:
            if pooled_covariance is not None:
                inverse_mass_matrix = pooled_covariance
            else:
                inverse_mass_matrix = np.asarray(adapt_state.inverse_mass_matrix)
                if nchains > 1 and inverse_mass_matrix.shape[:1] == (nchains,):
                    inverse_mass_matrix = inverse_mass_matrix.mean(axis=0)
            self.kernel_kwargs['inverse_mass_matrix'] = inverse_mass_matrix
            self.kernel_kwargs['adapt_mass_matrix'] = False

        self._numpyro_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=self._potential_fn, **self.kernel_kwargs)
        # The kernel changed, so the compiled MCMC built around the previous one is stale.
        self._mcmc, self._mcmc_nsteps, self._mcmc_started = None, 0, False

        self.logger.info('Adaptation done.')
        _log_adaptation(self.logger, self.kernel_kwargs)

    def run(self, n_steps, state):
        position, _, _ = state
        # For multi-chain: position shape is (nchains, ndim).
        # On first call, seed from the provided state; afterward keep last_state.z.
        if self._current_position is None:
            self._current_position = position
        elif self._nsamples_parallel > 1 and np.asarray(self._current_position).ndim == 1:
            # Adapt left a single-chain position; tile it for the first parallel run.
            self._current_position = np.broadcast_to(
                self._current_position, (self._nsamples_parallel,) + np.asarray(self._current_position).shape)

        # One persistent MCMC, continued through post_warmup_state. Two constraints shape this:
        #  - NumPyro caches compiled functions on the kernel instance, so wrapping the same kernel
        #    in a new MCMC raises "vmap ... rank should be at least 1" for vectorized chains --
        #    a new MCMC therefore needs a new kernel;
        #  - num_samples is baked into the compiled loop, so changing it recompiles.
        # Sampler.run asks for min(check_every - steps % check_every, ...) steps, i.e. a constant
        # check_every after a first partial batch (steps starts at 1) -- so the sequence is
        # 299, 300, 300, ... A *smaller* batch (the last one of a max_steps-limited run) is run at
        # the compiled length and truncated; only a *larger* one rebuilds, which in practice
        # happens once, at that 299 -> 300 step.
        # Size the compiled loop by the sampler's nominal batch length when it provides one, so
        # the short first batch does not compile at its own length and force a rebuild as soon as
        # full-length batches begin. The extra samples are truncated below.
        if not self._supports_continuation:
            # Kernels that must be handed init_params on every call (BarkerMH with a potential_fn)
            # cannot be resumed from post_warmup_state; they restart from the current position
            # instead. The MCMC object is still kept, sized by the sampler's nominal batch length
            # so a shorter batch truncates rather than rebuilding.
            compile_steps = max(n_steps, getattr(self, 'nsteps_hint', 0))
            if self._mcmc is None or compile_steps > self._mcmc_nsteps:
                self._mcmc = numpyro.infer.MCMC(
                    self._numpyro_kernel,
                    num_warmup=0,
                    num_samples=compile_steps,
                    num_chains=self._nsamples_parallel,
                    chain_method='vectorized',
                    progress_bar=False)
                self._mcmc_nsteps = compile_steps
            run_steps = self._mcmc_nsteps
            rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
            self._mcmc.run(rng_key, extra_fields=self._extra_fields,
                           init_params=self._current_position)
            self._current_position = self._mcmc.last_state.z
            return self._collect(self._mcmc, n_steps, run_steps)

        run_steps = n_steps
        compile_steps = max(n_steps, getattr(self, 'nsteps_hint', 0))
        if self._mcmc is None or compile_steps > self._mcmc_nsteps:
            self._numpyro_kernel = getattr(numpyro.infer, self._numpyro_cls)(
                potential_fn=self._potential_fn, **self.kernel_kwargs)
            self._mcmc = numpyro.infer.MCMC(
                self._numpyro_kernel,
                num_warmup=0,
                num_samples=compile_steps,
                num_chains=self._nsamples_parallel,
                chain_method='vectorized',
                progress_bar=False)
            self._mcmc_nsteps, self._mcmc_started = compile_steps, False
        run_steps = self._mcmc_nsteps

        mcmc = self._mcmc
        if self._mcmc_started:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=self._extra_fields)
        else:
            rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
            mcmc.run(rng_key, extra_fields=self._extra_fields,
                     init_params=self._current_position)
            self._mcmc_started = True
        # The chain state advances by run_steps even when only n_steps are kept: the discarded
        # tail is the end of a max_steps-limited run, so nothing follows it.
        self._current_position = mcmc.last_state.z

        return self._collect(mcmc, n_steps, run_steps)

    def _collect(self, mcmc, n_steps, run_steps):
        """Reshape one run's samples to ``n_steps`` (dropping any truncated tail) and log stats."""
        extra = mcmc.get_extra_fields()

        if self._nsamples_parallel > 1:
            # group_by_chain=True -> (nchains, run_steps, ndim)
            samples = np.asarray(mcmc.get_samples(group_by_chain=True)).reshape(
                self._nsamples_parallel, run_steps, -1)[:, :n_steps]
            log_post = -np.asarray(extra['potential_energy']).reshape(
                self._nsamples_parallel, run_steps)[:, :n_steps]
        else:
            samples = np.asarray(mcmc.get_samples()).reshape(run_steps, -1)[:n_steps]
            log_post = -np.asarray(extra['potential_energy']).reshape(run_steps)[:n_steps]

        if 'num_steps' in extra:
            nsteps = np.asarray(extra['num_steps']).ravel()
            self._total_likelihood_evaluations += int(nsteps.sum())
            self.logger.info('number of integration steps: mean %.1f, max %d',
                             nsteps.mean(), nsteps.max())
        else:
            # run_steps, not n_steps: a truncated batch still costs the full compiled length.
            self._total_likelihood_evaluations += run_steps * self._nsamples_parallel
        if 'accept_prob' in extra:
            self.logger.info('acceptance rate: mean %.3f',
                             float(np.asarray(extra['accept_prob']).mean()))
        self.logger.info('total likelihood evaluations(~): %d',
                         self._total_likelihood_evaluations)

        return samples, None, {'logposterior': log_post}


class NumpyroNUTS(_NumpyroKernel):
    """No-U-Turn Sampler (NUTS) via NumPyro.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.NUTS
    """

    logger = logging.getLogger('NumpyroNUTS')
    _numpyro_cls = 'NUTS'
    _extra_fields = ('accept_prob', 'potential_energy', 'num_steps')

    def __init__(self, step_size=1.0, inverse_mass_matrix=None, max_tree_depth=10, **kwargs):
        """
        Parameters
        ----------
        step_size : float
        inverse_mass_matrix : array_like or None
        max_tree_depth : int
        **kwargs
            Extra keyword arguments forwarded to ``numpyro.infer.NUTS``.
        """
        self.kernel_kwargs = dict(step_size=step_size, max_tree_depth=max_tree_depth, **kwargs)
        if inverse_mass_matrix is not None:
            self.kernel_kwargs['inverse_mass_matrix'] = inverse_mass_matrix


class NumpyroHMC(_NumpyroKernel):
    """Hamiltonian Monte Carlo via NumPyro.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.HMC
    """

    logger = logging.getLogger('NumpyroHMC')
    _numpyro_cls = 'HMC'
    _extra_fields = ('accept_prob', 'potential_energy', 'num_steps')

    def __init__(self, step_size=1.0, inverse_mass_matrix=None,
                 num_steps=None, trajectory_length=None, **kwargs):
        """
        Parameters
        ----------
        step_size : float
        inverse_mass_matrix : array_like or None
        num_steps : int or None
        trajectory_length : float or None
        **kwargs
            Extra keyword arguments forwarded to ``numpyro.infer.HMC``.
        """
        self.kernel_kwargs = dict(step_size=step_size, **kwargs)
        if inverse_mass_matrix is not None:
            self.kernel_kwargs['inverse_mass_matrix'] = inverse_mass_matrix
        if num_steps is not None:
            self.kernel_kwargs['num_steps'] = num_steps
        if trajectory_length is not None:
            self.kernel_kwargs['trajectory_length'] = trajectory_length


class NumpyroBarkerMH(_NumpyroKernel):
    """Barker Metropolis-Hastings sampler via NumPyro.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.barker.BarkerMH
    """

    logger = logging.getLogger('NumpyroBarkerMH')
    _numpyro_cls = 'BarkerMH'
    _extra_fields = ('accept_prob', 'potential_energy')
    # NumPyro's BarkerMH requires init_params whenever a potential_fn is used, so it cannot be
    # resumed from post_warmup_state; it also rejects chain_method='vectorized' outright, so
    # parallel runs get one kernel instance each rather than vectorized chains within one.
    _supports_continuation = False
    max_nparallel = 1

    def __init__(self, step_size=1.0, **kwargs):
        self.kernel_kwargs = dict(step_size=step_size, **kwargs)


class NumpyroSA(_NumpyroKernel):
    """Sample Adaptive (SA) MCMC sampler via NumPyro.

    SA is a gradient-free, sample-adaptive method.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.sa.SA
    """

    logger = logging.getLogger('NumpyroSA')
    _numpyro_cls = 'SA'
    _extra_fields = ('accept_prob', 'potential_energy')

    def __init__(self, **kwargs):
        self.kernel_kwargs = dict(**kwargs)


class _NumpyroEnsembleKernel(Kernel):
    """Common base for NumPyro ensemble MCMC kernels (AIES and ESS).

    Both kernels require an even number of walkers (``num_chains`` in numpyro's MCMC
    must be divisible by 2) and do not support extra fields collection via numpyro.
    Log-posterior values and derived quantities are evaluated in a single pass after
    each numpyro MCMC run.
    """

    logger = logging.getLogger('NumpyroEnsembleKernel')
    _numpyro_cls = None
    _sampler_cls = 'EnsembleSampler'
    max_nparallel = 1  # walker-level parallelism is handled internally by the ensemble

    @classmethod
    def install(cls, installer):
        installer.pip('numpyro')

    def init(self, posterior, rng, **context):
        if not NUMPYRO_INSTALLED:
            raise ImportError("The 'numpyro' package is required but not installed.")

        posterior_logpdf, posterior_logpdf_with_derived = posterior

        self._rng = rng
        self._ndim = context['ndim']

        if self.nwalkers is None:
            # 4 * ndim is always even; also satisfies the recommended nwalkers >= 2 * ndim.
            self.nwalkers = 4 * self._ndim
        if self.nwalkers % 2 != 0:
            raise ValueError(f'nwalkers must be even for {self._numpyro_cls}, got {self.nwalkers}.')

        self._posterior_logpdf_with_derived = posterior_logpdf_with_derived

        def potential_fn(flat):
            return -posterior_logpdf(jnp.asarray(flat)[None])[0]
        self._potential_fn = potential_fn

        self._numpyro_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=potential_fn, **self.kernel_kwargs)

        self._current_position = None
        self._total_likelihood_evaluations = 0

    def adapt(self, state, **kwargs):
        """No-op: ensemble samplers do not use step-size or mass-matrix adaptation."""

    def _log_run_info(self, mcmc):
        """Log kernel-specific run statistics from *mcmc.last_state*.  No-op by default."""

    def run(self, n_steps, state):
        position, _, _ = state
        # position shape: (nwalkers, ndim) in conditioned space
        if self._current_position is None:
            self._current_position = position

        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
        mcmc = numpyro.infer.MCMC(
            self._numpyro_kernel,
            num_warmup=0,
            num_samples=n_steps,
            num_chains=self.nwalkers,
            chain_method='vectorized',
            progress_bar=False)
        mcmc.run(rng_key, init_params=self._current_position)
        self._current_position = mcmc.last_state.z  # (nwalkers, ndim)
        self._log_run_info(mcmc)

        # get_samples(group_by_chain=True) → (nwalkers, n_steps, ndim); transpose → (n_steps, nwalkers, ndim)
        samples = np.asarray(mcmc.get_samples(group_by_chain=True)).transpose(1, 0, 2)

        # Evaluate log-posterior and derived quantities on all samples in one pass.
        flat_samples = jnp.asarray(samples.reshape(-1, self._ndim))
        results = self._posterior_logpdf_with_derived(flat_samples)
        log_post = np.array([result[0] for result in results]).reshape(n_steps, self.nwalkers)
        derived = np.array([result[1] for result in results]).reshape(n_steps, self.nwalkers, -1)

        self._total_likelihood_evaluations += n_steps * self.nwalkers
        self.logger.info('total likelihood evaluations: %d', self._total_likelihood_evaluations)

        return samples, derived, {'logposterior': log_post}


class NumpyroAIES(_NumpyroEnsembleKernel):
    """Affine Invariant Ensemble Sampler (AIES) via NumPyro.

    A gradient-free ensemble method that proposes moves using information from other
    walkers (differential-evolution or stretch moves).  The number of walkers must be
    even and at least ``2 * ndim``; ``4 * ndim`` (the default) is a safe choice.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.ensemble.AIES
    - https://arxiv.org/abs/1202.3665 (emcee)
    """

    logger = logging.getLogger('NumpyroAIES')
    _numpyro_cls = 'AIES'

    def __init__(self, nwalkers=None, moves=None, randomize_split=False, **kwargs):
        """
        Parameters
        ----------
        nwalkers : int or None
            Number of ensemble walkers.  Must be even and ``>= 2 * ndim``.
            ``None`` defers to ``4 * ndim``, set during :meth:`init`.
        moves : dict or None
            Mapping of move objects to their selection probabilities, e.g.
            ``{numpyro.infer.AIES.StretchMove(): 1.0}``.  ``None`` uses the
            default ``DEMove``.
        randomize_split : bool
            Whether to randomly permute walker order at each iteration.  Default ``False``.
        **kwargs
            Extra keyword arguments forwarded to ``numpyro.infer.AIES``.
        """
        self.nwalkers = nwalkers
        self.kernel_kwargs = dict(randomize_split=randomize_split, **kwargs)
        if moves is not None:
            self.kernel_kwargs['moves'] = moves

    def _log_run_info(self, mcmc):
        try:
            mean_accept = float(mcmc.last_state.inner_state.mean_accept_prob)
            self.logger.info('acceptance rate: mean %.3f', mean_accept)
        except Exception:
            pass


class NumpyroESS(_NumpyroEnsembleKernel):
    """Ensemble Slice Sampler (ESS) via NumPyro.

    A gradient-free ensemble method that uses slice-sampling directions informed by
    other walkers.  The number of walkers must be even and at least ``2 * ndim``;
    ``4 * ndim`` (the default) is a safe choice.

    .. rubric:: References
    - https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.ensemble.ESS
    - https://arxiv.org/abs/2002.06212 (Karamanis & Beutler)
    """

    logger = logging.getLogger('NumpyroESS')
    _numpyro_cls = 'ESS'

    def __init__(self, nwalkers=None, moves=None, randomize_split=True,
                 init_mu=1.0, tune_mu=True, max_steps=10000, **kwargs):
        """
        Parameters
        ----------
        nwalkers : int or None
            Number of ensemble walkers.  Must be even and ``>= 2 * ndim``.
            ``None`` defers to ``4 * ndim``, set during :meth:`init`.
        moves : dict or None
            Mapping of move objects to their selection probabilities, e.g.
            ``{numpyro.infer.ESS.GaussianMove(): 1.0}``.  ``None`` uses the
            default ``DifferentialMove``.
        randomize_split : bool
            Whether to randomly permute walker order at each iteration.  Default ``True``.
        init_mu : float
            Initial scale factor for the slice width.  Default ``1.0``.
        tune_mu : bool
            Whether to adapt the scale factor during sampling.  Default ``True``.
        max_steps : int
            Maximum number of stepping-out steps per sample.  Default ``10000``.
        **kwargs
            Extra keyword arguments forwarded to ``numpyro.infer.ESS``.
        """
        self.nwalkers = nwalkers
        self.kernel_kwargs = dict(randomize_split=randomize_split, init_mu=init_mu,
                                  tune_mu=tune_mu, max_steps=max_steps, **kwargs)
        if moves is not None:
            self.kernel_kwargs['moves'] = moves

    def _log_run_info(self, mcmc):
        try:
            mu = float(mcmc.last_state.inner_state.mu)
            self.logger.info('slice scale mu: %.3g', mu)
        except Exception:
            pass
