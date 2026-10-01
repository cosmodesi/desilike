"""BlackJAX HMC, NUTS, MCLMC and LAPS kernels."""

import logging
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp

try:
    import blackjax
    BLACKJAX_INSTALLED = True
except ModuleNotFoundError:
    BLACKJAX_INSTALLED = False

from .base import Kernel


def make_steps_factory(step):
    """Return a JIT-compiled function that advances a BlackJAX state by N steps.

    Parameters
    ----------
    step : callable
        The BlackJAX kernel step function ``(rng_key, state) -> (state, info)``.

    Returns
    -------
    callable
        ``(state, rng_keys) -> (final_state, (all_states, last_info))``
    """

    def make_one_step(state, rng_key):
        state, info = step(rng_key, state)
        return state, (state, info)

    def make_steps(args):
        state, rng_keys = args
        return jax.lax.scan(make_one_step, state, rng_keys)

    return jax.jit(make_steps)


def make_steps_vmap_factory(step):
    """Return a JIT-compiled function that advances a batch of BlackJAX states by N steps via vmap.

    Parameters
    ----------
    step : callable
        The BlackJAX kernel step function ``(rng_key, state) -> (state, info)``.

    Returns
    -------
    callable
        ``(batched_state, rng_keys) -> (final_states, (all_states, last_info))``
        where batched_state has a leading chain dimension and rng_keys has shape
        ``(nchains, n_steps, 2)``.
    """

    def make_one_step(state, rng_key):
        state, info = step(rng_key, state)
        return state, (state, info)

    def scan_one_chain(args):
        state, rng_keys = args
        return jax.lax.scan(make_one_step, state, rng_keys)

    batched = jax.vmap(scan_one_chain)

    @jax.jit
    def make_steps(args):
        states, rng_keys = args
        return batched((states, rng_keys))

    return make_steps


def _log_adaptation(logger, kernel_args):
    if 'step_size' in kernel_args:
        logger.info('step_size: %.3g', float(kernel_args['step_size']))
    if 'inverse_mass_matrix' in kernel_args:
        imm = np.asarray(kernel_args['inverse_mass_matrix'])
        if imm.ndim == 2:
            eig = np.linalg.eigvalsh(imm)
            logger.info('inverse_mass_matrix eigenvalues: min %.3g, max %.3g, cond %.3g, det^{1/n} %.3g',
                        eig.min(), eig.max(), eig.max() / eig.min(), eig.prod() ** (1. / len(eig)))
        else:
            imm = imm.ravel()
            logger.info('inverse_mass_matrix: min %.3g, max %.3g, det^{1/n} %.3g',
                        imm.min(), imm.max(), imm.prod() ** (1. / len(imm)))


class _BlackJAXKernel(Kernel):
    """Common base for BlackJAX gradient-based kernels."""

    logger = logging.getLogger('BlackJAXKernel')
    _kernel_type_name = None    # 'hmc', 'nuts', or 'mclmc'
    _adaptation_fn_name = None  # 'window_adaptation', 'mclmc_find_L_and_step_size'
    max_nparallel = None  # blackjax handles any number of chains via jax.vmap

    @classmethod
    def install(cls, installer):
        installer.pip('blackjax')

    def _check_installed(self):
        if not BLACKJAX_INSTALLED:
            raise ImportError("The 'blackjax' package is required but not installed.")

    def init(self, posterior, rng, **context):
        self._check_installed()
        self._rng = rng
        self._nsamples_parallel = context.get('nsamples_parallel', 1)

        posterior_logpdf, _ = posterior

        def _logpost_flat(flat):
            return posterior_logpdf(flat[None])[0]

        self._logposterior = jax.jit(_logpost_flat)

        kernel_type = getattr(blackjax, self._kernel_type_name)
        adaptation_fn = getattr(blackjax, self._adaptation_fn_name)

        self._kernel_cls = kernel_type
        self._adaptation_fn = adaptation_fn

        kernel = kernel_type(self._logposterior, **self.kernel_args, **self.fixed_kernel_args)
        if self._nsamples_parallel > 1:
            self._make_steps = make_steps_vmap_factory(kernel.step)
        else:
            self._make_steps = make_steps_factory(kernel.step)
        self._kernel = kernel
        self._state = None   # initialised lazily on first run / after adapt
        self._total_likelihood_evaluations = 0

    def get_state(self):
        """Adapted metric, step size and chain state, for reuse by a resumed run.

        BlackJAX carries less hidden state than NumPyro: momentum is resampled every step and the
        RNG is passed per call, so the chain state is just ``(position, logdensity,
        logdensity_grad)``.  Persisting it alongside ``kernel_args`` is therefore complete --
        unlike the metric-only case, which leaves the sampler to restart cold and puts a
        discontinuity at the join.
        """
        if self._state is None:
            return None
        state = {key: np.asarray(value) for key, value in self.kernel_args.items()}
        # BlackJAX kernels do not keep `_ndim`; take it from the restored chain state instead.
        state['ndim'] = np.asarray(np.shape(np.asarray(self._state.position))[-1])
        state['nchains'] = np.asarray(self._nsamples_parallel)
        state['chain_state'] = jax.device_get(self._state)
        return state

    def set_state(self, state):
        """Restore a saved metric and chain state; returns True so warmup is skipped."""
        if 'chain_state' not in state or 'inverse_mass_matrix' not in state:
            return False
        ndim = int(np.shape(np.asarray(state['chain_state'].position))[-1])
        if 'ndim' in state and int(state['ndim']) != ndim:
            self.logger.warning('Saved kernel state is inconsistent (%d vs %d parameters); '
                                're-adapting.', int(state['ndim']), ndim)
            return False
        if int(state.get('nchains', self._nsamples_parallel)) != self._nsamples_parallel:
            self.logger.warning('Saved kernel state is for %d chains, this run has %d; '
                                're-adapting.', int(state['nchains']), self._nsamples_parallel)
            return False
        for key in list(self.kernel_args):
            if key in state:
                value = state[key]
                self.kernel_args[key] = float(value) if value.ndim == 0 else np.asarray(value)
        self._kernel = self._kernel_cls(
            self._logposterior, **self.kernel_args, **self.fixed_kernel_args)
        self._make_steps = (make_steps_vmap_factory(self._kernel.step) if self._nsamples_parallel > 1
                            else make_steps_factory(self._kernel.step))
        self._state = jax.tree_util.tree_map(jnp.asarray, state['chain_state'])
        _log_adaptation(self.logger, self.kernel_args)
        return True

    def _init_state_single(self, initial_position):
        try:
            return self._kernel.init(initial_position)
        except TypeError:
            rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
            return self._kernel.init(initial_position, rng_key)

    def _get_or_init_state(self, initial_position=None):
        if self._state is None:
            if self._nsamples_parallel > 1:
                # initial_position: (nchains, ndim)
                self._state = jax.vmap(self._init_state_single)(initial_position)
            else:
                self._state = self._init_state_single(initial_position)
        return self._state

    def run(self, n_steps, state):
        position, _, _ = state
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))

        if self._nsamples_parallel > 1:
            current_state = self._get_or_init_state(initial_position=position)
            # rng_keys: (nchains, n_steps, 2)
            rng_keys = jax.random.split(rng_key, self._nsamples_parallel * n_steps)
            rng_keys = rng_keys.reshape(self._nsamples_parallel, n_steps, -1)
            self._state, (all_states, last_info) = self._make_steps((current_state, rng_keys))
            samples  = np.asarray(all_states.position).reshape(self._nsamples_parallel, n_steps, -1)
            log_post = np.asarray(all_states.logdensity).reshape(self._nsamples_parallel, n_steps)
        else:
            current_state = self._get_or_init_state(initial_position=position)
            rng_keys = jax.random.split(rng_key, n_steps)
            self._state, (all_states, last_info) = self._make_steps((current_state, rng_keys))
            samples  = np.asarray(all_states.position).reshape(n_steps, -1)
            log_post = np.asarray(all_states.logdensity).reshape(n_steps)

        if hasattr(last_info, 'num_integration_steps'):
            nsteps = np.asarray(last_info.num_integration_steps).ravel()
            self._total_likelihood_evaluations += int(nsteps.sum())
            self.logger.info('number of integration steps: mean %.1f, max %d',
                             nsteps.mean(), nsteps.max())
        if hasattr(last_info, 'acceptance_rate'):
            arate = np.asarray(last_info.acceptance_rate).ravel()
            self.logger.info('acceptance rate: mean %.3f', arate.mean())
        if self._total_likelihood_evaluations:
            self.logger.info('total likelihood evaluations(~): %d', self._total_likelihood_evaluations)

        return samples, None, {'logposterior': log_post}


class BlackjaxHMC(_BlackJAXKernel):
    """Hamiltonian Monte Carlo (HMC) kernel via BlackJAX.

    .. rubric:: References
    - https://github.com/blackjax-devs/blackjax
    """

    logger = logging.getLogger('BlackjaxHMC')
    _kernel_type_name = 'hmc'
    _adaptation_fn_name = 'window_adaptation'

    def __init__(self, step_size=1e-3, inverse_mass_matrix=None,
                 num_integration_steps=60, **kwargs):
        """
        Parameters
        ----------
        step_size : float
        inverse_mass_matrix : array_like or None
        num_integration_steps : int
        **kwargs
            Extra fixed kwargs passed to ``blackjax.hmc``.
        """
        self.kernel_args = dict(step_size=step_size)
        self._imm_init = inverse_mass_matrix
        self.fixed_kernel_args = dict(num_integration_steps=num_integration_steps, **kwargs)

    def init(self, posterior, rng, **context):
        if self._imm_init is None:
            self.kernel_args['inverse_mass_matrix'] = np.ones(context['ndim'])
        else:
            self.kernel_args['inverse_mass_matrix'] = np.asarray(self._imm_init)
        super().init(posterior, rng, **context)

    def adapt(self, state, **kwargs):
        """Adapt step size and mass matrix via ``blackjax.window_adaptation``."""
        position, _, _ = state
        # Use first chain's position if batched.
        init_position = np.asarray(position)[0] if np.asarray(position).ndim > 1 else position
        steps = kwargs.pop('steps')
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
        single_state = self._init_state_single(init_position)
        (single_state, parameters), _ = self._adaptation_fn(
            self._kernel_cls, self._logposterior,
            **self.fixed_kernel_args, **kwargs).run(
            rng_key, single_state.position, num_steps=steps)
        self.kernel_args.update({k: v for k, v in parameters.items()
                                  if k not in self.fixed_kernel_args})
        self._kernel = self._kernel_cls(
            self._logposterior, **self.kernel_args, **self.fixed_kernel_args)
        if self._nsamples_parallel > 1:
            self._make_steps = make_steps_vmap_factory(self._kernel.step)
            # Leave self._state = None so _get_or_init_state re-initialises from
            # the batched position on the first run() call.
            self._state = None
        else:
            self._make_steps = make_steps_factory(self._kernel.step)
            self._state = single_state
        self.logger.info('Adaptation done.')
        _log_adaptation(self.logger, self.kernel_args)


class BlackjaxNUTS(_BlackJAXKernel):
    """No-U-Turn Sampler (NUTS) kernel via BlackJAX.

    .. rubric:: References
    - https://github.com/blackjax-devs/blackjax
    """

    logger = logging.getLogger('BlackjaxNUTS')
    _kernel_type_name = 'nuts'
    _adaptation_fn_name = 'window_adaptation'

    def __init__(self, step_size=1e-3, inverse_mass_matrix=None, **kwargs):
        """
        Parameters
        ----------
        step_size : float
        inverse_mass_matrix : array_like or None
        **kwargs
            Extra fixed kwargs passed to ``blackjax.nuts``.
        """
        self.kernel_args = dict(step_size=step_size)
        self._imm_init = inverse_mass_matrix
        self.fixed_kernel_args = dict(**kwargs)

    def init(self, posterior, rng, **context):
        if self._imm_init is None:
            self.kernel_args['inverse_mass_matrix'] = np.ones(context['ndim'])
        else:
            self.kernel_args['inverse_mass_matrix'] = np.asarray(self._imm_init)
        super().init(posterior, rng, **context)

    def adapt(self, state, **kwargs):
        """Adapt step size and mass matrix via ``blackjax.window_adaptation``."""
        position, _, _ = state
        # Use first chain's position if batched.
        init_position = np.asarray(position)[0] if np.asarray(position).ndim > 1 else position
        steps = kwargs.pop('steps')
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
        single_state = self._init_state_single(init_position)
        (single_state, parameters), _ = self._adaptation_fn(
            self._kernel_cls, self._logposterior,
            **self.fixed_kernel_args, **kwargs).run(
            rng_key, single_state.position, num_steps=steps)
        self.kernel_args.update({k: v for k, v in parameters.items()
                                  if k not in self.fixed_kernel_args})
        self._kernel = self._kernel_cls(
            self._logposterior, **self.kernel_args, **self.fixed_kernel_args)
        if self._nsamples_parallel > 1:
            self._make_steps = make_steps_vmap_factory(self._kernel.step)
            self._state = None
        else:
            self._make_steps = make_steps_factory(self._kernel.step)
            self._state = single_state
        self.logger.info('Adaptation done.')
        _log_adaptation(self.logger, self.kernel_args)


class BlackjaxMCLMC(_BlackJAXKernel):
    """Microcanonical Langevin Monte Carlo (MCLMC) kernel via BlackJAX.

    .. rubric:: References
    - https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    - https://arxiv.org/abs/2212.08549
    """

    logger = logging.getLogger('BlackjaxMCLMC')
    _kernel_type_name = 'mclmc'
    _adaptation_fn_name = 'mclmc_find_L_and_step_size'

    def __init__(self, L=1., step_size=0.1, **kwargs):
        self.kernel_args = dict(L=L, step_size=step_size)
        self.fixed_kernel_args = dict(**kwargs)

    def adapt(self, state, **kwargs):
        """Adapt ``L`` and ``step_size`` via ``blackjax.mclmc_find_L_and_step_size``."""
        import inspect
        import blackjax.mcmc.mclmc as mclmc_mod

        position, _, _ = state
        # Use first chain's position if batched.
        init_position = np.asarray(position)[0] if np.asarray(position).ndim > 1 else position
        steps = kwargs.pop('steps')

        single_state = self._init_state_single(init_position)
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))

        _mass_matrix_kwarg = (
            'inverse_mass_matrix'
            if 'inverse_mass_matrix' in inspect.signature(mclmc_mod.as_top_level_api).parameters
            else 'sqrt_diag_cov'
        )

        def mclmc_kernel_factory(mass_matrix):
            return mclmc_mod.build_kernel(
                self._logposterior,
                mass_matrix,
                mclmc_mod.isokinetic_mclachlan,
            )

        single_state, params, *_ = self._adaptation_fn(
            mclmc_kernel_factory, num_steps=steps,
            state=single_state, rng_key=rng_key, **kwargs)

        L, step_size = float(params.L), float(params.step_size)
        # A hard -inf boundary makes the energy error undefined, and `mclmc_find_L_and_step_size`
        # answers by driving L to zero -- blackjax then raises `ZeroDivisionError` from
        # `partially_refresh_momentum` (`exp(2 * step_size / L)`), or the chain comes back all
        # NaN. Reproduced on a plain Gaussian with box priors biting at 1.5 sigma: L = 0 exactly.
        # Fail here, where the cause is nameable, instead of deep inside the integrator.
        if not np.isfinite(L) or L <= 0. or not np.isfinite(step_size) or step_size <= 0.:
            raise ValueError(
                f'MCLMC adaptation returned L={L:.3g}, step_size={step_size:.3g}. This is what a hard -inf '
                'boundary in the posterior does to it -- the energy error is undefined at the '
                'wall and the tuner collapses. Restrict the priors so the sampled region has no '
                'cliff (e.g. to an emulator\'s trained box), or use a kernel that screens '
                'impossible points (emcee, pocoMC, nautilus) instead.')
        self.kernel_args.update(dict(L=L, step_size=step_size))
        adapted_mass_matrix = np.asarray(getattr(params, _mass_matrix_kwarg))
        self._kernel = self._kernel_cls(
            self._logposterior,
            **self.kernel_args, **self.fixed_kernel_args,
            **{_mass_matrix_kwarg: adapted_mass_matrix})
        if self._nsamples_parallel > 1:
            self._make_steps = make_steps_vmap_factory(self._kernel.step)
            self._state = None
        else:
            self._make_steps = make_steps_factory(self._kernel.step)
            self._state = single_state
        self.logger.info('Adaptation done.')
        self.logger.info('L: %.3g  step_size: %.3g', self.kernel_args['L'], self.kernel_args['step_size'])
        imm = adapted_mass_matrix.ravel()
        self.logger.info('mass_matrix (%s): min %.3g, max %.3g, det^{1/n} %.3g',
                         _mass_matrix_kwarg, imm.min(), imm.max(), imm.prod() ** (1. / len(imm)))


class BlackjaxLAPS(Kernel):
    """Late Adjusted Parallel Sampler (LAPS) via BlackJAX.

    An ensemble of ``nwalkers`` chains shares its adaptation: the ensemble moments set the
    trajectory length, the diagonal preconditioner and the step size at every step.  Warmup is
    the LAPS algorithm proper, in two phases:

    1. unadjusted MCLMC, with the step size driven by the energy-error variance and stopped early
       once the ensemble has converged (``r_end``);
    2. Metropolis-adjusted MCLMC, with the step size bisected to the target acceptance.

    Sampling then continues with adjusted MCLMC at the adapted, now fixed, parameters, one sample
    per walker per step, so each walker is an ordinary chain for the convergence checks.

    LAPS is gradient-based and needs a smooth target: a hard ``-inf`` wall stalls the phase-1
    energy-error estimate, as it does for :class:`BlackjaxMCLMC`.  It pays off with many chains --
    hundreds to thousands, on GPU -- and needs ``blackjax >= 1.6``.

    .. rubric:: References
    - https://github.com/blackjax-devs/blackjax (``blackjax.adaptation.laps``)
    """

    logger = logging.getLogger('BlackjaxLAPS')
    _sampler_cls = 'EnsembleSampler'

    def __init__(self, nwalkers=None, num_steps1=1000, num_steps2=3000, steps_per_sample=15,
                 acc_prob=None, integrator_coefficients=None, diagonal_preconditioning=True,
                 alpha=1.9, C=0.1, r_end=0.01, early_stop=True, save_frac=0.2, bias_type=3,
                 L_proposal_factor=1.25):
        """
        Parameters
        ----------
        nwalkers : int or None
            Number of chains.  ``None`` defers to ``max(128, 4 * ndim)``, rounded up to a multiple
            of the number of jax devices the chains are sharded over.
        num_steps1 : int
            Maximum number of unadjusted steps in phase 1 (fewer with ``early_stop``).
        num_steps2 : int
            Gradient-evaluation budget per chain of phase 2; the number of adjusted samples is
            ``num_steps2 // (gradient calls per integration step * steps_per_sample)``.
        steps_per_sample : int
            Integration steps per adjusted sample.
        acc_prob : float or None
            Target acceptance in phase 2.  ``None`` gives 0.7, or 0.9 above 200 dimensions.
        integrator_coefficients : list or None
            Isokinetic integrator coefficients.  ``None`` gives McLachlan, or Omelyan above 200
            dimensions.
        diagonal_preconditioning : bool
            Precondition phase 2 and sampling with the phase-1 ensemble variances.
        alpha, C, r_end, save_frac, bias_type : float, float, float, float, int
            Phase-1 tuning, as in ``blackjax.adaptation.laps.laps``: ``L = alpha sqrt(d) sigma``,
            ``C`` sets the energy-error target, ``r_end`` the early-stop threshold on the ensemble
            fluctuations over the last ``save_frac`` of the steps, ``bias_type`` which bias
            estimate the step size follows (3 = diagonal equipartition).
        early_stop : bool
            Stop phase 1 once the fluctuations fall below ``r_end``.
        L_proposal_factor : float
            Partial momentum refreshment of the adjusted kernel, in units of the trajectory length.
        """
        self.nwalkers = nwalkers
        self.num_steps1 = int(num_steps1)
        self.num_steps2 = int(num_steps2)
        self.steps_per_sample = int(steps_per_sample)
        self.acc_prob = acc_prob
        self.integrator_coefficients = integrator_coefficients
        self.diagonal_preconditioning = bool(diagonal_preconditioning)
        self.alpha, self.C, self.r_end = float(alpha), float(C), float(r_end)
        self.early_stop = bool(early_stop)
        self.save_frac = float(save_frac)
        self.bias_type = int(bias_type)
        self.L_proposal_factor = float(L_proposal_factor)

    @classmethod
    def install(cls, installer):
        installer.pip('blackjax>=1.6')

    def init(self, posterior, rng, **context):
        if not BLACKJAX_INSTALLED:
            raise ImportError("The 'blackjax' package is required but not installed.")
        try:
            from blackjax.adaptation import laps  # noqa: F401
        except ImportError as exc:
            raise ImportError(f'BlackjaxLAPS needs blackjax >= 1.6 (blackjax.adaptation.laps); '
                              f'blackjax {blackjax.__version__} is installed.') from exc
        from blackjax.mcmc.integrators import mclachlan_coefficients, omelyan_coefficients

        posterior_logpdf, _ = posterior
        self._logdensity = lambda flat: posterior_logpdf(flat[None])[0]
        self._rng = rng
        self._ndim = context['ndim']
        devices = jax.devices()
        self._mesh = jax.sharding.Mesh(np.array(devices), ('chains',))
        # the chain axis of every array the kernel holds is sharded over the jax devices
        self._sharding = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec('chains'))
        if self.nwalkers is None:
            self.nwalkers = -(-max(128, 4 * self._ndim) // len(devices)) * len(devices)
        if self.nwalkers % len(devices):
            raise ValueError(f'nwalkers = {self.nwalkers} must be a multiple of the {len(devices)} '
                             'jax devices the chains are sharded over.')
        high_dimensional = self._ndim > 200
        self._coefficients = self.integrator_coefficients
        if self._coefficients is None:
            self._coefficients = omelyan_coefficients if high_dimensional else mclachlan_coefficients
        self._acc_prob = self.acc_prob
        if self._acc_prob is None:
            self._acc_prob = 0.9 if high_dimensional or self.integrator_coefficients is not None else 0.7
        # B updates in BABA...B are len // 2 + 1, the last one's gradient reused by the next step.
        self._gradient_calls_per_step = len(self._coefficients) // 2
        self._params = None
        self._state = None
        self._sample = None
        self._total_likelihood_evaluations = 0

    def adapt(self, state, steps=None, num_steps1=None, num_steps2=None):
        """Run both LAPS phases from the current walkers.

        ``steps`` sets both ``num_steps1`` and ``num_steps2``; either can be given on its own.
        """
        from blackjax.adaptation import laps, laps_burn_in
        from blackjax.eca import ensemble_execute_fn, run_eca
        from blackjax.mcmc.hmc import HMCState
        from blackjax.mcmc.integrators import IntegratorState, _normalized_flatten_array

        num_steps1 = int(num_steps1 or steps or self.num_steps1)
        num_steps2 = int(num_steps2 or steps or self.num_steps2)
        position, _, _ = state
        position = np.asarray(position).reshape(self.nwalkers, self._ndim)

        # Phase 1 starts from the walkers the sampler placed (`laps_burn_in.initialize` would draw
        # its own positions): velocity along +- grad log p, the sign set by the ensemble equipartition.
        def sequential_init(key, position, args):
            logdensity, logdensity_grad = jax.value_and_grad(self._logdensity)(position)
            velocity = _normalized_flatten_array(logdensity_grad)[0]
            return IntegratorState(position, velocity, logdensity, logdensity_grad), None

        def flip(key, integrator_state, signs):
            return integrator_state._replace(momentum=signs * integrator_state.momentum), None

        key_init, key_flip = jax.random.split(jax.random.PRNGKey(int(self._rng.integers(2**32))))
        integrator_state, equipartition = ensemble_execute_fn(
            sequential_init, key_init, self.nwalkers, self._mesh,
            x=jax.device_put(jnp.asarray(position), self._sharding),
            summary_statistics_fn=lambda integrator_state: -integrator_state.position * integrator_state.logdensity_grad)
        signs = -2. * (equipartition < 1.) + 1.
        integrator_state, _ = ensemble_execute_fn(flip, key_flip, self.nwalkers, self._mesh, x=integrator_state, args=signs)

        # Phase 1: unadjusted MCLMC, early-stopped on the ensemble fluctuations.
        adaptation = laps_burn_in.Adaptation(
            self._ndim, microcanonical=True, alpha=self.alpha, bias_type=self.bias_type,
            save_num=int(round(self.save_frac * num_steps1)), C=self.C, r_end=self.r_end)
        integrator_state, adapted, info1 = run_eca(
            jax.random.PRNGKey(int(self._rng.integers(2**32))), integrator_state,
            laps_burn_in.build_kernel(self._logdensity, self._ndim, microcanonical=True),
            adaptation, num_steps1, self.nwalkers, self._mesh, superchain_size=1,
            early_stop=self.early_stop)
        nsteps1 = len(info1['step_size'])
        self._total_likelihood_evaluations += nsteps1 * self.nwalkers
        self.logger.info('Phase 1: %d unadjusted steps (of %d); r_max %.3g, equipartition %.3g.',
                         nsteps1, num_steps1, float(info1['r_max'][-1]), float(info1['equi_diag'][-1]))

        # Phase 2: adjusted MCLMC, step size bisected to the target acceptance.
        if self.diagonal_preconditioning:
            inverse_mass_matrix = jnp.asarray(adapted.inverse_mass_matrix)
            # The step size reflects the average scale change of the preconditioning.
            adapted = adapted._replace(step_size=adapted.step_size / jnp.sqrt(jnp.mean(inverse_mass_matrix)))
        else:
            inverse_mass_matrix = jnp.ones(self._ndim)
        adapted = adapted._replace(step_size=float(adapted.step_size))
        calls_per_sample = self._gradient_calls_per_step * self.steps_per_sample
        nsamples2 = num_steps2 // calls_per_sample
        if nsamples2 < 1:
            raise ValueError(f'num_steps2 = {num_steps2} buys no adjusted sample: it needs at least '
                             f'{calls_per_sample} gradient calls.')
        adaptation = laps.Adaptation(adapted, nsamples2 // 2, self.steps_per_sample, self._acc_prob)
        kernel = self._build_adjusted_kernel(inverse_mass_matrix)
        chain_state, adapted, info2 = run_eca(
            jax.random.PRNGKey(int(self._rng.integers(2**32))),
            HMCState(integrator_state.position, integrator_state.logdensity, integrator_state.logdensity_grad),
            lambda key, chain_state, params: kernel(key, chain_state, params.step_size),
            adaptation, nsamples2, self.nwalkers, self._mesh, superchain_size=1)
        self._total_likelihood_evaluations += nsamples2 * calls_per_sample * self.nwalkers

        step_size = float(adapted.step_size)
        if not np.isfinite(step_size) or step_size <= 0.:
            raise ValueError(f'LAPS adaptation returned step_size = {step_size:.3g}. A hard -inf '
                             'boundary in the posterior does this to a gradient sampler: restrict the '
                             'priors so the sampled region has no cliff.')
        self._params = dict(step_size=step_size, inverse_mass_matrix=np.asarray(inverse_mass_matrix))
        self._state = chain_state
        self._sample = None
        self.logger.info('Phase 2: %d adjusted samples; acceptance %.3f (target %.2f).',
                         nsamples2, float(info2['acc_prob'][-1]), self._acc_prob)
        inverse_mass_matrix = self._params['inverse_mass_matrix']
        self.logger.info('step_size %.3g, L %.3g, steps_per_sample %d; inverse_mass_matrix: min %.3g, max %.3g.',
                         self._params['step_size'], self._params['step_size'] * self.steps_per_sample,
                         self.steps_per_sample, inverse_mass_matrix.min(), inverse_mass_matrix.max())

    def _build_adjusted_kernel(self, inverse_mass_matrix):
        """The adjusted MCLMC step ``(key, state, step_size) -> (state, info)``, for the adaptation and the sampling."""
        from blackjax.mcmc.adjusted_mclmc import build_kernel
        from blackjax.mcmc.integrators import generate_isokinetic_integrator

        kernel = build_kernel(integrator=generate_isokinetic_integrator(self._coefficients))

        def step(key, state, step_size):
            return kernel(rng_key=key, state=state, logdensity_fn=self._logdensity,
                          step_size=step_size, integration_steps_params=(self.steps_per_sample,),
                          inverse_mass_matrix=inverse_mass_matrix,
                          L_proposal_factor=self.L_proposal_factor)

        return step

    def get_state(self):
        """Adapted step size and preconditioner, and the chain state, for a resumed run."""
        if self._state is None:
            return None
        return dict(**self._params, ndim=np.asarray(self._ndim), nwalkers=np.asarray(self.nwalkers),
                    chain_state=jax.device_get(self._state))

    def set_state(self, state):
        """Restore a saved adaptation and chain state; returns True so warmup is skipped."""
        if 'chain_state' not in state:
            return False
        if int(state['ndim']) != self._ndim or int(state['nwalkers']) != self.nwalkers:
            self.logger.warning('Saved kernel state is for %d walkers in %d dimensions, this run has '
                                '%d in %d; re-adapting.', int(state['nwalkers']), int(state['ndim']),
                                self.nwalkers, self._ndim)
            return False
        self._params = dict(step_size=float(state['step_size']),
                            inverse_mass_matrix=np.asarray(state['inverse_mass_matrix']))
        self._state = jax.tree_util.tree_map(jnp.asarray, state['chain_state'])
        self._sample = None
        inverse_mass_matrix = self._params['inverse_mass_matrix']
        self.logger.info('step_size %.3g, L %.3g, steps_per_sample %d; inverse_mass_matrix: min %.3g, max %.3g.',
                         self._params['step_size'], self._params['step_size'] * self.steps_per_sample,
                         self.steps_per_sample, inverse_mass_matrix.min(), inverse_mass_matrix.max())
        return True

    def run(self, n_steps, state):
        if self._params is None:
            self.logger.info('No adaptation requested; running LAPS warmup with num_steps1 = %d, '
                             'num_steps2 = %d.', self.num_steps1, self.num_steps2)
            self.adapt(state)
        if self._sample is None:
            kernel = self._build_adjusted_kernel(jnp.asarray(self._params['inverse_mass_matrix']))
            step_size = self._params['step_size']

            def one_chain(chain_state, keys):
                def body(chain_state, key):
                    chain_state, info = kernel(key, chain_state, step_size)
                    return chain_state, (chain_state.position, chain_state.logdensity, info.acceptance_rate)
                return jax.lax.scan(body, chain_state, keys)

            # The chain axis is sharded over the mesh on input; jit partitions the vmap along it.
            self._sample = jax.jit(jax.vmap(one_chain))
        keys = jax.random.split(jax.random.PRNGKey(int(self._rng.integers(2**32))), self.nwalkers * n_steps)
        keys = keys.reshape(self.nwalkers, n_steps, -1)
        self._state, (positions, logdensity, acceptance) = self._sample(
            jax.device_put(self._state, self._sharding), jax.device_put(keys, self._sharding))
        self._total_likelihood_evaluations += n_steps * self.steps_per_sample * self._gradient_calls_per_step * self.nwalkers
        self.logger.info('acceptance rate: mean %.3f; total likelihood evaluations(~): %d',
                         float(np.mean(acceptance)), self._total_likelihood_evaluations)
        # (nwalkers, n_steps, ...) -> (n_steps, nwalkers, ...), the ensemble layout.
        samples = np.swapaxes(np.asarray(positions), 0, 1)
        log_post = np.swapaxes(np.asarray(logdensity), 0, 1)
        return samples, None, {'logposterior': log_post}
