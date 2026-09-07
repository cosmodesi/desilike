"""BlackJAX HMC, NUTS, and MCLMC kernels."""

import logging
from functools import partial

import numpy as np
import jax

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

        self.kernel_args.update(dict(L=float(params.L), step_size=float(params.step_size)))
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
