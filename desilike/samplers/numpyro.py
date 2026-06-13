"""NumPyro NUTS, HMC, BarkerMH, and SA kernels."""

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

    def init(self, posterior_logpdf, rng, **context):
        if not NUMPYRO_INSTALLED:
            raise ImportError("The 'numpyro' package is required but not installed.")

        self._rng = rng
        self._ndim = context['ndim']

        import jax.numpy as _jnp
        def potential_fn(flat):
            return -posterior_logpdf(_jnp.asarray(flat)[None])[0]
        self._potential_fn = potential_fn

        self._numpyro_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=potential_fn, **self.kernel_kwargs)

        self._current_position = None
        self._total_likelihood_evaluations = 0

    def adapt(self, initial_position=None, **kwargs):
        """Run NumPyro warmup and rebuild the kernel with adapted parameters.

        Parameters
        ----------
        initial_position : dict or None
            Starting position ``{name: array}`` in rescaled space.  Used only on the
            first call; ignored if the sampler already has a current position.
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
        if self._current_position is None:
            self._current_position = initial_position
        steps = kwargs.pop('steps')

        warmup_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=self._potential_fn, **self.kernel_kwargs, **kwargs)
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
        warmup_mcmc = numpyro.infer.MCMC(
            warmup_kernel, num_warmup=steps, num_samples=1, progress_bar=False)
        warmup_mcmc.run(rng_key, init_params=self._current_position)
        self._current_position = warmup_mcmc.last_state.z

        adapt_state = warmup_mcmc.last_state.adapt_state
        kernel_sig = inspect.signature(
            getattr(numpyro.infer, self._numpyro_cls).__init__).parameters

        _warmup_only = {'adapt_step_size', 'adapt_mass_matrix', 'target_accept_prob'}
        for key, value in kwargs.items():
            if key not in _warmup_only and key in kernel_sig:
                self.kernel_kwargs[key] = value

        if hasattr(adapt_state, 'step_size') and 'step_size' in kernel_sig:
            self.kernel_kwargs['step_size'] = float(adapt_state.step_size)
            self.kernel_kwargs['adapt_step_size'] = False
        if hasattr(adapt_state, 'inverse_mass_matrix') and 'inverse_mass_matrix' in kernel_sig:
            self.kernel_kwargs['inverse_mass_matrix'] = np.asarray(adapt_state.inverse_mass_matrix)
            self.kernel_kwargs['adapt_mass_matrix'] = False

        self._numpyro_kernel = getattr(numpyro.infer, self._numpyro_cls)(
            potential_fn=self._potential_fn, **self.kernel_kwargs)

        self.logger.info('Adaptation done.')
        _log_adaptation(self.logger, self.kernel_kwargs)

    def run(self, n_steps, initial_position=None):
        if self._current_position is None:
            self._current_position = initial_position
        rng_key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
        mcmc = numpyro.infer.MCMC(
            self._numpyro_kernel, num_warmup=0, num_samples=n_steps, progress_bar=False)
        mcmc.run(rng_key, extra_fields=self._extra_fields,
                 init_params=self._current_position)
        self._current_position = mcmc.last_state.z

        samples = np.asarray(mcmc.get_samples()).reshape(n_steps, -1)
        extra = mcmc.get_extra_fields()
        log_post = -np.asarray(extra['potential_energy']).reshape(n_steps)

        if 'num_steps' in extra:
            nsteps = np.asarray(extra['num_steps']).ravel()
            self._total_likelihood_evaluations += int(nsteps.sum())
            self.logger.info('number of integration steps: mean %.1f, max %d',
                             nsteps.mean(), nsteps.max())
        else:
            self._total_likelihood_evaluations += n_steps
        if 'accept_prob' in extra:
            self.logger.info('acceptance rate: mean %.3f',
                             float(np.asarray(extra['accept_prob']).mean()))
        self.logger.info('total likelihood evaluations(~): %d',
                         self._total_likelihood_evaluations)

        return samples, log_post


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
