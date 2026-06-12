"""Abstract Kernel base classes and sampler-class registry."""

import logging


class Kernel:
    """Abstract base class for MCMC kernels.

    A kernel encapsulates the sampling algorithm.  The surrounding
    :class:`~desilike.samplers.base.MCMCSampler` (or
    :class:`~desilike.samplers.base.EnsembleKernelSampler`) owns all
    infrastructure: chain accumulation, convergence checks, MPI, rescaling
    and directory I/O.

    Kernels are stateful: :meth:`init` must be called once before any
    :meth:`run` call, and the kernel retains its internal state (current
    position, adapted parameters, etc.) across subsequent :meth:`run` calls.

    Kernels operate entirely in *rescaled* space (the space defined by
    :attr:`~desilike.samplers.base.BaseSampler._loc` and
    :attr:`~desilike.samplers.base.BaseSampler._scale`).
    """

    logger = logging.getLogger('Kernel')

    # Infrastructure class to use when wrapped by the Sampler factory.
    # Override to 'EnsembleKernelSampler' for ensemble/multi-walker kernels.
    _sampler_cls = 'MCMCSampler'

    def init(self, logposterior, params, rng, **context):
        """Initialise the kernel before sampling.

        Parameters
        ----------
        logposterior : callable
            JAX-compatible function ``{name: jax_array} -> float`` returning the
            log-posterior in *rescaled* space.
        params : list of str
            Varied parameter names, in the order they appear in the flat array.
        rng : numpy.random.Generator
            Per-chain random-number generator.
        **context : dict
            Extra information provided by the sampler:

            ``ndim`` : int
                Total flat size of the parameter vector.
            ``param_shapes`` : dict[str, tuple]
                Shape of each parameter (scalar → ``()``).
            ``initial_position`` : dict[str, array]
                Starting position in rescaled space.

        """
        raise NotImplementedError

    def run(self, n_steps):
        """Draw ``n_steps`` posterior samples.

        The kernel updates its own internal state on every call so that
        consecutive calls continue from where the previous one left off.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.

        Returns
        -------
        samples : numpy.ndarray, shape ``(n_steps, *walker_shape, ndim)``
            Posterior samples in rescaled space.
        log_post : numpy.ndarray, shape ``(n_steps, *walker_shape)``
            Log-posterior values.

        """
        raise NotImplementedError

    def adapt(self, **kwargs):
        """Run warmup / adaptation.  No-op by default.

        Parameters
        ----------
        steps : int
            Number of warmup steps (required by most adaptive kernels).
        **kwargs : dict
            Kernel-specific adaptation options.

        """

    @property
    def walker_shape(self):
        """Extra shape dimensions between ``n_steps`` and ``ndim``.

        ``()`` for point samplers; ``(nwalkers,)`` for ensemble samplers.
        The surrounding sampler uses this to reshape samples and derived
        arrays correctly.
        """
        return ()


class NestedKernel:
    """Abstract base class for nested / population sampling kernels.

    Unlike :class:`Kernel` (MCMC), these kernels receive the sampling
    callables at *run time* rather than at :meth:`init` time, so that
    the infrastructure class can materialise batched / pool-aware versions
    before each run.

    A :class:`NestedKernel` subclass should override :meth:`run` and set
    ``_sampler_cls = 'NestedSampler'`` (the default).
    """

    logger = logging.getLogger('NestedKernel')
    _sampler_cls = 'NestedSampler'

    def run(self, loglikelihood, logprior, prior_transform, **kwargs):
        """Run the sampler and return all posterior samples.

        Called on **all** MPI processes (both main and workers).  The
        implementation is responsible for the main/worker split, including
        calling ``pool.stop_wait()`` on the main process and
        ``pool.wait()`` (or ``wait_many``) on workers before returning.

        Parameters
        ----------
        loglikelihood : callable
            Batched log-likelihood ``(N, ndim) → list[(log_l, derived)]`` or
            ``(N, ndim) → list[log_l]`` depending on whether derived
            parameters are present.  Pool-aware (already wrapped).
        logprior : callable
            Batched log-prior ``(N, ndim) → list[log_prior]``.  Pool-aware.
        prior_transform : callable
            Unit-hypercube to parameter-space transform ``(N, ndim) → (N, ndim)``.
            Pool-aware.
        pool : Pool
            MPI pool for distributing evaluations.
        rng : numpy.random.Generator
            Random-number generator for the main process.
        ndim : int
            Dimensionality of the parameter space.
        directory : Path or None
            Checkpoint directory.
        **kwargs : dict
            Run-time options forwarded to the underlying sampler's ``run``
            method.

        Returns
        -------
        samples : numpy.ndarray, shape ``(n_samples, ndim)``
            Samples in rescaled parameter space.  **Main process only**;
            workers return ``None``.
        derived : numpy.ndarray, shape ``(n_samples, n_derived)``
            Derived-parameter values stacked column-wise.
        extras : dict
            At minimum ``aweight`` (importance weights) and ``logposterior``.

        """
        raise NotImplementedError


# Registry mapping _sampler_cls strings to the actual sampler classes.
# Populated by base.py when MCMCSampler / EnsembleKernelSampler / NestedSampler are defined.
_SAMPLER_REGISTRY = {}
