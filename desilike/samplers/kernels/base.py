"""Abstract Kernel base class and sampler-class registry."""

import logging


class Kernel:
    """Abstract base class for MCMC kernels.

    A kernel encapsulates the sampling algorithm.  The surrounding
    :class:`~desilike.samplers.base.MCMCSampler` (or
    :class:`~desilike.samplers.base.PopulationSampler`) owns all
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
    # Override to 'PopulationSampler' for ensemble/multi-walker kernels.
    _sampler_cls = 'MCMCSampler'

    def init(self, logposterior, loglikelihood, logprior, params, rng, **context):
        """Initialise the kernel before sampling.

        Parameters
        ----------
        logposterior : callable
            JAX-compatible function ``{name: jax_array} -> float`` returning the
            log-posterior in *rescaled* space.
        loglikelihood : callable or None
            Same signature; may be ``None`` for pure MCMC kernels.
        logprior : callable or None
            Same signature; may be ``None``.
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


# Registry mapping _sampler_cls strings to the actual sampler classes.
# Populated by base.py when MCMCSampler / PopulationSampler are defined.
_SAMPLER_REGISTRY = {}
