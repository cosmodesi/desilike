"""Base classes for posterior samplers."""

import copy
import json
import sys
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from desilike.parameter import VariableCollection
from desilike.samples import MCSamples, Covariance, diagnostics
from desilike.distributed import default_mpicomm, get_mpicomm
from .pool import make_pool


# ── module-level helpers ──────────────────────────────────────────────────────

def update_kwargs(user_kwargs, sampler_name, **desilike_kwargs):
    """Return *user_kwargs* updated with *desilike_kwargs*, warning on conflicts."""
    kwargs = user_kwargs.copy()
    for key, value in desilike_kwargs.items():
        if key in user_kwargs:
            warnings.warn(f"Keyword argument '{key}' passed to {sampler_name} is overwritten.")
        kwargs[key] = value
    return kwargs


def _param_sizes(varied_params):
    """Return a list of ``(param, size, col_start)`` for each varied parameter.

    Parameters
    ----------
    varied_params : VariableCollection

    Returns
    -------
    list of (Parameter, int, int)
        Each tuple contains the parameter, its flat scalar size, and its
        starting column index in the flat parameter vector.
    """
    result = []
    col = 0
    for param in varied_params:
        size = int(np.prod(param.shape)) if param.shape else 1
        result.append((param, size, col))
        col += size
    return result


def _normalize_chain_ids(nchains):
    """Return explicit chain ids from an integer count or an iterable of ids.

    Parameters
    ----------
    nchains : int or iterable
        If an integer, use chain ids ``1, ..., nchains``.  Otherwise, use the
        provided values as explicit chain ids, e.g. ``[4, 5, 6, 7]``.

    Returns
    -------
    list
        Explicit chain ids, suitable for filenames such as ``samples_<id>.h5``.
    """
    if isinstance(nchains, (int, np.integer)):
        if nchains < 1:
            raise ValueError('nchains must be >= 1.')
        return list(range(1, int(nchains) + 1))

    chain_ids = list(nchains)
    if not chain_ids:
        raise ValueError('nchains cannot be an empty list.')
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError(f'Duplicate chain ids in nchains={chain_ids}.')
    return chain_ids


def _flat_to_dict(sample, varied_params):
    """Convert a flat ``(ndim,)`` array to a ``{name: shaped_array}`` dict.

    Parameters
    ----------
    sample : numpy.ndarray, shape (ndim,)
    varied_params : VariableCollection

    Returns
    -------
    dict
        Maps each parameter name to an array of shape ``param.shape``, or a
        scalar when ``param.shape`` is empty.  The values keep the dtype of
        *sample* (so this stays JAX-traceable when *sample* is a tracer).
    """
    result = {}
    for param, size, col in _param_sizes(varied_params):
        chunk = sample[col:col + size]
        result[param.name] = chunk.reshape(param.shape) if param.shape else chunk[0]
    return result


def _batched(core, returns_tuple):
    """Wrap a single-sample core into a batched evaluator.

    The vectorized pool calls the returned function once on a stacked
    ``(N, ndim)`` batch and iterates the result, so it yields ``N``
    per-sample results: an ``(N, ...)`` array for the array-returning cores,
    or a list of ``N`` ``(log, derived)`` tuples for the posterior/likelihood
    cores.  A single ``(ndim,)`` sample is also accepted (promoted to a batch
    of one, leading axis squeezed on return) so callers that evaluate one
    point at a time outside the pool — e.g. mhmcmc's standalone sampler —
    still work.
    """
    vfn = jax.jit(jax.vmap(core))
    if returns_tuple:
        def batched(batch):
            batch = jnp.asarray(batch)
            single = batch.ndim == 1
            log, derived = vfn(batch[None] if single else batch)
            results = list(zip(np.asarray(log), np.asarray(derived)))
            return results[0] if single else results
        return batched
    def batched_array(batch):
        batch = jnp.asarray(batch)
        single = batch.ndim == 1
        out = vfn(batch[None] if single else batch)
        return out[0] if single else out
    return batched_array


# ── Kernel ABCs ───────────────────────────────────────────────────────────────


class Kernel:
    """Abstract base class for MCMC kernels.

    A kernel encapsulates the sampling algorithm.  The surrounding
    :class:`MCMCSampler` (or :class:`EnsembleSampler`) owns all
    infrastructure: chain accumulation, convergence checks, MPI, rescaling
    and output_dir I/O.

    Kernels are stateful: :meth:`init` must be called once before any
    :meth:`run` call, and the kernel retains its internal state (current
    position, adapted parameters, etc.) across subsequent :meth:`run` calls.

    Kernels operate entirely in *rescaled* space (the space defined by
    :attr:`BaseSampler._loc` and :attr:`BaseSampler._scale`).
    """

    logger = logging.getLogger('Kernel')

    # Infrastructure class to use when wrapped by the Sampler factory.
    # Override to 'EnsembleSampler' for ensemble/multi-walker kernels.
    _sampler_cls = 'MCMCSampler'

    def init(self, posterior_logpdf, rng, **context):
        """Initialise the kernel before sampling.

        Parameters
        ----------
        posterior_logpdf : callable
            JAX-pure function ``(n, ndim) → (n,)`` returning log-posterior values
            in *rescaled* space.  JAX-differentiable (built via ``jax.jit(jax.vmap(...))``)
            so gradient-based kernels can differentiate through it.
        rng : numpy.random.Generator
            Per-chain random-number generator.
        **context : dict
            Extra information provided by the sampler:

            ``ndim`` : int
                Total flat size of the parameter vector.
            ``param_shapes`` : dict[str, tuple]
                Shape of each parameter (scalar → ``()``).

        """
        raise NotImplementedError

    def run(self, n_steps, initial_position=None):
        """Draw ``n_steps`` posterior samples.

        The kernel updates its own internal state on every call so that
        consecutive calls continue from where the previous one left off.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.
        initial_position : numpy.ndarray or None
            Starting position array in rescaled space — shape ``(ndim,)`` for
            point kernels, ``(nwalkers, ndim)`` for ensemble kernels.  Used only
            on the first call to initialise the kernel's internal state; ignored
            on subsequent calls (the kernel tracks its own state internally).

        Returns
        -------
        samples : numpy.ndarray, shape ``(n_steps, ndim)`` or ``(n_steps, nwalkers, ndim)``
            Posterior samples in rescaled space.
        log_post : numpy.ndarray, shape ``(n_steps,)`` or ``(n_steps, nwalkers)``
            Log-posterior values.

        """
        raise NotImplementedError

    def adapt(self, initial_position=None, **kwargs):
        """Run warmup / adaptation.  No-op by default.

        Parameters
        ----------
        initial_position : numpy.ndarray or None
            Starting position array in rescaled space.  Used only before the
            first step; ignored if the kernel already has an initialised state.
        steps : int
            Number of warmup steps (required by most adaptive kernels).
        **kwargs : dict
            Kernel-specific adaptation options.

        """


class PopulationKernel:
    """Abstract base class for nested / population sampling kernels.

    Subclasses receive the sampling callables once at :meth:`init` time and
    implement the sampling loop in :meth:`run`.

    A :class:`PopulationKernel` subclass should override :meth:`init` and
    :meth:`run`, and set ``_sampler_cls = 'PopulationSampler'`` (the default).
    """

    logger = logging.getLogger('PopulationKernel')
    _sampler_cls = 'PopulationSampler'

    def init(self, likelihood, prior, rng, **context):
        """Store callables and context for use in :meth:`run`.

        Called once on **all** MPI processes during
        :class:`~desilike.samplers.base.PopulationSampler` construction.

        Parameters
        ----------
        likelihood : tuple of (likelihood_logpdf, likelihood_logpdf_with_derived)
            Pool-saved callables returning ``log_l`` and ``(log_l, derived)``
            respectively for a single rescaled-space ``(ndim,)`` sample.
        prior : tuple of (prior_logpdf, prior_ppf, prior_bounds)
            ``prior_logpdf``: pool-aware log-prior callable.
            ``prior_ppf``: unit-hypercube → parameter-space transform.
            ``prior_bounds``: ``(ndim, 2)`` array of lower/upper bounds in rescaled space.
        rng : numpy.random.Generator
            Random-number generator (main process).
        **context : dict
            ``pool`` : Pool — MPI pool for distributing evaluations.
            ``ndim`` : int — dimensionality of the parameter space.
            ``output_dir`` : Path or None — checkpoint directory.
            ``params`` : VariableCollection — transformed parameter collection.
        """

    def run(self, **kwargs):
        """Run the sampler and return all posterior samples.

        Called on **all** MPI processes (both main and workers).  The
        implementation is responsible for the main/worker split, including
        calling ``pool.stop_wait()`` on the main process and
        ``pool.wait()`` (or ``wait_many``) on workers before returning.

        Parameters
        ----------
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


class StaticKernel:
    """Abstract base class for static (grid / QMC / importance) kernels.

    A static kernel pre-determines all sample points in the original parameter
    space *before* any posterior evaluations are made.  The surrounding
    :class:`StaticSampler` handles the pool evaluation, rescaling, and
    :class:`~desilike.samples.MCSamples` assembly.

    Subclasses must override :meth:`get_samples`.  Optionally override
    :meth:`post_process` to adjust the assembled samples after evaluation
    (e.g. importance-weight reweighting).
    """

    logger = logging.getLogger('StaticKernel')
    _sampler_cls = 'StaticSampler'

    def get_samples(self, varied_params, **kwargs):
        """Return an ``(n_samples, ndim)`` array of points in **original** parameter space.

        Parameters
        ----------
        varied_params : VariableCollection
            Varied parameters with prior information.
        **kwargs : dict
            Run-time options (e.g. ``grid``, ``size``, ``engine``).

        Returns
        -------
        numpy.ndarray, shape ``(n_samples, ndim)``
        """
        raise NotImplementedError

    def post_process(self, results, **kwargs):
        """Optional post-processing of the assembled :class:`~desilike.samples.MCSamples`.

        Called on all MPI ranks after the pool evaluation completes.
        The default implementation is a no-op identity.

        Parameters
        ----------
        results : MCSamples or None
            Assembled samples on the main rank; ``None`` on workers.
        **kwargs : dict
            Same keyword arguments that were passed to :meth:`get_samples`.

        Returns
        -------
        MCSamples or None
        """
        return results


# Registry mapping _sampler_cls strings to the actual sampler classes.
# Populated at the bottom of this module when the infrastructure classes are defined.
_SAMPLER_REGISTRY = {}


# ── BaseSampler ───────────────────────────────────────────────────────────────

class BaseSampler(ABC):
    """Abstract base class for all samplers."""

    logger = logging.getLogger('BaseSampler')

    @default_mpicomm
    def __init__(self, posterior, rng=None, mpicomm=None, output_dir=None,
                 rescale=False, covariance=None, batch_size=None):
        """
        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        rng : numpy.random.Generator, int, or None
            Random number generator.  Default is ``None``.
        mpicomm : MPI communicator, optional
            Communicator for pool parallelism.  Defaults to
            ``desilike.mpi.COMM_WORLD``.
        output_dir : str, Path, or None
            Save samples to this folder.  Default is ``None``.
        rescale : bool or {'diag', 'full'}
            Internally normalise parameters so that their expected variation
            range is ~ unity (mirrors :class:`~desilike.profilers.base.BaseProfiler`).
            The sampler explores the rescaled space while the posterior is evaluated
            in original space.  ``False`` (default) disables rescaling.
            ``True`` or ``'full'``: use the full covariance when *covariance* is a
            dense :class:`~desilike.parameter.Covariance` (Cholesky whitening), else
            its diagonal.  ``'diag'``: always use only the diagonal of *covariance*,
            even when *covariance* is dense.
        covariance : array_like or Covariance, optional
            Covariance used to set the rescaling scale.
            When ``None``, each parameter's ``ref.std()`` is used instead.
        batch_size : int or None, optional
            Controls how the pool batches likelihood/posterior calls.
            ``None`` (default) — pass all tasks as one stacked array per rank.
            ``0`` — evaluate one task at a time (no batching).
            ``N > 0`` — group tasks into chunks of N.
        """
        # ── parameter sets ────────────────────────────────────────────────────
        self.varied_params = posterior.params.select(varied=True, solved=False)
        if not self.varied_params:
            raise ValueError('No varied parameters found in the posterior.')
        # Derived = pure derived outputs (logposterior etc.) + analytically solved params.
        self.derived_params = posterior.params.select(derived=True) + posterior.params.select(solved=True)
        # Flat count of derived scalar values (for array_to_samples bookkeeping)
        self.nderived = int(sum(
            int(np.prod(p.shape)) if p.shape else 1
            for p in self.derived_params
        ))

        # ── rescaling transform ───────────────────────────────────────────────
        # Build _loc/_scale (flat, per scalar dimension) and a _transformed_params
        # collection whose priors/refs/proposals live in rescaled space.  The
        # sampler works in rescaled coordinates; _forward/_backward convert to and
        # from original parameter values at the posterior / storage boundaries.
        self._set_rescaling(rescale=rescale, covariance=covariance)

        # ── MPI communicator ─────────────────────────────────────────────────
        self.mpicomm = mpicomm

        # Compiled posterior graph.  The pooled evaluators below are built from
        # JAX-pure single-sample cores that call it; set_pool wraps each core in
        # jax.jit(jax.vmap(...)) so the pool always receives a batched function.
        self.posterior = posterior

        self.set_pool(mpicomm=self.mpicomm, batch_size=batch_size)

        # ── output_dir ────────────────────────────────────────────────────────
        if output_dir is not None:
            output_dir = Path(output_dir)
            if output_dir.suffix:
                raise ValueError('output_dir cannot have a suffix (must be a folder).')
            if self.mpicomm.rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        if self.mpicomm.rank == 0:
            self.logger.info('Varied parameters: %s', self.varied_params.names())
            if self.output_dir is not None:
                self.logger.info('Samples will be written to: %s', self.output_dir)

        self.samples = None

        if self.output_dir is not None:
            try:
                self.read()
            except FileNotFoundError:
                pass

        self.set_rng(rng=rng)

    def _set_rescaling(self, rescale=False, covariance=None):
        """Build the ``_loc``/``_scale`` flat vectors and ``_transformed_params``.

        ``_loc[k]`` and ``_scale[k]`` are the centre and step of flat scalar element
        ``k`` (mirrors :class:`~desilike.profilers.base.BaseProfiler`).  ``_loc`` is the
        parameter centre (``value`` or ``ref.center()``); ``_scale`` is ``sqrt(diag(covariance))``
        when *covariance* is given, each parameter's ``ref.std()`` when *rescale* is set,
        or all-ones otherwise (no rescaling).  When ``rescale`` is ``True`` or ``'full'``
        and *covariance* is a dense :class:`~desilike.parameter.Covariance`, Cholesky
        whitening is applied (``_L``/``_L_inv``).  ``rescale='diag'`` skips the Cholesky
        and always uses only the diagonal.
        """
        # Flat per-scalar layout: shape () → 1 element, shape (n,) → n elements.
        loc_parts = []
        for param, size, col in _param_sizes(self.varied_params):
            center = np.asarray(
                param.value if param.value is not None else param.ref.center()).ravel()
            if center.size == 1 and size > 1:
                center = np.full(size, float(center[0]))
            loc_parts.append(center.astype('f8'))
        self._loc = np.concatenate(loc_parts) if loc_parts else np.array([], dtype='f8')

        flat_size = self._loc.size
        self._L = self._L_inv = None
        if rescale:
            C_full = None
            if isinstance(covariance, Covariance):
                param_sizes_list = list(_param_sizes(self.varied_params))
                C_full = np.zeros((flat_size, flat_size), dtype='f8')
                # Fill the joint block from the Covariance for all known params at once.
                in_cov_indices = []
                params_in_cov = []
                for param, size, col in param_sizes_list:
                    if param.name in covariance:
                        in_cov_indices.extend(range(col, col + size))
                        params_in_cov.append(param)
                if params_in_cov:
                    sub = covariance.select(params_in_cov).value
                    ix = np.ix_(in_cov_indices, in_cov_indices)
                    C_full[ix] = sub
                # Fill diagonal for params absent from the Covariance.
                for param, size, col in param_sizes_list:
                    if param.name not in covariance:
                        std = param.ref.std()
                        if std is None or not np.isfinite(std) or std <= 0.:
                            raise ValueError(
                                f'Parameter {param.name!r}: cannot determine rescale scale from '
                                f'ref.std()={std!r}. Provide covariance or set a proper ref distribution.')
                        for k in range(size):
                            C_full[col + k, col + k] = float(std) ** 2
            elif covariance is not None:
                C_full = covariance
            if C_full is not None:
                self._scale = np.sqrt(np.diag(C_full))
                if rescale != 'diag' and np.any(C_full != np.diag(np.diag(C_full))):
                    _L = np.linalg.cholesky(C_full)
                    self._L = jnp.array(_L)
                    self._L_inv = jnp.array(np.linalg.inv(_L))
            else:
                scale_parts = []
                for param, size, col in _param_sizes(self.varied_params):
                    std = param.ref.std()
                    if std is None or not np.isfinite(std) or std <= 0.:
                        raise ValueError(
                            f'Parameter {param.name!r}: cannot determine rescale scale from '
                            f'ref.std()={std!r}. Provide covariance or set a proper ref distribution.')
                    scale_parts.append(np.full(size, float(std), dtype='f8'))
                self._scale = np.concatenate(scale_parts) if scale_parts else np.array([], dtype='f8')
        else:
            self._scale = np.ones(flat_size, dtype='f8')

        self._gauss_mu_orig = None  # sentinel: no Gaussian prior

        # Transformed collection: priors/refs expressed in rescaled space (the rescaled
        # step size is recovered from the transformed ref.std()).
        self._transformed_params = VariableCollection()
        for param, size, col in _param_sizes(self.varied_params):
            loc_p   = self._loc[col:col + size].reshape(param.shape or ())
            scale_p = self._scale[col:col + size].reshape(param.shape or ())
            prior = param.prior.affine_transform(loc=-loc_p / scale_p, scale=1. / scale_p)
            ref   = param.ref.affine_transform(loc=-loc_p / scale_p, scale=1. / scale_p)
            self._transformed_params.set(param.clone(prior=prior, ref=ref))

    def _set_gaussian_prior(self, prior):
        """Build a Gaussian prior from a :class:`~desilike.samples.Covariance` object.

        Parameters present in *prior* get a joint multivariate-Gaussian prior centred on
        ``prior.center`` with covariance ``prior.value``.  Parameters absent from *prior*
        keep their existing per-parameter prior (uniform / normal / …).

        Hard prior limits from each parameter's prior distribution are always enforced:
        the prior logpdf returns ``-inf`` outside those limits, and the PPF clips to them
        so that ``ppf(0)`` / ``ppf(1)`` return finite hard bounds (used by PocoMC and
        Dynesty to set the sampling volume).

        Must be called *after* :meth:`_set_rescaling` (depends on ``_loc`` and ``_scale``/
        ``_L``).  Stores the Gaussian parameters in original (pre-rescaling) space so that
        ``_prior_logpdf_one`` and ``_prior_ppf_one`` need only call ``_forward`` once.
        """
        from ..samples import Covariance as _Covariance
        if not isinstance(prior, _Covariance):
            raise TypeError(f'prior must be a Covariance instance, got {type(prior)}')

        param_sizes_list = list(_param_sizes(self.varied_params))

        # ── split varied params into Gaussian group and individual group ──────
        gauss_param_sizes = []   # (param, size, col) for params in prior
        indiv_param_sizes = []   # (param, size, col) for params not in prior
        for param, size, col in param_sizes_list:
            if param.name in prior:
                gauss_param_sizes.append((param, size, col))
            else:
                indiv_param_sizes.append((param, size, col))

        if not gauss_param_sizes:
            raise ValueError('None of the varied parameters are present in the prior Covariance.')

        # ── covariance and mean in original space ──────────────────────────────────────
        gauss_params_list = [param for param, size, col in gauss_param_sizes]
        gauss_prior = prior.select(gauss_params_list)
        C_gauss_orig = gauss_prior.value  # (n_gauss, n_gauss)
        mu_gauss_orig = gauss_prior.center  # n_gauss

        # ── Cholesky of C_gauss_orig ──────────────────────────────────────────
        try:
            L_gauss = np.linalg.cholesky(C_gauss_orig)
        except np.linalg.LinAlgError as exc:
            raise ValueError('prior covariance is not positive-definite.') from exc
        L_gauss_inv = np.linalg.inv(L_gauss)
        precision = L_gauss_inv.T @ L_gauss_inv  # (n_gauss, n_gauss)
        n_gauss = mu_gauss_orig.size
        # log det = 2 * sum(log(diag(L)))
        log_norm = 0.5 * n_gauss * np.log(2. * np.pi) + np.sum(np.log(np.diag(L_gauss)))

        # ── flat column indices in the ndim vector for Gaussian params ────────
        gauss_flat_cols = np.concatenate([np.arange(col, col + size)
                                          for param, size, col in gauss_param_sizes]).astype('i4')

        # ── store ─────────────────────────────────────────────────────────────
        self._gauss_mu_orig      = jnp.array(mu_gauss_orig)
        self._gauss_L_orig       = jnp.array(L_gauss)
        self._gauss_precision    = jnp.array(precision)
        self._gauss_log_norm     = float(log_norm)
        self._gauss_flat_cols    = gauss_flat_cols
        self._indiv_param_sizes  = indiv_param_sizes

    def _forward(self, x):
        """Rescaled → original space along the last axis.

        Diagonal: ``x * scale + loc``.
        Full Cholesky: ``x @ L.T + loc``.
        JAX-safe (used inside jitted/vmapped cores); broadcasts over leading axes.
        """
        if self._L is not None:
            return jnp.asarray(x) @ self._L.T + self._loc
        return jnp.asarray(x) * self._scale + self._loc

    def _backward(self, x):
        """Original → rescaled space along the last axis.

        Diagonal: ``(x - loc) / scale``.
        Full Cholesky: ``(x - loc) @ L_inv.T``.
        JAX-safe; broadcasts over leading axes.
        """
        if self._L is not None:
            return (jnp.asarray(x) - self._loc) @ self._L_inv.T
        return (jnp.asarray(x) - self._loc) / self._scale

    def _forward_dict(self, sample):
        """Map a ``{name: rescaled_value}`` dict to original parameter values.

        For samplers (e.g. blackjax) that carry positions as per-name dicts in the
        rescaled working space rather than as a flat vector.  JAX-safe.
        """
        if self._L is not None:
            flat_rescaled = jnp.concatenate([
                jnp.atleast_1d(jnp.ravel(jnp.asarray(sample[param.name])))
                for param, size, col in _param_sizes(self.varied_params)
            ])
            flat_original = flat_rescaled @ self._L.T + self._loc
            result = {}
            for param, size, col in _param_sizes(self.varied_params):
                value = flat_original[col:col + size]
                result[param.name] = value.reshape(param.shape) if param.shape else value[0]
            return result
        result = {}
        for param, size, col in _param_sizes(self.varied_params):
            scale = self._scale[col:col + size]
            loc   = self._loc[col:col + size]
            value = jnp.ravel(jnp.asarray(sample[param.name])) * scale + loc
            result[param.name] = value.reshape(param.shape) if param.shape else value[0]
        return result

    def set_rng(self, rng):
        """Set the random number generator."""
        if hasattr(self, 'rng') and rng is None:
            pass
        else:
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            self.rng = rng

    @property
    def ndim(self):
        """Total number of scalar dimensions across all varied parameters."""
        return int(sum(
            int(np.prod(param.shape)) if param.shape else 1
            for param in self.varied_params
        ))

    def set_pool(self, mpicomm, batch_size=None):
        """Create the pool and register the batched evaluators.

        Pool-dispatched attributes set here:
        ``prior_ppf``, ``prior_logpdf``, ``posterior_logpdf``, ``posterior_logpdf_with_derived``,
        ``likelihood_logpdf``, ``likelihood_logpdf_with_derived``.
        """
        self.pool = make_pool(mpicomm, batch_size=batch_size)
        specs = [('prior_ppf',                     self._prior_ppf_one,                       False),
                 ('prior_logpdf',                   self._prior_logpdf_one,                   False),
                 ('posterior_logpdf',               self._posterior_logpdf_one,               False),
                 ('posterior_logpdf_with_derived',  self._posterior_logpdf_with_derived_one,  True),
                 ('likelihood_logpdf',              self._likelihood_logpdf_one,              False),
                 ('likelihood_logpdf_with_derived', self._likelihood_logpdf_with_derived_one, True)]
        for name, core, returns_tuple in specs:
            setattr(self, name, self.pool.save_function(_batched(core, returns_tuple), name))

    @property
    def prior_bounds(self):
        """Axis-aligned bounding box of the prior support in whitened (rescaled) parameter space.

        Returns an ``(ndim, 2)`` array; column 0 is lower bounds, column 1 is upper bounds.
        Exact for diagonal (or no) rescaling.  For full Cholesky whitening, computed
        analytically as the linear image of the original-space parameter bounding box —
        O(ndim^2), no sampling required.
        """
        lo_orig = np.full(self.ndim, -np.inf)
        hi_orig = np.full(self.ndim, np.inf)
        for param, size, col in _param_sizes(self.varied_params):
            if param.prior is not None:
                lo_val, hi_val = param.prior.limits
                lo_orig[col:col + size] = lo_val
                hi_orig[col:col + size] = hi_val

        if self._L_inv is None:
            # Diagonal (or unit) scaling: y = (x - loc) / scale — exact, component-wise.
            lo_white = (lo_orig - self._loc) / self._scale
            hi_white = (hi_orig - self._loc) / self._scale
        else:
            # Full Cholesky: y = (x - loc) @ L_inv.T, so y_j = sum_k L_inv[j,k] (x_k - loc_k).
            # The bounding box of a linear image of the box [lo_orig, hi_orig] is:
            #   lo_white[j] = sum_k max(B[j,k], 0) * delta_lo[k] + min(B[j,k], 0) * delta_hi[k]
            # where B = L_inv and delta = (orig - loc).  np.where guards against 0 * ±inf = nan.
            delta_lo = lo_orig - self._loc
            delta_hi = hi_orig - self._loc
            B_pos = np.maximum(self._L_inv, 0.)
            B_neg = np.minimum(self._L_inv, 0.)
            lo_white = (np.where(B_pos == 0., 0., B_pos * delta_lo) + np.where(B_neg == 0., 0., B_neg * delta_hi)).sum(axis=-1)
            hi_white = (np.where(B_pos == 0., 0., B_pos * delta_hi) + np.where(B_neg == 0., 0., B_neg * delta_lo)).sum(axis=-1)

        return np.column_stack([lo_white, hi_white])

    def _prior_ppf_one(self, sample):
        """Map a unit-cube sample ``(ndim,)`` to *rescaled* parameter space via each prior's PPF.

        When a Gaussian prior is set (via :meth:`_set_gaussian_prior`), the first
        ``n_gauss`` unit-cube dimensions are mapped through the unconstrained joint
        Cholesky PPF of the Gaussian, and the remaining dimensions map each non-Gaussian
        param through its individual prior PPF.  Without a Gaussian prior, every param
        uses its individual prior PPF.  Either way the result is transformed to the
        sampler's rescaled working space via :meth:`_backward`.
        """
        if self._gauss_mu_orig is not None:
            n_gauss = self._gauss_flat_cols.size
            # Gaussian group: Cholesky PPF in original space (unconstrained).
            # Clip z to a large finite range before the matmul to avoid 0*inf=NaN when
            # L has zero entries (diagonal L) and u=0/1 gives z=±inf.
            z = jnp.clip(jax.scipy.stats.norm.ppf(sample[:n_gauss]), -1e38, 1e38)
            x_gauss = self._gauss_mu_orig + self._gauss_L_orig @ z
            x_orig = jnp.zeros(self.ndim).at[self._gauss_flat_cols].set(x_gauss)
            # Individual group: per-param PPF
            u_col = n_gauss
            for param, size, col in self._indiv_param_sizes:
                u_chunk = sample[u_col:u_col + size]
                x_orig = x_orig.at[col:col + size].set(jnp.atleast_1d(param.prior.ppf(u_chunk)))
                u_col += size
            return self._backward(x_orig)
        parts = []
        for param, size, col in _param_sizes(self.varied_params):
            u_chunk = sample[col:col + size]
            parts.append(jnp.atleast_1d(param.prior.ppf(u_chunk)))
        return self._backward(jnp.concatenate(parts))

    def _prior_logpdf_one(self, sample):
        """Return the log-prior for a single rescaled-space ``(ndim,)`` sample.

        When a Gaussian prior is set, evaluates the unconstrained multivariate-Gaussian
        logpdf for the Gaussian-group params and sums the individual per-param logpdfs
        for the remaining params.
        Without a Gaussian prior, evaluates each original prior's logpdf after mapping
        to original space via :meth:`_forward`.
        """
        x_orig = self._forward(sample)
        if self._gauss_mu_orig is not None:
            # Gaussian group: unconstrained multivariate Gaussian logpdf
            x_gauss = x_orig[self._gauss_flat_cols]
            d = x_gauss - self._gauss_mu_orig
            log_gauss = -0.5 * (d @ self._gauss_precision @ d) - self._gauss_log_norm
            # Individual group
            result = log_gauss
            for param, size, col in self._indiv_param_sizes:
                if param.prior is None:
                    continue
                chunk = x_orig[col:col + size]
                chunk = chunk.reshape(param.shape) if param.shape else chunk[0]
                result = result + param.prior.logpdf(chunk)
            return result
        result = jnp.array(0.)
        for param, size, col in _param_sizes(self.varied_params):
            if param.prior is None:
                continue
            chunk = x_orig[col:col + size]
            chunk = chunk.reshape(param.shape) if param.shape else chunk[0]
            result = result + param.prior.logpdf(chunk)
        return result

    def _posterior_logpdf_one(self, sample):
        """Return ``log_posterior`` for a single rescaled-space ``(ndim,)`` sample."""
        return self.posterior(_flat_to_dict(self._forward(sample), self.varied_params), return_derived=False)

    def _posterior_logpdf_with_derived_one(self, sample):
        """Return ``(log_posterior, derived_flat)`` for a single rescaled-space ``(ndim,)`` sample."""
        sample = _flat_to_dict(self._forward(sample), self.varied_params)
        if self.nderived:
            log_post, derived_dict = self.posterior(sample, return_derived=True)
            derived_flat = jnp.concatenate([
                jnp.ravel(jnp.asarray(derived_dict.get(param.name, jnp.zeros(param.shape))))
                for param in self.derived_params])
        else:
            log_post = self.posterior(sample, return_derived=False)
            derived_flat = jnp.zeros(0)
        return log_post, derived_flat

    def _likelihood_logpdf_one(self, sample):
        """Return ``log_likelihood`` for a single ``(ndim,)`` sample (no derived)."""
        return self._posterior_logpdf_one(sample) - self._prior_logpdf_one(sample)

    def _likelihood_logpdf_with_derived_one(self, sample):
        """Return ``(log_likelihood, derived_flat)`` for a single ``(ndim,)`` sample."""
        log_prior = self._prior_logpdf_one(sample)
        log_post, derived = self._posterior_logpdf_with_derived_one(sample)
        return log_post - log_prior, derived

    def _get_start(self, size=1):
        """Return a dict ``{name: array}`` sampled from each parameter's ref.

        Parameters
        ----------
        size : int
            Number of draws.  When 1 the batch axis is squeezed away.

        Returns
        -------
        dict
        """
        key = jax.random.PRNGKey(int(np.random.default_rng().integers(2**32)))
        start = {}
        for param in self.varied_params:
            if param.ref is not None and param.ref.is_proper():
                subkey, key = jax.random.split(key)
                value = np.asarray(param.ref.sample(subkey, shape=(size,) + param.shape))
            else:
                value = np.broadcast_to(np.asarray(param.value), (size,) + param.shape)
            start[param.name] = value.squeeze(0) if size == 1 else value
        return start

    def array_to_samples(self, samples, derived, **kwargs):
        """Convert parameter arrays to a :class:`~desilike.samples.MCSamples`.

        Parameters
        ----------
        samples : numpy.ndarray, shape (..., ndim)
            Values of varied parameters.  The last axis indexes parameters in
            the order of ``self.varied_params``.
        derived : numpy.ndarray, shape (..., n_derived)
            Flat derived-parameter values (same ordering as ``self.derived_params``).
        **kwargs
            Extra attributes to set on the returned samples (e.g.
            ``logposterior``, ``aweight``).

        Notes
        -----
        *samples* is in the sampler's rescaled working space; it is mapped back to
        original parameter values via :meth:`_forward` before being stored.
        """
        samples = np.asarray(self._forward(samples))
        data = []
        # ── varied params ─────────────────────────────────────────────────────
        for param, size, col in _param_sizes(self.varied_params):
            slice_arr  = samples[..., col:col + size].reshape(samples.shape[:-1] + param.shape)
            data.append(param.clone(value=slice_arr))
        # ── derived params ────────────────────────────────────────────────────
        col = 0
        for param in self.derived_params:
            size = int(np.prod(param.shape)) if param.shape else 1
            slice_arr  = derived[..., col:col + size].reshape(derived.shape[:-1] + param.shape)
            data.append(param.clone(value=slice_arr))
            col += size

        new_samples = MCSamples(data)
        for key, value in kwargs.items():
            setattr(new_samples, key, value)
        return new_samples

    def write(self):
        """Write sampler state to disk."""
        if self.pool.main:
            with open(self.output_dir / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)
                self.samples.write(self.output_dir / 'samples.h5')

    def read(self):
        """Read sampler state from disk."""
        if self.pool.main:
            with open(self.output_dir / 'rng.json', 'r') as fstream:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = json.load(fstream)
                self.samples = MCSamples.read(self.output_dir / 'samples.h5')


# ── Static sampler ────────────────────────────────────────────────────────────

class StaticSampler(BaseSampler):
    """Base for samplers that pre-determine all sample points (grid, QMC, …).

    Can be used directly by passing a :class:`~desilike.samplers.kernels.StaticKernel`
    instance (via the :func:`Sampler` factory), or subclassed with a concrete
    :meth:`get_samples` implementation (legacy pattern).
    """

    logger = logging.getLogger('StaticSampler')

    @default_mpicomm
    def __init__(self, posterior, kernel=None, rng=None, mpicomm=None,
                 output_dir=None, rescale=False, covariance=None, batch_size=None):
        self.kernel = kernel
        super().__init__(posterior, rng=rng, mpicomm=mpicomm, output_dir=output_dir,
                         rescale=rescale, covariance=covariance, batch_size=batch_size)

    def get_samples(self, **kwargs):
        """Return an ``(n_samples, ndim)`` array of points in original parameter space."""
        if self.kernel is not None:
            return self.kernel.get_samples(self.varied_params, **kwargs)
        raise NotImplementedError('Subclasses must implement get_samples() or provide a kernel.')

    def run(self, **kwargs):
        """Evaluate the posterior on the sample grid and return a MCSamples."""
        if self.pool.main:
            if self.samples is None:
                # get_samples returns original-space points; the cores and
                # array_to_samples work in the rescaled space, so map once here.
                grid      = np.asarray(self._backward(self.get_samples(**kwargs)))
                log_prior = np.array(self.pool.map(self.prior_logpdf, grid))
                results   = self.pool.map(self.posterior_logpdf_with_derived, grid)
                log_post  = np.array([result[0] for result in results])
                derived   = np.array([result[1] for result in results])
                self.samples = self.array_to_samples(
                    grid, derived,
                    logposterior=log_post,
                    aweight=np.exp(log_post - logsumexp(log_post)),
                )
                self.samples['logprior'] = log_prior
                self.pool.stop_wait()
        else:
            self.samples = None
            self.pool.wait()

        if self.samples is not None and self.kernel is not None:
            self.samples = self.kernel.post_process(self.samples, **kwargs)
        if self.output_dir is not None:
            self.write()
        return self.samples


# ── Kernel-based infrastructure ───────────────────────────────────────────────

class MCMCSampler(BaseSampler):
    """Kernel-based MCMC sampler running one or more independent chains.

    Delegates the sampling algorithm to a :class:`~desilike.samplers.kernels.Kernel`
    and handles chain management, convergence diagnostics, MPI, rescaling, and I/O.

    Instantiate via the :func:`Sampler` factory rather than directly.
    """

    logger = logging.getLogger('MCMCSampler')

    @default_mpicomm
    def __init__(self, posterior, kernel, nparallel=1, chains=None, rng=None,
                 mpicomm=None, output_dir=None, rescale=False, covariance=None,
                 batch_size=None):
        """
        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        kernel : Kernel
            Algorithm kernel, e.g. ``BlackjaxHMC()``, ``Emcee()``.
        nparallel : int or sequence
            Number of independent chains, or explicit chain ids.  If an integer,
            ids are ``1, ..., nparallel``.  If a sequence, the values are used
            directly as chain ids in checkpoint filenames.  Default is 1.
        chains : list of MCSamples, optional
            Pre-existing chains to resume from.  Default is ``None``.
        rng : numpy.random.Generator, int, or None
        mpicomm : MPI communicator, optional
        output_dir : str, Path, or None
        rescale : bool or {'diag', 'full'}
        covariance : array_like or Covariance, optional
        batch_size : int or None, optional
        """
        self.kernel = kernel
        self.mpicomm = mpicomm
        self._chain = None

        input_chains = False
        if self.mpicomm.rank == 0:
            input_chains = chains is not None
            chain_ids = _normalize_chain_ids(nparallel)
            if input_chains:
                if not isinstance(chains, (tuple, list)):
                    chains = [chains]
                if len(chains) != len(chain_ids):
                    raise ValueError(
                        f'Expected {len(chain_ids)} input chains, got {len(chains)}.')
        else:
            chain_ids = None

        input_chains, self.chain_ids = self.mpicomm.bcast((input_chains, chain_ids), root=0)
        self.nchains = len(self.chain_ids)

        super().__init__(posterior, rng=rng, mpicomm=mpicomm, output_dir=output_dir,
                         rescale=rescale, covariance=covariance, batch_size=batch_size)

        if input_chains:
            for chain_idx, dest_rank in enumerate(self._pool_mains):
                samples = MCSamples.sendrecv(
                    chains[chain_idx] if self.mpicomm.rank == 0 else None,
                    source=0, dest=dest_rank, mpicomm=self.mpicomm)
                if self.mpicomm.rank == dest_rank:
                    self._chain = samples

        self.checks = []
        self._thinning = 1
        self.kernel.init(
            (self.posterior_logpdf, self.posterior_logpdf_with_derived),
            self.rng,
            ndim=self.ndim,
            pool=self.pool,
            param_shapes={param.name: param.shape for param in self.varied_params},
        )

    def set_rng(self, rng):
        if hasattr(self, 'rng') and rng is None:
            pass
        else:
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            seed_seq = np.random.SeedSequence(rng.integers(0, 2**63, size=4))
            self.rng = [np.random.default_rng(seed) for seed in seed_seq.spawn(self.nchains)][self._ichain]

    def set_pool(self, mpicomm, batch_size=None):
        if self.nchains > mpicomm.size:
            raise ValueError(f'nchains={self.nchains} cannot exceed MPI size={mpicomm.size}')
        color = mpicomm.rank * self.nchains // mpicomm.size
        if mpicomm.size > 1:
            sub_comm = mpicomm.Split(color=color, key=mpicomm.rank)
        else:
            sub_comm = mpicomm
        super().set_pool(mpicomm=sub_comm, batch_size=batch_size)
        mains = self.mpicomm.allgather(self.mpicomm.rank if self.pool.main else None)
        self._pool_mains = [rank for rank in mains if rank is not None]
        self._ichain = color

    def _compute_derived(self, samples):
        """Compute derived parameters for a ``(n, ndim)`` batch of rescaled-space samples."""
        if not self.nderived:
            return np.zeros((len(samples), 0))
        results = self.pool.map(self.posterior_logpdf_with_derived, samples)
        return np.array([result[1] for result in results])

    def initialize_samples(self, max_init_attempts=100, shape=None):
        """Draw initial chain states with finite log-posterior."""
        if max_init_attempts is None:
            max_init_attempts = sys.maxsize
        if shape is None:
            shape = ()
        shape = tuple(shape)
        total_size = int(np.empty(shape).size) if shape else 1

        if self.pool.main:
            if self._chain is None:
                all_samples, all_log_post, all_derived = [], [], []
                for _ in range(max_init_attempts):
                    batch_shape = shape or (1,)
                    batch_samples = np.zeros(batch_shape + (self.ndim,))
                    key = jax.random.PRNGKey(int(self.rng.integers(2**32)))
                    for param, size, col in _param_sizes(self.varied_params):
                        if param.ref is not None and param.ref.is_proper():
                            key, subkey = jax.random.split(key)
                            drawn = np.asarray(param.ref.sample(subkey, shape=batch_shape))
                            batch_samples[..., col:col + size] = drawn.reshape(batch_shape + (size,))
                        else:
                            batch_samples[..., col:col + size] = np.asarray(param.value).ravel()

                    batch_samples = np.asarray(self._backward(batch_samples))

                    results = self.pool.map(
                        self.posterior_logpdf_with_derived,
                        batch_samples.reshape(total_size, self.ndim))
                    batch_log_post = np.array([result[0] for result in results])
                    batch_derived  = np.array([result[1] for result in results])

                    finite_mask = np.isfinite(batch_log_post)
                    all_samples  += batch_samples.reshape(total_size, self.ndim)[finite_mask].tolist()
                    all_log_post += batch_log_post[finite_mask].tolist()
                    all_derived  += batch_derived[finite_mask].tolist()

                    if len(all_samples) >= total_size:
                        final_samples  = np.array(all_samples[:total_size])
                        final_log_post = np.array(all_log_post[:total_size])
                        final_derived  = np.array(all_derived[:total_size])
                        if shape:
                            final_samples  = final_samples[np.newaxis]
                            final_log_post = final_log_post[np.newaxis]
                            final_derived  = final_derived[np.newaxis]
                        self._chain = self.array_to_samples(
                            final_samples, final_derived, logposterior=final_log_post)
                        break
            self.pool.stop_wait()
        else:
            self.pool.wait()

        if any(np.array(self.mpicomm.allgather(self._chain is None))[self._pool_mains]):
            raise ValueError(
                f'Could not find finite posterior after {max_init_attempts} attempts.')

    @property
    def chains(self):
        """Gather all local chains on rank 0 and return as a list."""
        gathered = []
        for source_rank in self._pool_mains:
            gathered.append(MCSamples.sendrecv(
                self._chain, source=source_rank, dest=0, mpicomm=self.mpicomm))
        return gathered if self.mpicomm.rank == 0 else None

    @property
    def chain_id(self):
        """Explicit id of the chain handled by this sampler rank."""
        return self.chain_ids[self._ichain]

    @property
    def state(self):
        """Current chain position as ``(samples, derived, log_post)`` in rescaled space."""
        walker_shape = self._chain.shape[1:]
        samples  = np.concatenate([
            np.asarray(self._chain[param.name])[-1].reshape(walker_shape + (-1,))
            for param in self.varied_params], axis=-1)
        derived  = np.concatenate([
            np.asarray(self._chain[param.name])[-1].reshape(walker_shape + (-1,))
            for param in self.derived_params], axis=-1) if self.nderived else np.empty(walker_shape + (0,))
        log_post = np.asarray(self._chain.logposterior)[-1]
        return np.asarray(self._backward(samples)), np.array(derived), np.array(log_post)

    def extend(self, samples, derived, log_post):
        """Append new steps to the local chain."""
        if self._thinning > 1:
            samples  = samples[::self._thinning]
            derived  = derived[::self._thinning]
            log_post = log_post[::self._thinning]
        new_samples = self.array_to_samples(samples, derived, logposterior=log_post)
        self._chain = MCSamples.concatenate(self._chain, new_samples)

    def check(self, burn_in=0.2, gelman_rubin=1.1, geweke=None, ess=None, quiet=False):
        """Run convergence diagnostics; return True if all checks pass."""
        passed_all = True
        all_chains = self.chains
        if self.mpicomm.rank == 0:
            trimmed = [chain.remove_burnin(burn_in) for chain in all_chains]
            if not quiet:
                self.logger.info('Diagnostics:')

            nsplits = 4 // len(trimmed)
            gr_value = float(np.max(diagnostics.gelman_rubin(trimmed, method='diag', nsplits=nsplits)))
            try:
                geweke_value = float(np.max(diagnostics.geweke(trimmed, first=0.1, last=0.5)))
            except ValueError:
                geweke_value = float('inf')
            iact = diagnostics.integrated_autocorrelation_time(trimmed, check_valid='ignore')
            ess_value = float(np.mean([chain.size for chain in trimmed]) / np.max(iact))

            for stat_name, threshold, is_upper_bound, value in [
                ('Gelman-Rubin',          gelman_rubin, True,  gr_value),
                ('Geweke',                geweke,       True,  geweke_value),
                ('Effective Sample Size', ess,          False, ess_value),
            ]:
                if not quiet:
                    self.logger.info(f'{stat_name}: {value:.3g}')
                if threshold is not None:
                    passed = value < threshold if is_upper_bound else value >= threshold
                    passed_all = passed_all and passed
                    if not quiet:
                        self.logger.info(
                            f'{value:.3g} {"<" if value < threshold else ">="} '
                            f'{threshold:.3g} ({"" if passed else "not "}passed)')
        return self.mpicomm.bcast(passed_all, root=0)

    def is_converged(self, min_steps=0, max_steps=sys.maxsize, checks_passed=10):
        """Return True when sampling should stop."""
        converged = True
        if self.pool.main:
            raw_steps = len(self._chain) * self._thinning
            converged = (
                raw_steps >= max_steps or
                (raw_steps >= min_steps and
                 len(self.checks) >= checks_passed and
                 all(self.checks[-checks_passed:]))
            )
        return all(self.mpicomm.allgather(converged))

    def run(self, burn_in=0.2, min_steps=0, max_steps=None, adaptation=None,
            check_every=300, checks_passed=2, gelman_rubin=1.1, geweke=None, ess=None,
            save_every=300, max_init_attempts=100, concatenate=True, thinning=1):
        """Run the sampler until convergence and return the chains.

        Parameters
        ----------
        burn_in : float or int
            Fraction (or number) of steps to discard as burn-in.  Default is 0.2.
        min_steps : int
            Minimum number of steps before stopping.  Default is 0.
        max_steps : int or None
            Hard step limit.  Default is no limit.
        adaptation : dict or None
            Kwargs forwarded to :meth:`adapt_sampler` (e.g. ``{'steps': 500}``).
            ``None`` skips adaptation.
        check_every : int
            Steps between convergence checks.  Default is 300.
        checks_passed : int
            Consecutive passed checks required to stop.  Default is 2.
        gelman_rubin : float or None
            Gelman-Rubin threshold.  Default is 1.1.
        geweke : float or None
            Geweke threshold.  Default is ``None`` (not checked).
        ess : float or None
            Effective sample size threshold.  Default is ``None`` (not checked).
        save_every : int
            Checkpoint interval in steps.  Default is 300.
        max_init_attempts : int
            Maximum initialisation attempts per chain.  Default is 100.
        concatenate : bool
            Concatenate all chains before returning.  Default is ``True``.
        thinning : int
            Keep every *thinning*-th sample in the output.  Default is 1.
        """
        self.initialize_samples(max_init_attempts=max_init_attempts)
        self._thinning = int(thinning)

        if self.output_dir is None:
            save_every = check_every

        if adaptation is not None:
            if self.pool.main:
                self.kernel.adapt(self.state, **adaptation)
                self.pool.stop_wait()
            else:
                self.pool.wait()

        steps = min(self.mpicomm.allgather(
            len(self._chain) * self._thinning if self.pool.main else sys.maxsize))

        if max_steps is None:
            max_steps = sys.maxsize

        while not self.is_converged(min_steps=min_steps, max_steps=max_steps,
                                    checks_passed=checks_passed):
            steps_to_take = min(
                check_every - (steps % check_every),
                save_every  - (steps % save_every),
                max_steps   - steps,
            )
            steps += steps_to_take
            if self.pool.main:
                samples, derived, extras = self.kernel.run(steps_to_take, self.state)
                if derived is None:
                    derived = self._compute_derived(
                        samples.reshape(-1, self.ndim)).reshape(samples.shape[:-1] + (-1,))
                self.extend(samples, derived, extras['logposterior'])
                self.pool.stop_wait()
            else:
                self.pool.wait()

            if steps % check_every == 0:
                self.checks.append(self.check(
                    burn_in=burn_in, gelman_rubin=gelman_rubin,
                    geweke=geweke, ess=ess))

            if self.output_dir is not None and steps % save_every == 0:
                self.write()

        if self.output_dir is not None and steps % save_every != 0:
            self.write()

        all_chains = self.chains
        if concatenate and self.mpicomm.rank == 0:
            all_chains = MCSamples.concatenate(all_chains)
        return all_chains

    def write(self):
        if self.pool.main:
            with open(self.output_dir / f'rng_{self.chain_id}.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)
            self._chain.write(self.output_dir / f'samples_{self.chain_id}.h5')
        if self.mpicomm.rank == 0:
            with open(self.output_dir / 'checks.json', 'w') as fstream:
                json.dump(self.checks, fstream)

    def read(self):
        if self.pool.main:
            rng_path     = self.output_dir / f'rng_{self.chain_id}.json'
            samples_path = self.output_dir / f'samples_{self.chain_id}.h5'
            if rng_path.exists():
                with open(rng_path, 'r') as fstream:
                    self.rng = np.random.default_rng()
                    self.rng.bit_generator.state = json.load(fstream)
            if samples_path.exists():
                self._chain = MCSamples.read(samples_path)
        checks_path = self.output_dir / 'checks.json'
        if checks_path.exists():
            with open(checks_path, 'r') as fstream:
                self.checks = json.load(fstream)


class EnsembleSampler(MCMCSampler):
    """Kernel-based ensemble MCMC infrastructure — delegates to a multi-walker :class:`~desilike.samplers.kernels.Kernel`.

    Instantiate via the :func:`Sampler` factory rather than directly.
    """

    logger = logging.getLogger('EnsembleSampler')

    def initialize_samples(self, max_init_attempts=100):
        return super().initialize_samples(max_init_attempts=max_init_attempts,
                                          shape=(self.kernel.nwalkers,))


class PopulationSampler(BaseSampler):
    """Kernel-based infrastructure for nested / population samplers (dynesty, nautilus, pocomc, …).

    Delegates entirely to a :class:`~desilike.samplers.kernels.base.PopulationKernel`; the kernel
    receives batched, pool-aware callables at run time.

    Instantiate via the :func:`Sampler` factory rather than directly.
    """

    logger = logging.getLogger('PopulationSampler')

    def __init__(self, posterior, kernel, rng=None, output_dir=None,
                 rescale=False, covariance=None, batch_size=None, prior=None):
        self.kernel = kernel
        if batch_size is None:
            batch_size = getattr(kernel, '_batch_size', None)
        super().__init__(posterior, rng=rng, output_dir=output_dir,
                         rescale=rescale, covariance=covariance, batch_size=batch_size)
        if prior is not None:
            self._set_gaussian_prior(prior)
        self.kernel.init(
            (self.likelihood_logpdf, self.likelihood_logpdf_with_derived),
            (self.prior_logpdf, self.prior_ppf, self.prior_bounds),
            self.rng,
            pool=self.pool, ndim=self.ndim, output_dir=self.output_dir,
            params=self._transformed_params,
        )

    def run(self, **kwargs):
        output = self.kernel.run(**kwargs)
        if self.pool.main:
            samples, derived, extras = output
            self.samples = self.array_to_samples(samples, derived, **extras)
        else:
            self.samples = None
        if self.output_dir is not None:
            self.write()
        return self.samples


def Sampler(posterior, kernel, nparallel=1, chains=None, rng=None, output_dir=None,
            rescale=False, covariance=None, batch_size=None, prior=None):
    """Factory creating the appropriate infrastructure class for *kernel*.

    Parameters
    ----------
    posterior : CompiledGraph
    kernel : Kernel, PopulationKernel, or StaticKernel
        Algorithm instance, e.g. ``BlackjaxHMC(step_size=1e-3)``, ``Emcee(nwalkers=32)``,
        ``Dynesty(dynamic=True)``, ``Grid()``, ``QMC()``, ``Importance()``.
    nparallel : int
        Number of independent chains (point MCMC) or independent ensemble runs (multi-walker).
        Ignored for :class:`PopulationSampler` and :class:`StaticSampler`.  Default is 1.
    chains : list of MCSamples or None
        Pre-existing chains to resume from.  Ignored for :class:`PopulationSampler`
        and :class:`StaticSampler`.  Default is ``None``.
    rng : int or numpy.random.Generator or None
    output_dir : str or Path or None
    rescale : bool or {'diag', 'full'}
    covariance : array_like or Covariance or None
    batch_size : int or None
    prior : Covariance or None
        Optional Gaussian prior for :class:`PopulationSampler` kernels (PocoMC, Dynesty,
        Nautilus, …).  The prior is a multivariate Gaussian centred on ``prior.center``
        with covariance ``prior.value``, automatically rescaled to the sampler's working
        space.  Hard bounds from each parameter's prior distribution are always enforced.
        Ignored for non-:class:`PopulationSampler` kernels.
    """
    cls = _SAMPLER_REGISTRY[kernel._sampler_cls]
    if cls is PopulationSampler:
        return cls(posterior, kernel=kernel, rng=rng, output_dir=output_dir,
                   rescale=rescale, covariance=covariance, batch_size=batch_size, prior=prior)
    if cls is StaticSampler:
        return cls(posterior, kernel=kernel, rng=rng, output_dir=output_dir,
                   rescale=rescale, covariance=covariance)
    return cls(posterior, kernel=kernel, nparallel=nparallel, chains=chains,
               rng=rng, output_dir=output_dir, rescale=rescale, covariance=covariance, batch_size=batch_size)


# Register here so kernel modules can look up these classes without a circular import.
_SAMPLER_REGISTRY['MCMCSampler'] = MCMCSampler
_SAMPLER_REGISTRY['EnsembleSampler'] = EnsembleSampler
_SAMPLER_REGISTRY['PopulationSampler'] = PopulationSampler
_SAMPLER_REGISTRY['StaticSampler'] = StaticSampler