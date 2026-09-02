"""Base classes for posterior samplers."""

import copy
import json
import math
import sys
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from ..base import build, CompiledGraph
from ..parameter import VariableCollection, _cumsize_params
from ..samples import MCSamples, Covariance, diagnostics
from ..distributed import default_mpicomm, get_mpicomm
from ..conditioning import AffineConditioner
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


def _normalize_sample_ids(nsamples):
    """Return explicit sample ids from an integer count or an iterable of ids.

    Parameters
    ----------
    nsamples : int or iterable
        If an integer, use sample ids ``1, ..., nsamples``.  Otherwise, use the
        provided values as explicit sample ids, e.g. ``[4, 5, 6, 7]``.

    Returns
    -------
    list
        Explicit sample ids, suitable for filenames such as ``samples_<id>.h5``.
    """
    if isinstance(nsamples, (int, np.integer)):
        if nsamples < 1:
            raise ValueError('nsamples must be >= 1.')
        return list(range(1, int(nsamples) + 1))

    sample_ids = list(nsamples)
    if not sample_ids:
        raise ValueError('nsamples cannot be an empty list.')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f'Duplicate sample ids in nsamples={sample_ids}.')
    return sample_ids


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
    cumsize = _cumsize_params(varied_params)
    result = {}
    for i, param in enumerate(varied_params):
        chunk = sample[cumsize[i]:cumsize[i + 1]]
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
    infrastructure: samples accumulation, convergence checks, MPI, rescaling
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

    # Maximum number of parallel runs this kernel can handle simultaneously.
    # ``1`` means purely sequential; ``None`` means the kernel handles any
    # number internally (e.g. via JAX vmap or numpyro num_chains).
    max_nparallel = 1

    def init(self, posterior_logpdf, rng, **context):
        """Initialise the kernel before sampling.

        Parameters
        ----------
        posterior_logpdf : callable
            JAX-pure function ``(n, ndim) → (n,)`` returning log-posterior values
            in *rescaled* space.  JAX-differentiable (built via ``jax.jit(jax.vmap(...))``)
            so gradient-based kernels can differentiate through it.
        rng : numpy.random.Generator
            Per-run random-number generator.
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

    # Population kernels always run one population at a time.
    max_nparallel = 1

    def reset_state(self):
        """Reset lazy-created sampler for a new independent run.  No-op by default."""

    def init(self, likelihood, prior, rng, **context):
        """Store callables and context for use in :meth:`run`.

        Called once on **all** MPI processes during
        :class:`~desilike.samplers.base.PopulationSampler` construction.

        Parameters
        ----------
        likelihood : tuple of (likelihood_logpdf, likelihood_logpdf_with_derived)
            Pool-saved callables returning ``log_l`` and ``(log_l, derived)``
            respectively for a single rescaled-space ``(ndim,)`` sample.
        prior : tuple of (prior_logpdf, prior_ppf, prior_rvs, prior_bounds)
            ``prior_logpdf``: pool-aware log-prior callable.
            ``prior_ppf``: unit-hypercube → parameter-space transform.
            ``prior_rvs``: ``(size, rng)`` draws; see :meth:`BaseSampler.prior_rvs`.
            ``prior_bounds``: ``(ndim, 2)`` array of lower/upper bounds in rescaled space.

            All four describe the beta = 0 distribution, which is the proposal rather than
            the prior when one is set.
        rng : numpy.random.Generator
            Random-number generator (main process).
        **context : dict
            ``pool`` : Pool — MPI pool for distributing evaluations.
            ``ndim`` : int — dimensionality of the conditioned parameter space.
            ``output_dir`` : Path or None — checkpoint directory.
            ``proposal`` : object or None — the proposal that replaced the prior, if any.
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

    #: Random-number generator, injected by :class:`StaticSampler` before :meth:`get_samples`.
    rng = None

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
                 conditioner=None, batch_size=None):
        """
        Parameters
        ----------
        posterior : CompiledGraph or Calculator
            Compiled pipeline returning the log-posterior.  A calculator is compiled here.
        rng : numpy.random.Generator, int, or None
            Random number generator.  Default is ``None``.
        mpicomm : MPI communicator, optional
            Communicator for pool parallelism.  Defaults to
            ``desilike.mpi.COMM_WORLD``.
        output_dir : str, Path, or None
            Save samples to this folder.  Default is ``None``.
        conditioner : AffineConditioner or None
            Conditioning transform applied between original and working space.
            ``None`` (default) uses :class:`AffineConditioner` with no rescaling
            (identity transform).
        batch_size : int or None, optional
            Controls how the pool batches likelihood/posterior calls.
            ``None`` (default) — pass all tasks as one stacked array per rank.
            ``0`` — evaluate one task at a time (no batching).
            ``N > 0`` — group tasks into chunks of N.
        """
        # A Calculator is built here rather than being rejected -- but only the no-argument
        # form: `build(root, output=...)` is a choice about what the pipeline returns, and a
        # sampler cannot guess it. An already-built graph is taken as is: building runs the whole
        # pipeline once.
        posterior = posterior if isinstance(posterior, CompiledGraph) else build(posterior)

        # ── parameter sets ────────────────────────────────────────────────────
        self.varied_params = posterior.params.select(varied=True, derived=False)
        if not self.varied_params:
            raise ValueError('No varied parameters found in the posterior.')
        # Derived = pure derived outputs (logposterior etc.) + analytically solved params.
        self.derived_params = posterior.params.select(derived=True) + posterior.params.select(solved=True)
        # Flat count of derived scalar values (for array_to_samples bookkeeping)
        self.nderived = int(_cumsize_params(self.derived_params)[-1])

        # ── conditioner transform ─────────────────────────────────────────────
        if conditioner is None:
            conditioner = AffineConditioner()
        self.conditioner = conditioner
        self.conditioner.init(self.varied_params)
        self._proposal_custom = None  # sentinel: no proposal (the prior is the beta = 0 distribution)

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

    def _set_proposal(self, proposal):
        """Install *proposal* as the starting / annealing distribution for population kernels.

        The proposal replaces the prior as the beta = 0 distribution of the tempered path.
        This does **not** change the inferred posterior: kernels receive the likelihood as
        ``log_posterior - log_proposal`` (see :meth:`_likelihood_logpdf_one`), so the
        tempered target ``proposal * likelihood^beta`` equals the exact posterior at
        beta = 1 for *any* proposal, and the kernel's evidence estimate remains the true
        one.  The proposal only shapes the annealing path.

        **The proposal must over-cover the posterior.** Measured on an analytic Gaussian
        target (`test_pocomc_proposal_accuracy`): a proposal 1.5-2x wider than the
        posterior (shifted by 1 sigma) recovers moments to better than 0.06 sigma, while a
        proposal 30% *narrower* leaves 5-17% residual under-dispersion that PocoMC's
        rejuvenation does not repair.  Inflate: 1.5-2x the estimated covariance.

        Parameters
        ----------
        proposal : Covariance or object
            A :class:`~desilike.samples.Covariance` is wrapped into a
            :class:`~desilike.samplers.proposals.GaussianProposal` centred on
            ``proposal.center``, with the parameters absent from it keeping their own prior.
            Otherwise, any object with

            - ``logpdf(x)``: log-density of a flat ``(ndim,)`` point in original parameter
              space (varied-parameter order); must be JAX-traceable, and may be unnormalized
              (the constant cancels in ``log_posterior - log_proposal``),

            and at least one way of drawing from it:

            - ``ppf(u)``: unit-cube ``(ndim,)`` to original parameter space (required by
              nested samplers and PocoMC, which sample the initial population that way),
            - ``rvs(size, rng)``: ``(size, ndim)`` draws in original parameter space, for
              proposals with no closed-form inverse CDF -- a set of chain rows, say. Kernels
              that need a ``ppf`` raise a clear error when only ``rvs`` is available; see
              :meth:`prior_rvs`.
        """
        from ..samples import Covariance as _Covariance
        if isinstance(proposal, _Covariance):
            from .proposals import GaussianProposal, ProductProposal
            if not any(param.name in proposal for param in self.varied_params):
                raise ValueError('None of the varied parameters are present in the proposal Covariance.')
            # The product fills in the parameters the covariance does not cover with their priors.
            proposal = ProductProposal(GaussianProposal(proposal))
        if not callable(getattr(proposal, 'logpdf', None)):
            raise TypeError(f'proposal must be a Covariance or expose a callable {"logpdf"!r}, '
                            f'got {type(proposal)}')
        if not any(callable(getattr(proposal, name, None)) for name in ('ppf', 'rvs')):
            raise TypeError(f'proposal must expose a callable {"ppf"!r} or {"rvs"!r}, got {type(proposal)}')
        # Proposals from desilike.samplers.proposals bind themselves to the sampler's own
        # parameter layout here, so they never have to guess the flat ordering.
        init = getattr(proposal, 'init', None)
        if callable(init):
            init(self.varied_params)
            # A proposal binds only the parameters it covers, so a partial one would leave the
            # sampler feeding it vectors of the wrong width. Wrap it in a ProductProposal to
            # give the rest their priors.
            ndim = getattr(proposal, 'ndim', self.ndim)
            if ndim != self.ndim:
                raise ValueError(f'proposal {type(proposal).__name__} covers {ndim} of the '
                                 f'{self.ndim} varied dimensions; wrap it in a ProductProposal.')
        self._proposal_custom = proposal

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
        return int(_cumsize_params(self.varied_params)[-1])

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
        """Axis-aligned bounding box of the prior support in conditioned parameter space.

        Returns an ``(ndim, 2)`` array; column 0 is lower bounds, column 1 is upper bounds.
        Delegates to :meth:`AffineConditioner.prior_bounds`.
        """
        return self.conditioner.prior_bounds()

    def _prior_ppf_one(self, sample):
        """Map a unit-cube sample ``(ndim,)`` to *rescaled* parameter space via each prior's PPF.

        When a proposal is set it supplies the PPF instead, over whichever parameters it
        covers. Either way the result is transformed to the sampler's conditioned working
        space via :meth:`AffineConditioner.inverse`.
        """
        if self._proposal_custom is not None:
            ppf = getattr(self._proposal_custom, 'ppf', None)
            if not callable(ppf):
                raise NotImplementedError(
                    f'proposal {type(self._proposal_custom).__name__} exposes no ppf, so it cannot be '
                    'inverted from the unit cube. Use a kernel that draws through rvs (e.g. SMC).')
            return self.conditioner.inverse(jnp.asarray(ppf(sample)))
        cumsize = _cumsize_params(self.varied_params)
        parts = [jnp.atleast_1d(param.prior.ppf(sample[cumsize[i]:cumsize[i + 1]]))
                 for i, param in enumerate(self.varied_params)]
        return self.conditioner.inverse(jnp.concatenate(parts))

    def prior_rvs(self, size, rng):
        """Draw *size* points from the prior (or from the proposal, when one is set).

        Returns points in the sampler's *rescaled* working space, so they can be fed
        straight to the pooled evaluators. Draws come from the proposal's own ``rvs``
        when it has one -- the only route for proposals with no closed-form inverse CDF,
        such as a set of chain rows -- and otherwise from the unit cube through
        :meth:`_prior_ppf_one`. Either way, points outside the hard prior box are
        rejected and redrawn: the target density is zero there, and samplers that
        logit-transform bounded dimensions turn such points into NaNs that poison their
        preconditioner.

        Parameters
        ----------
        size : int
            Number of points to draw.
        rng : numpy.random.Generator
            Random-number generator.

        Returns
        -------
        numpy.ndarray, shape ``(size, ndim)``
        """
        rvs = getattr(self._proposal_custom, 'rvs', None) if self._proposal_custom is not None else None
        if callable(rvs):
            def draw(ndraws):
                draws = np.asarray(rvs(ndraws, rng))
                if draws.shape != (ndraws, self.ndim):
                    raise ValueError(f'proposal.rvs returned shape {draws.shape}, '
                                     f'expected {(ndraws, self.ndim)}.')
                return np.asarray(self.conditioner.inverse(draws))
        else:
            def draw(ndraws):
                return np.asarray(self.prior_ppf(rng.random((ndraws, self.ndim))))

        bounds = np.asarray(self.prior_bounds)
        accepted, ndraws = [], 0
        for _ in range(100):
            candidates = draw(2 * size)
            mask = np.all((candidates >= bounds[:, 0]) & (candidates <= bounds[:, 1]), axis=1)
            accepted.append(candidates[mask])
            ndraws += int(mask.sum())
            if ndraws >= size:
                # Shuffle before truncating: draws are over-generated, and some proposals
                # return them in a meaningful order (systematic resampling of a chain hands
                # back ascending indices), so keeping the first `size` would keep a slice of
                # the chain rather than a sample of it.
                candidates = np.concatenate(accepted)
                return candidates[rng.permutation(len(candidates))[:size]]
        raise RuntimeError(f'prior_rvs: only {ndraws} of {size} draws fell inside the prior bounds '
                           'after 100 attempts.')

    def _prior_logpdf_one(self, sample):
        """Return the log-prior for a single rescaled-space ``(ndim,)`` sample.

        When a proposal is set it supplies the density instead. Either way the sample is
        mapped to original space via :meth:`AffineConditioner.forward` first.
        """
        x_orig = self.conditioner.forward(sample)
        if self._proposal_custom is not None:
            return self._proposal_custom.logpdf(x_orig)
        cumsize = _cumsize_params(self.varied_params)
        result = jnp.array(0.)
        for i, param in enumerate(self.varied_params):
            if param.prior is not None:
                chunk = x_orig[cumsize[i]:cumsize[i + 1]]
                chunk = chunk.reshape(param.shape) if param.shape else chunk[0]
                result = result + param.prior.logpdf(chunk)
        return result

    def _posterior_logpdf_one(self, sample):
        """Return ``log_posterior`` for a single rescaled-space ``(ndim,)`` sample."""
        return self.posterior(_flat_to_dict(self.conditioner.forward(sample), self.varied_params), return_derived=False)

    def _posterior_logpdf_with_derived_one(self, sample):
        """Return ``(log_posterior, derived_flat)`` for a single rescaled-space ``(ndim,)`` sample."""
        sample = _flat_to_dict(self.conditioner.forward(sample), self.varied_params)
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
        original parameter values via :meth:`AffineConditioner.forward` before being stored.
        """
        samples = np.asarray(self.conditioner.forward(samples))
        data = []
        # ── varied params ─────────────────────────────────────────────────────
        cumsize = _cumsize_params(self.varied_params)
        for i, param in enumerate(self.varied_params):
            slice_arr = samples[..., cumsize[i]:cumsize[i + 1]].reshape(samples.shape[:-1] + param.shape)
            data.append(param.clone(value=slice_arr))
        # ── derived params ────────────────────────────────────────────────────
        cumsize = _cumsize_params(self.derived_params)
        for i, param in enumerate(self.derived_params):
            slice_arr = derived[..., cumsize[i]:cumsize[i + 1]].reshape(derived.shape[:-1] + param.shape)
            data.append(param.clone(value=slice_arr))

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
                 output_dir=None, conditioner=None, batch_size=None):
        self.kernel = kernel
        super().__init__(posterior, rng=rng, mpicomm=mpicomm, output_dir=output_dir,
                         conditioner=conditioner, batch_size=batch_size)
        if self.kernel is not None:
            self.kernel.rng = self.rng

    def get_samples(self, **kwargs):
        """Return an ``(n_samples, ndim)`` array of points in original parameter space."""
        if self.kernel is not None:
            return self.kernel.get_samples(self.varied_params, **kwargs)
        raise NotImplementedError('Subclasses must implement get_samples() or provide a kernel.')

    def run(self, reuse=None, **kwargs):
        """Evaluate the posterior on the sample grid and return a MCSamples.

        Parameters
        ----------
        reuse : bool or None, optional
            Whether to return the samples already held by the sampler (read back from
            ``output_dir``, or produced by an earlier :meth:`run`) instead of evaluating
            again. ``None`` (default) reuses them only when this call passes no
            run-time options, so that ``run(samples=other)`` re-evaluates rather than
            silently returning a previous, unrelated result.
        """
        if reuse is None:
            reuse = not kwargs
        if self.pool.main:
            if self.samples is None or not reuse:
                # get_samples returns original-space points; the cores and
                # array_to_samples work in the rescaled space, so map once here.
                grid      = np.asarray(self.conditioner.inverse(self.get_samples(**kwargs)))
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
    """Kernel-based MCMC sampler running one or more independent sampling runs.

    Delegates the sampling algorithm to a :class:`~desilike.samplers.kernels.Kernel`
    and handles samples management, convergence diagnostics, MPI, rescaling, and I/O.

    Instantiate via the :func:`Sampler` factory rather than directly.
    """

    logger = logging.getLogger('MCMCSampler')

    @default_mpicomm
    def __init__(self, posterior, kernel, nparallel=1, rng=None,
                 mpicomm=None, output_dir=None, conditioner=None,
                 batch_size=None):
        """
        Parameters
        ----------
        posterior : CompiledGraph or Calculator
            Compiled pipeline returning the log-posterior.  A calculator is compiled here.
        kernel : Kernel
            Algorithm kernel, e.g. ``BlackjaxHMC()``, ``Emcee()``.
        nparallel : int or sequence
            Total number of independent sampling runs.  If an integer,
            sample ids are ``1, ..., nparallel``.  If a sequence, the values
            are used directly as sample ids in checkpoint filenames.
            The runs are distributed across MPI ranks and, when the kernel
            supports it, run in parallel within each rank.  Default is 1.
        rng : numpy.random.Generator, int, or None
        mpicomm : MPI communicator, optional
        output_dir : str, Path, or None
        conditioner : AffineConditioner or None
        batch_size : int or None, optional
        """
        self.kernel = kernel
        self.mpicomm = mpicomm

        if self.mpicomm.rank == 0:
            sample_ids = _normalize_sample_ids(nparallel)
        else:
            sample_ids = None

        self.sample_ids = self.mpicomm.bcast(sample_ids, root=0)
        self.nsamples = len(self.sample_ids)

        # Compute MPI / kernel parallelism structure.
        # _ngroups       : number of MPI sub-communicators (≤ mpicomm.size)
        # _runs_per_group: runs assigned to this sub-communicator (local, computed in set_pool)
        # _batch_nparallel: runs the kernel handles simultaneously per call (computed in set_pool)
        self._max_npar = getattr(kernel, 'max_nparallel', 1)
        self._ngroups = min(self.nsamples, self.mpicomm.size)
        # Pre-allocate with the upper-bound per-group count; trimmed to the actual local
        # _runs_per_group in set_pool once _igroup is known.
        max_runs = math.ceil(self.nsamples / self._ngroups)
        self._round_samples = [None] * max_runs
        # Kernel instances: one per batch; extended in run() as needed.
        self._kernels = [kernel]

        super().__init__(posterior, rng=rng, mpicomm=mpicomm, output_dir=output_dir,
                         conditioner=conditioner, batch_size=batch_size)

        self.checks = []
        self._thinning = 1
        self.kernel.init(
            (self.posterior_logpdf, self.posterior_logpdf_with_derived),
            self.rng,
            ndim=self.ndim,
            pool=self.pool,
            nsamples_parallel=self._batch_nparallel,
            param_shapes={param.name: param.shape for param in self.varied_params},
            nderived=self.nderived,
        )

    def set_rng(self, rng):
        if hasattr(self, '_group_rngs') and rng is None:
            pass
        else:
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            seed_seq = np.random.SeedSequence(rng.integers(0, 2**63, size=4))
            all_rngs = [np.random.default_rng(seed) for seed in seed_seq.spawn(self.nsamples)]
            self._group_rngs = all_rngs[self._group_start : self._group_start + self._runs_per_group]
            # Override with RNG states restored from disk (written by read()).
            if hasattr(self, '_saved_rng_states'):
                for local_idx, state in self._saved_rng_states.items():
                    self._group_rngs[local_idx] = np.random.default_rng()
                    self._group_rngs[local_idx].bit_generator.state = state
                del self._saved_rng_states
            self.rng = self._group_rngs[0]

    def set_pool(self, mpicomm, batch_size=None):
        color = mpicomm.rank * self._ngroups // mpicomm.size
        if mpicomm.size > 1:
            sub_comm = mpicomm.Split(color=color, key=mpicomm.rank)
        else:
            sub_comm = mpicomm
        super().set_pool(mpicomm=sub_comm, batch_size=batch_size)
        mains = self.mpicomm.allgather(self.mpicomm.rank if self.pool.main else None)
        self._pool_mains = [rank for rank in mains if rank is not None]
        self._igroup = color
        # Local distribution: group g gets base runs, plus one extra if g < extra.
        base = self.nsamples // self._ngroups
        extra = self.nsamples % self._ngroups
        self._runs_per_group = base + (1 if self._igroup < extra else 0)
        self._group_start = self._igroup * base + min(self._igroup, extra)
        # Trim pre-allocated slots to the actual local count.
        self._round_samples = self._round_samples[:self._runs_per_group]
        # Local batch parallelism: how many runs the kernel handles per call.
        self._batch_nparallel = (self._runs_per_group if self._max_npar is None
                                  else min(self._max_npar, self._runs_per_group))
        # Store per-pool-main run counts so chains / gather loops can use the correct count.
        all_rpg = self.mpicomm.allgather(self._runs_per_group if self.pool.main else 0)
        self._runs_per_group_all = {rank: all_rpg[rank] for rank in self._pool_mains}

    def _compute_derived(self, samples):
        """Compute derived parameters for a ``(n, ndim)`` batch of rescaled-space samples."""
        if not self.nderived:
            return np.zeros((len(samples), 0))
        results = self.pool.map(self.posterior_logpdf_with_derived, samples)
        return np.array([result[1] for result in results])

    def initialize_samples(self, max_init_attempts=100, shape=None, local_idx=0):
        """Draw initial position for local samples *local_idx* with finite log-posterior."""
        if max_init_attempts is None:
            max_init_attempts = sys.maxsize
        if shape is None:
            shape = ()
        shape = tuple(shape)
        total_size = int(np.empty(shape).size) if shape else 1

        if self.pool.main:
            if self._round_samples[local_idx] is None:
                all_samples, all_log_post, all_derived = [], [], []
                rng_local = self._group_rngs[local_idx]
                for _ in range(max_init_attempts):
                    batch_shape = shape or (1,)
                    batch_samples = np.zeros(batch_shape + (self.ndim,))
                    key = jax.random.PRNGKey(int(rng_local.integers(2**32)))
                    cumsize = _cumsize_params(self.varied_params)
                    for i, param in enumerate(self.varied_params):
                        sl = slice(cumsize[i], cumsize[i + 1])
                        if param.ref is not None and param.ref.is_proper():
                            key, subkey = jax.random.split(key)
                            drawn = np.asarray(param.ref.sample(subkey, shape=batch_shape))
                            batch_samples[..., sl] = drawn.reshape(batch_shape + (param.size,))
                        else:
                            batch_samples[..., sl] = np.asarray(param.value).ravel()

                    batch_samples = np.asarray(self.conditioner.inverse(batch_samples))

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
                        self._round_samples[local_idx] = self.array_to_samples(
                            final_samples, final_derived, logposterior=final_log_post)
                        break
            self.pool.stop_wait()
        else:
            self.pool.wait()

        if any(np.array(self.mpicomm.allgather(
                self._round_samples[local_idx] is None))[self._pool_mains]):
            raise ValueError(
                f'Could not find finite posterior after {max_init_attempts} attempts.')

    @property
    def _group_sample_ids(self):
        """Sample IDs assigned to this MPI group."""
        return self.sample_ids[self._group_start : self._group_start + self._runs_per_group]

    @property
    def chains(self):
        """Gather all local samples on rank 0 and return as a list of MCSamples."""
        gathered = []
        for source_rank in self._pool_mains:
            for local_idx in range(self._runs_per_group_all[source_rank]):
                samples = MCSamples.sendrecv(
                    self._round_samples[local_idx] if self.pool.main else None,
                    source=source_rank, dest=0, mpicomm=self.mpicomm)
                if self.mpicomm.rank == 0:
                    gathered.append(samples)
        return gathered if self.mpicomm.rank == 0 else None

    def _get_state(self, local_idx=0):
        """Return state of local samples *local_idx* as ``(position, derived, log_post)`` in rescaled space."""
        samples = self._round_samples[local_idx]
        walker_shape = samples.shape[1:]
        position = np.concatenate([
            np.asarray(samples[param.name])[-1].reshape(walker_shape + (-1,))
            for param in self.varied_params], axis=-1)
        derived  = np.concatenate([
            np.asarray(samples[param.name])[-1].reshape(walker_shape + (-1,))
            for param in self.derived_params], axis=-1) if self.nderived else np.empty(walker_shape + (0,))
        log_post = np.asarray(samples.logposterior)[-1]
        return np.asarray(self.conditioner.inverse(position)), np.array(derived), np.array(log_post)

    def _get_batched_state(self, local_indices=None):
        """Return stacked state for the given local runs as ``(positions, deriveds, log_posts)``."""
        if local_indices is None:
            local_indices = range(self._batch_nparallel)
        states = [self._get_state(k) for k in local_indices]
        return (np.stack([s[0] for s in states]),
                np.stack([s[1] for s in states]),
                np.array([s[2] for s in states]))

    @property
    def state(self):
        """Current sampler state in rescaled space."""
        if self._batch_nparallel == 1:
            return self._get_state(0)
        return self._get_batched_state()

    def extend(self, samples, derived, log_post, local_idx=0):
        """Append new steps to local samples *local_idx*."""
        if self._thinning > 1:
            samples  = samples[::self._thinning]
            derived  = derived[::self._thinning]
            log_post = log_post[::self._thinning]
        new_samples = self.array_to_samples(samples, derived, logposterior=log_post)
        if self._round_samples[local_idx] is None:
            self._round_samples[local_idx] = new_samples
        else:
            self._round_samples[local_idx] = MCSamples.concatenate(
                self._round_samples[local_idx], new_samples)

    def check(self, burn_in=0.2, gelman_rubin=1.1, geweke=None, ess=None, quiet=False):
        """Run convergence diagnostics; return True if all checks pass."""
        passed_all = True
        all_samples = self.chains
        if self.mpicomm.rank == 0:
            trimmed = [s.remove_burnin(burn_in) for s in all_samples]
            if not quiet:
                self.logger.info('Diagnostics:')

            nsplits = 4 // len(trimmed)
            gr_value = float(np.max(diagnostics.gelman_rubin(trimmed, method='diag', nsplits=nsplits)))
            try:
                geweke_value = float(np.max(diagnostics.geweke(trimmed, first=0.1, last=0.5)))
            except ValueError:
                geweke_value = float('inf')
            iact = diagnostics.integrated_autocorrelation_time(trimmed, check_valid='ignore')
            ess_value = float(np.mean([s.size for s in trimmed]) / np.max(iact))

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
            raw_steps = len(self._round_samples[0]) * self._thinning
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
        """Run the sampler until convergence and return the samples.

        Parameters
        ----------
        burn_in : float or int
            Fraction (or number) of steps to discard as burn-in.  Default is 0.2.
        min_steps : int
            Minimum number of steps before stopping.  Default is 0.
        max_steps : int or None
            Hard step limit.  Default is no limit.
        adaptation : dict or None
            Kwargs forwarded to kernel.adapt (e.g. ``{'steps': 500}``).
            ``None`` skips adaptation.  In sequential multi-run mode each
            run adapts independently from its own starting position.
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
            Maximum initialisation attempts per run.  Default is 100.
        concatenate : bool
            Concatenate all samples before returning.  Default is ``True``.
        thinning : int
            Keep every *thinning*-th sample in the output.  Default is 1.
        """
        self._thinning = int(thinning)

        if self.output_dir is None:
            save_every = check_every

        if max_steps is None:
            max_steps = sys.maxsize

        # Number of kernel calls per convergence-loop iteration.
        # Each batch covers _batch_nparallel runs; the last may be smaller.
        n_batches = math.ceil(self._runs_per_group / self._batch_nparallel)

        # Build additional kernel instances: one per batch beyond the first.
        # Each copy gets the RNG of the first run in its batch and a cleared
        # position state so adaptation (if any) starts from that run's position.
        if self.pool.main and len(self._kernels) < n_batches:
            for batch_idx in range(1, n_batches):
                batch_start = batch_idx * self._batch_nparallel
                k_new = copy.deepcopy(self._kernels[0])
                k_new._rng = self._group_rngs[batch_start]
                self._kernels.append(k_new)

        # Nominal number of steps per kernel call. The first call is short by the samples already
        # present (typically the initial position, i.e. one step), and a kernel that compiles its
        # sampling loop for a given length would otherwise recompile once the full-length batches
        # start. Kernels may ignore this.
        if self.pool.main:
            for kernel in self._kernels:
                kernel.nsteps_hint = min(check_every, save_every, max_steps)

        # Initialise all local runs.
        for local_idx in range(self._runs_per_group):
            if self._round_samples[local_idx] is None:
                self.initialize_samples(max_init_attempts=max_init_attempts, local_idx=local_idx)

        if adaptation is not None:
            if self.pool.main:
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self._batch_nparallel
                    self._kernels[batch_idx].adapt(self._get_state(batch_start), **adaptation)
                self.pool.stop_wait()
            else:
                self.pool.wait()

        steps = min(self.mpicomm.allgather(
            len(self._round_samples[0]) * self._thinning if self.pool.main else sys.maxsize))

        while not self.is_converged(min_steps=min_steps, max_steps=max_steps,
                                    checks_passed=checks_passed):
            steps_to_take = min(
                check_every - (steps % check_every),
                save_every  - (steps % save_every),
                max_steps   - steps,
            )
            steps += steps_to_take

            if self.pool.main:
                # Iterate over batches; each batch uses one kernel instance and covers
                # up to _batch_nparallel local runs.
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self._batch_nparallel
                    batch_end = min(batch_start + self._batch_nparallel, self._runs_per_group)
                    local_indices = list(range(batch_start, batch_end))

                    if len(local_indices) > 1:
                        state = self._get_batched_state(local_indices)
                    else:
                        state = self._get_state(local_indices[0])

                    samples_b, derived_b, extras_b = self._kernels[batch_idx].run(
                        steps_to_take, state)

                    # Normalize to batched form so the loop below is uniform.
                    if len(local_indices) == 1:
                        samples_b = samples_b[np.newaxis]
                        if derived_b is not None:
                            derived_b = derived_b[np.newaxis]
                        extras_b = {'logposterior': extras_b['logposterior'][np.newaxis]}

                    for k_off, local_idx in enumerate(local_indices):
                        samples_k = samples_b[k_off]
                        derived_k = derived_b[k_off] if derived_b is not None else None
                        if derived_k is None:
                            derived_k = self._compute_derived(
                                samples_k.reshape(-1, self.ndim)
                            ).reshape(samples_k.shape[:-1] + (-1,))
                        self.extend(samples_k, derived_k,
                                    extras_b['logposterior'][k_off],
                                    local_idx=local_idx)
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

        all_samples = self.chains
        if concatenate and self.mpicomm.rank == 0:
            all_samples = MCSamples.concatenate(all_samples)
        return all_samples

    def write(self):
        if self.pool.main:
            for local_idx, sample_id in enumerate(self._group_sample_ids):
                with open(self.output_dir / f'rng_{sample_id}.json', 'w') as fstream:
                    json.dump(self._group_rngs[local_idx].bit_generator.state, fstream)
                self._round_samples[local_idx].write(
                    self.output_dir / f'samples_{sample_id}.h5')
        if self.mpicomm.rank == 0:
            with open(self.output_dir / 'checks.json', 'w') as fstream:
                json.dump(self.checks, fstream)

    def read(self):
        self._saved_rng_states = {}
        if self.pool.main:
            for local_idx, sample_id in enumerate(self._group_sample_ids):
                rng_path     = self.output_dir / f'rng_{sample_id}.json'
                samples_path = self.output_dir / f'samples_{sample_id}.h5'
                if rng_path.exists():
                    with open(rng_path, 'r') as fstream:
                        self._saved_rng_states[local_idx] = json.load(fstream)
                if samples_path.exists():
                    self._round_samples[local_idx] = MCSamples.read(samples_path)
        checks_path = self.output_dir / 'checks.json'
        if checks_path.exists():
            with open(checks_path, 'r') as fstream:
                self.checks = json.load(fstream)


class EnsembleSampler(MCMCSampler):
    """Kernel-based ensemble MCMC infrastructure — delegates to a multi-walker :class:`~desilike.samplers.kernels.Kernel`.

    Instantiate via the :func:`Sampler` factory rather than directly.
    """

    logger = logging.getLogger('EnsembleSampler')

    def initialize_samples(self, max_init_attempts=100, local_idx=0):
        return super().initialize_samples(max_init_attempts=max_init_attempts,
                                          shape=(self.kernel.nwalkers,),
                                          local_idx=local_idx)


class PopulationSampler(BaseSampler):
    """Kernel-based infrastructure for nested / population samplers (dynesty, nautilus, pocomc, …).

    Supports *nparallel* independent runs: each MPI sub-communicator runs its
    assigned runs sequentially and writes ``samples_{sample_id}.h5`` per run.

    Instantiate via the :func:`Sampler` factory rather than directly.
    """

    logger = logging.getLogger('PopulationSampler')

    @default_mpicomm
    def __init__(self, posterior, kernel, nparallel=1, rng=None, mpicomm=None,
                 output_dir=None, conditioner=None, batch_size=None,
                 proposal=None):
        self.kernel = kernel
        self.mpicomm = mpicomm

        if self.mpicomm.rank == 0:
            sample_ids = _normalize_sample_ids(nparallel)
        else:
            sample_ids = None
        self.sample_ids = self.mpicomm.bcast(sample_ids, root=0)
        self.nsamples = len(self.sample_ids)

        self._ngroups = min(self.nsamples, self.mpicomm.size)
        # Pre-allocate with upper-bound; trimmed to actual local count in set_pool.
        max_runs = math.ceil(self.nsamples / self._ngroups)
        self._run_samples = [None] * max_runs

        if batch_size is None:
            batch_size = getattr(kernel, '_batch_size', None)
        super().__init__(posterior, rng=rng, mpicomm=mpicomm, output_dir=output_dir,
                         conditioner=conditioner, batch_size=batch_size)
        if proposal is not None:
            self._set_proposal(proposal)

        # The kernel checkpoint directory for the first (or only) local run.
        kernel_output_dir = self._kernel_output_dir(0)
        if kernel_output_dir is not None and self.pool.main:
            kernel_output_dir.mkdir(parents=True, exist_ok=True)
        self.kernel.init(
            (self.likelihood_logpdf, self.likelihood_logpdf_with_derived),
            (self.prior_logpdf, self.prior_ppf, self.prior_rvs, self.prior_bounds),
            self.rng,
            pool=self.pool, ndim=self.ndim, output_dir=kernel_output_dir,
            proposal=proposal,
        )

    def _kernel_output_dir(self, local_idx):
        """Return the checkpoint directory for local run *local_idx*."""
        if self.output_dir is None:
            return None
        sample_id = self._group_sample_ids[local_idx]
        return self.output_dir / f'samples_{sample_id}'

    @property
    def _group_sample_ids(self):
        """Sample IDs assigned to this MPI group."""
        return self.sample_ids[self._group_start : self._group_start + self._runs_per_group]

    def set_pool(self, mpicomm, batch_size=None):
        color = mpicomm.rank * self._ngroups // mpicomm.size
        if mpicomm.size > 1:
            sub_comm = mpicomm.Split(color=color, key=mpicomm.rank)
        else:
            sub_comm = mpicomm
        super().set_pool(mpicomm=sub_comm, batch_size=batch_size)
        mains = self.mpicomm.allgather(self.mpicomm.rank if self.pool.main else None)
        self._pool_mains = [rank for rank in mains if rank is not None]
        self._igroup = color
        # Local distribution: group g gets base runs, plus one extra if g < extra.
        base = self.nsamples // self._ngroups
        extra = self.nsamples % self._ngroups
        self._runs_per_group = base + (1 if self._igroup < extra else 0)
        self._group_start = self._igroup * base + min(self._igroup, extra)
        # Trim pre-allocated slots to the actual local count.
        self._run_samples = self._run_samples[:self._runs_per_group]
        # Store per-pool-main run counts so the gather loop can use the correct count.
        all_rpg = self.mpicomm.allgather(self._runs_per_group if self.pool.main else 0)
        self._runs_per_group_all = {rank: all_rpg[rank] for rank in self._pool_mains}

    def set_rng(self, rng):
        if hasattr(self, '_group_rngs') and rng is None:
            pass
        else:
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            seed_seq = np.random.SeedSequence(rng.integers(0, 2**63, size=4))
            all_rngs = [np.random.default_rng(seed) for seed in seed_seq.spawn(self.nsamples)]
            self._group_rngs = all_rngs[self._group_start : self._group_start + self._runs_per_group]
            self.rng = self._group_rngs[0]

    def run(self, concatenate=True, **kwargs):
        """Run all local population-sampler runs sequentially and return all samples."""
        for local_idx, sample_id in enumerate(self._group_sample_ids):
            if local_idx > 0:
                # Reset lazy-created sampler state for a fresh independent run.
                self.kernel.reset_state()
                self.kernel._rng = self._group_rngs[local_idx]
                new_output_dir = self._kernel_output_dir(local_idx)
                if new_output_dir is not None:
                    if self.pool.main:
                        new_output_dir.mkdir(parents=True, exist_ok=True)
                    self.kernel._output_dir = new_output_dir

            output = self.kernel.run(**kwargs)
            if self.pool.main:
                samples, derived, extras = output
                self._run_samples[local_idx] = self.array_to_samples(samples, derived, **extras)
            else:
                self._run_samples[local_idx] = None

        if self.output_dir is not None:
            self.write()

        # Gather all runs from all groups to rank 0.
        gathered = []
        for source_rank in self._pool_mains:
            for local_idx in range(self._runs_per_group_all[source_rank]):
                sample = MCSamples.sendrecv(
                    self._run_samples[local_idx] if self.pool.main else None,
                    source=source_rank, dest=0, mpicomm=self.mpicomm)
                if self.mpicomm.rank == 0:
                    gathered.append(sample)
        if self.mpicomm.rank == 0:
            if concatenate:
                return MCSamples.concatenate(gathered)
            return gathered
        return None

    def write(self):
        if self.pool.main:
            for local_idx, sample_id in enumerate(self._group_sample_ids):
                if self._run_samples[local_idx] is not None:
                    self._run_samples[local_idx].write(
                        self.output_dir / f'samples_{sample_id}.h5')

    def read(self):
        if self.pool.main:
            for local_idx, sample_id in enumerate(self._group_sample_ids):
                samples_path = self.output_dir / f'samples_{sample_id}.h5'
                if samples_path.exists():
                    self._run_samples[local_idx] = MCSamples.read(samples_path)


def Sampler(posterior, kernel, nparallel=1, rng=None, output_dir=None,
            conditioner=None, batch_size=None, proposal=None):
    """Factory creating the appropriate infrastructure class for *kernel*.

    Selects :class:`MCMCSampler`, :class:`EnsembleSampler`, :class:`PopulationSampler`,
    or :class:`StaticSampler` based on the kernel's ``_sampler_cls`` attribute and
    forwards all arguments.

    Parameters
    ----------
    posterior : CompiledGraph or Calculator
        Compiled pipeline returning the log-posterior scalar.  A calculator is compiled for you.
    kernel : Kernel, PopulationKernel, or StaticKernel
        Algorithm instance, e.g. ``BlackjaxHMC(step_size=1e-3)``, ``Emcee(nwalkers=32)``,
        ``Dynesty(dynamic=True)``, ``Grid()``, ``QMC()``, ``Importance()``.
    nparallel : int or sequence of int, optional
        Number of independent sampling runs.  When an integer, sample ids are
        ``1, …, nparallel``; when a sequence the values are used directly as ids
        in checkpoint filenames.  Runs are distributed across MPI ranks; when the
        kernel supports it (``max_nparallel > 1``) multiple runs are handled in
        parallel per rank.  Ignored for :class:`StaticSampler`.  Default is 1.
    rng : int, numpy.random.Generator, or None
        Random seed or generator.  ``None`` draws a fresh unseeded generator.
    output_dir : str, Path, or None
        Directory where per-run checkpoint files are written.  Created if absent.
        ``None`` disables checkpointing.
    conditioner : AffineConditioner or None
        Affine transform between original and working parameter space.
        ``None`` (default) uses the identity (no rescaling, no centering).
        Pass ``AffineConditioner(rescale=True)`` to whiten each parameter by its
        ``ref.std()``, or ``AffineConditioner(covariance=cov, rescale='full')``
        for full Cholesky whitening.
    batch_size : int or None, optional
        Controls how the pool batches posterior/likelihood calls.
        ``None`` — one stacked array per rank (default).
        ``0`` — one task at a time.
        ``N > 0`` — chunks of N tasks.
        Ignored for :class:`StaticSampler`.
    proposal : Covariance, object, or None, optional
        Starting / annealing distribution for :class:`PopulationSampler`
        kernels (PocoMC, Dynesty, Nautilus, …), replacing the prior as the beta = 0
        distribution of the tempered path.  A :class:`Covariance` gives a multivariate
        Gaussian centred on ``proposal.center`` with covariance ``proposal.value``
        (hard per-parameter prior bounds enforced on top); any other object must
        expose ``logpdf(x)`` and ``ppf(u)`` in original parameter space (see
        :meth:`BaseSampler._set_proposal`).  The inferred posterior and the evidence
        are unchanged for any proposal (kernels receive the likelihood as
        ``log_posterior - log_proposal``); the proposal only shortens the annealing
        path, and must over-cover the posterior (inflate a fitted covariance by
        1.5-2x).  Ignored for all other kernel types.

    Returns
    -------
    MCMCSampler or EnsembleSampler or PopulationSampler or StaticSampler
    """
    cls = _SAMPLER_REGISTRY[kernel._sampler_cls]
    if cls is PopulationSampler:
        return cls(posterior, kernel=kernel, nparallel=nparallel, rng=rng,
                   output_dir=output_dir, conditioner=conditioner,
                   batch_size=batch_size, proposal=proposal)
    if cls is StaticSampler:
        return cls(posterior, kernel=kernel, rng=rng, output_dir=output_dir,
                   conditioner=conditioner, batch_size=batch_size)
    return cls(posterior, kernel=kernel, nparallel=nparallel,
               rng=rng, output_dir=output_dir, conditioner=conditioner,
               batch_size=batch_size)


# Register here so kernel modules can look up these classes without a circular import.
_SAMPLER_REGISTRY['MCMCSampler'] = MCMCSampler
_SAMPLER_REGISTRY['EnsembleSampler'] = EnsembleSampler
_SAMPLER_REGISTRY['PopulationSampler'] = PopulationSampler
_SAMPLER_REGISTRY['StaticSampler'] = StaticSampler