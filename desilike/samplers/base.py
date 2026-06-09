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
from desilike.samples import MCSamples, diagnostics
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
        out = np.asarray(vfn(batch[None] if single else batch))
        return out[0] if single else out
    return batched_array


# ── BaseSampler ───────────────────────────────────────────────────────────────

class BaseSampler(ABC):
    """Abstract base class for all samplers."""

    logger = logging.getLogger('BaseSampler')

    @default_mpicomm
    def __init__(self, posterior, rng=None, mpicomm=None, directory=None,
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
        directory : str, Path, or None
            Save samples to this folder.  Default is ``None``.
        rescale : bool
            Internally normalise parameters so that their expected variation
            range is ~ unity (mirrors :class:`~desilike.profilers.base.BaseProfiler`).
            The sampler then explores the rescaled space while the posterior is
            evaluated in original space.  Default is ``False``.
        covariance : array_like, optional
            ``(ndim, ndim)`` covariance whose diagonal sets the rescaling scale.
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
        self.n_derived = int(sum(
            int(np.prod(p.shape)) if p.shape else 1
            for p in self.derived_params
        ))

        # ── rescaling transform ───────────────────────────────────────────────
        # Build _loc/_scale (flat, per scalar dimension) and a _transformed_params
        # collection whose priors/refs/proposals live in rescaled space.  The
        # sampler works in rescaled coordinates; _forward/_backward convert to and
        # from original parameter values at the posterior / storage boundaries.
        self._build_rescaling(rescale=rescale, covariance=covariance)

        # ── MPI communicator ─────────────────────────────────────────────────
        self.mpicomm = mpicomm

        # Compiled posterior graph.  The pooled evaluators below are built from
        # JAX-pure single-sample cores that call it; set_pool wraps each core in
        # jax.jit(jax.vmap(...)) so the pool always receives a batched function.
        self.posterior = posterior

        self.set_pool(mpicomm=self.mpicomm, batch_size=batch_size)

        # ── directory ────────────────────────────────────────────────────────
        if directory is not None:
            directory = Path(directory)
            if directory.suffix:
                raise ValueError('directory cannot have a suffix (must be a folder).')
            if self.mpicomm.rank == 0:
                directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

        if self.mpicomm.rank == 0:
            self.logger.info('Varied parameters: %s', self.varied_params.names())
            if self.directory is not None:
                self.logger.info('Samples will be written to: %s', self.directory)

        self.samples = None

        if self.directory is not None:
            try:
                self.read()
            except FileNotFoundError:
                pass

        self.set_rng(rng=rng)

    def _build_rescaling(self, rescale=False, covariance=None):
        """Build the ``_loc``/``_scale`` flat vectors and ``_transformed_params``.

        ``_loc[k]`` and ``_scale[k]`` are the centre and step of flat scalar element
        ``k`` (mirrors :class:`~desilike.profilers.base.BaseProfiler`).  ``_loc`` is the
        parameter centre (``value`` or ``ref.center()``); ``_scale`` is ``sqrt(diag(covariance))``
        when *covariance* is given, each parameter's ``ref.std()`` when *rescale* is set,
        or all-ones otherwise (no rescaling).
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
        if rescale:
            if covariance is not None:
                self._scale = np.sqrt(np.diag(np.asarray(covariance)))
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

        # Transformed collection: priors/refs expressed in rescaled space (the rescaled
        # step size is recovered from the transformed ref.std()).
        self._transformed_params = VariableCollection()
        for param, size, col in _param_sizes(self.varied_params):
            loc_p   = self._loc[col:col + size]
            scale_p = self._scale[col:col + size]
            param_copy = copy.copy(param)
            if not param.shape:
                loc_s, scale_s = float(loc_p[0]), float(scale_p[0])
                param_copy.prior = param.prior.affine_transform(loc=-loc_s / scale_s, scale=1. / scale_s)
                param_copy.ref   = param.ref.affine_transform(loc=-loc_s / scale_s, scale=1. / scale_s)
            else:
                loc_arr   = loc_p.reshape(param.shape)
                scale_arr = scale_p.reshape(param.shape)
                param_copy.prior = param.prior.affine_transform(loc=-loc_arr / scale_arr, scale=1. / scale_arr)
                param_copy.ref   = param.ref.affine_transform(loc=-loc_arr / scale_arr, scale=1. / scale_arr)
            self._transformed_params.set(param_copy)

    def _forward(self, x):
        """Rescaled → original space along the last axis: ``x * scale + loc``.

        JAX-safe (used inside jitted/vmapped cores); broadcasts over leading axes.
        """
        return jnp.asarray(x) * self._scale + self._loc

    def _backward(self, x):
        """Original → rescaled space along the last axis: ``(x - loc) / scale``.

        JAX-safe; broadcasts over leading axes.
        """
        return (jnp.asarray(x) - self._loc) / self._scale

    def _forward_dict(self, sample):
        """Map a ``{name: rescaled_value}`` dict to original parameter values.

        For samplers (e.g. blackjax) that carry positions as per-name dicts in the
        rescaled working space rather than as a flat vector.  JAX-safe.
        """
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

        ``prior_transform``, ``compute_prior``, ``compute_posterior`` and
        ``compute_likelihood`` are built from JAX-pure single-sample cores
        (``_*_one``) wrapped in ``jax.jit(jax.vmap(...))``.
        """
        self.pool = make_pool(mpicomm, batch_size=batch_size)
        specs = [('prior_transform',    self._prior_transform_one,    False),
                 ('compute_prior',      self._compute_prior_one,      False),
                 ('compute_posterior',  self._compute_posterior_one,  True),
                 ('compute_likelihood', self._compute_likelihood_one, True)]
        for name, core, returns_tuple in specs:
            setattr(self, name, self.pool.save_function(_batched(core, returns_tuple), name))

    def _prior_transform_one(self, sample):
        """Map a unit-cube sample ``(ndim,)`` to *rescaled* parameter space via each prior's PPF.

        The transformed priors' PPF already returns rescaled-space values, so the result
        is the sampler's working-space vector (``_forward`` maps it back to original).
        """
        result = []
        for param, size, col in _param_sizes(self._transformed_params):
            u_chunk = sample[col:col + size]
            result.append(jnp.atleast_1d(param.prior.ppf(u_chunk)))
        return jnp.concatenate(result)

    def _compute_prior_one(self, sample):
        """Return the log-prior for a single rescaled-space ``(ndim,)`` sample (Parameter priors only).

        The sample is mapped back to original space before evaluating the original priors,
        so the log-prior is consistent with the posterior's internal prior.

        Warning
        -------
        This is *not* necessarily the real prior, which may be more complex.
        """
        sample = _flat_to_dict(self._forward(sample), self.varied_params)
        return sum((param.prior.logpdf(sample[param.name])
                    for param in self.varied_params if param.prior is not None),
                   jnp.array(0.))

    def _compute_posterior_one(self, sample):
        """Return ``(log_posterior, derived_flat)`` for a single rescaled-space ``(ndim,)`` sample."""
        sample = _flat_to_dict(self._forward(sample), self.varied_params)
        if self.n_derived:
            log_post, derived_dict = self.posterior(sample, return_derived=True)
            derived_flat = jnp.concatenate([
                jnp.ravel(jnp.asarray(derived_dict.get(param.name, jnp.zeros(param.shape))))
                for param in self.derived_params])
        else:
            log_post = self.posterior(sample, return_derived=False)
            derived_flat = jnp.zeros(0)
        return log_post, derived_flat

    def _compute_likelihood_one(self, sample):
        """Return ``(log_likelihood, derived_flat)`` for a single ``(ndim,)`` sample.

        The log-likelihood is ``log_posterior − log_prior``.
        """
        log_prior = self._compute_prior_one(sample)
        log_post, derived = self._compute_posterior_one(sample)
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
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)
                self.samples.write(self.directory / 'samples.h5')

    def read(self):
        """Read sampler state from disk."""
        if self.pool.main:
            with open(self.directory / 'rng.json', 'r') as fstream:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = json.load(fstream)
                self.samples = MCSamples.read(self.directory / 'samples.h5')


# ── Static sampler ────────────────────────────────────────────────────────────

class StaticSampler(BaseSampler):
    """Base for samplers that pre-determine all sample points (grid, QMC, …)."""

    logger = logging.getLogger('StaticSampler')

    @abstractmethod
    def get_samples(self, **kwargs):
        """Return an ``(n_samples, ndim)`` array of points to evaluate."""
        pass

    def run(self, **kwargs):
        """Evaluate the posterior on the sample grid and return a MCSamples."""
        if self.pool.main:
            if self.samples is None:
                # get_samples returns original-space points; the cores and
                # array_to_samples work in the rescaled space, so map once here.
                grid      = np.asarray(self._backward(self.get_samples(**kwargs)))
                log_prior = np.array(self.pool.map(self.compute_prior, grid))
                results   = self.pool.map(self.compute_posterior, grid)
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

        if self.directory is not None:
            self.write()
        return self.samples


# ── Population sampler ────────────────────────────────────────────────────────

class PopulationSampler(BaseSampler):
    """Base for population-based samplers (dynesty, nautilus, …)."""

    logger = logging.getLogger('PopulationSampler')

    @abstractmethod
    def run_sampler(self, **kwargs):
        """Run the sampler; return ``(samples, derived, extras_dict)``."""
        pass

    def run(self, **kwargs):
        if self.pool.main:
            samples, derived, extras = self.run_sampler(**kwargs)
            result = self.array_to_samples(samples, derived, **extras)
            self.pool.stop_wait()
        else:
            result = None
            self.pool.wait()
        self.samples = result
        if self.directory is not None:
            self.write()
        return self.samples


# ── Markov chain sampler ───────────────────────────────────────────────────────

class MarkovChainSampler(BaseSampler):
    """Base for MCMC samplers running one or more independent chains."""

    logger = logging.getLogger('MarkovChainSampler')

    default_adaptation_steps = 0

    @default_mpicomm
    def __init__(self, posterior, nchains=1, chains=None, rng=None,
                 mpicomm=None, directory=None, rescale=False, covariance=None,
                 batch_size=None):
        """
        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int or sequence
            Number of independent chains, or explicit chain ids.  If an integer,
            ids are ``1, ..., nchains``.  If a sequence, ids are taken from it
            and used in checkpoint filenames, e.g. ``[4, 5, 6, 7]`` writes
            ``samples_4.h5`` through ``samples_7.h5``.  Default is 1.
        chains : list of MCSamples, optional
            If provided (at least on rank 0), continue from these chains.
        rng : numpy.random.Generator, int, or None
        mpicomm : MPI communicator, optional
        directory : str, Path, or None
        rescale : bool
            Normalise parameters to ~ unit variation range (see :class:`BaseSampler`).
        covariance : array_like, optional
            ``(ndim, ndim)`` covariance setting the rescaling scale.
        batch_size : int or None, optional
            Pool batching (see :class:`BaseSampler`).
        """
        self.mpicomm = mpicomm

        if not hasattr(self, '_samples'):
            self._chain = None

        # Broadcast explicit chain ids and whether input chains were supplied.
        input_chains = False
        if self.mpicomm.rank == 0:
            input_chains = chains is not None
            chain_ids = _normalize_chain_ids(nchains)
            if input_chains:
                if not isinstance(chains, (tuple, list)):
                    chains = [chains]
                if len(chains) != len(chain_ids):
                    raise ValueError(
                        f'Expected {len(chain_ids)} input chains, got {len(chains)}.'
                    )
        else:
            chain_ids = None

        input_chains, self.chain_ids = self.mpicomm.bcast((input_chains, chain_ids), root=0)
        self.nchains = len(self.chain_ids)

        super().__init__(posterior, rng=rng, mpicomm=mpicomm, directory=directory,
                         rescale=rescale, covariance=covariance, batch_size=batch_size)

        # Distribute pre-supplied chains to the owning ranks.
        if input_chains:
            for chain_idx, dest_rank in enumerate(self._pool_mains):
                samples = MCSamples.sendrecv(
                    chains[chain_idx] if self.mpicomm.rank == 0 else None,
                    source=0, dest=dest_rank, mpicomm=self.mpicomm)
                if self.mpicomm.rank == dest_rank:
                    self._chain = samples

        self.checks = []

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
        # Split communicator so each chain gets its own sub-communicator.
        if mpicomm.size > 1:
            sub_comm = mpicomm.Split(color=color, key=mpicomm.rank)
        else:
            sub_comm = mpicomm
        super().set_pool(mpicomm=sub_comm, batch_size=batch_size)
        # Collect the rank-0 process of each chain's sub-communicator.
        mains = self.mpicomm.allgather(self.mpicomm.rank if self.pool.main else None)
        self._pool_mains = [rank for rank in mains if rank is not None]
        self._ichain = color

    @abstractmethod
    def run_sampler(self, n_steps):
        pass

    @abstractmethod
    def adapt_sampler(self, steps):
        pass

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

                    # Drawn in original space; map to the sampler's rescaled working space.
                    batch_samples = np.asarray(self._backward(batch_samples))

                    results = self.pool.map(
                        self.compute_posterior,
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
        """Return current chain position as ``(samples, derived, log_post)``.

        ``samples`` is returned in the sampler's rescaled working space (stored
        parameter values are in original space and mapped via :meth:`_backward`).
        """
        # self._chain[name] returns a Variable; use ._value to get the raw array.
        walker_shape = self._chain.shape[1:]
        samples  = np.concatenate([
            np.asarray(self._chain[param.name])[-1].reshape(walker_shape + (-1,))
            for param in self.varied_params], axis=-1)
        derived  = np.concatenate([
            np.asarray(self._chain[param.name])[-1].reshape(walker_shape + (-1,))
            for param in self.derived_params], axis=-1) if self.n_derived else np.empty(0)
        log_post = np.asarray(self._chain.logposterior)[-1]
        return np.asarray(self._backward(samples)), np.array(derived), np.array(log_post)

    def extend(self, samples, derived, log_post):
        """Append new steps to the local chain."""
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
            converged = (
                len(self._chain) >= max_steps or
                (len(self._chain) >= min_steps and
                 len(self.checks) >= checks_passed and
                 all(self.checks[-checks_passed:]))
            )
        return all(self.mpicomm.allgather(converged))

    def run(self, burn_in=0.2, min_steps=0, max_steps=None, adaptation_steps=None,
            check_every=300, checks_passed=2, gelman_rubin=1.1, geweke=None, ess=None,
            save_every=300, max_init_attempts=100, concatenate=True):
        """Run the sampler until convergence and return the chains.

        Parameters
        ----------
        burn_in : float or int
            Fraction (or number) of steps to remove as burn-in.  Default is 0.2.
        min_steps : int
            Minimum number of steps.  Default is 0.
        max_steps : int or None
            Hard step limit.  Default is no limit.
        adaptation_steps : int or None
            Steps used for online adaptation.  ``None`` uses the sampler default.
        check_every : int
            How often (in steps) to run diagnostics.  Default is 300.
        checks_passed : int
            Number of consecutive passed checks required to stop.  Default is 2.
        gelman_rubin : float or None
            Gelman-Rubin threshold for convergence.  Default is 1.1.
        geweke : float or None
            Geweke threshold.  Default is ``None`` (not checked).
        ess : float or None
            ESS threshold.  Default is ``None`` (not checked).
        save_every : int
            Save checkpoint every this many steps.  Default is 300.
        max_init_attempts : int
            Maximum initialisation attempts per chain.  Default is 100.
        concatenate : bool
            Concatenate all chains before returning.  Default is ``True``.
        """
        self.initialize_samples(max_init_attempts=max_init_attempts)

        if self.directory is None:
            save_every = check_every  # skip intermediate saves

        if adaptation_steps is None:
            adaptation_steps = self.default_adaptation_steps
        self.adaptation_steps = adaptation_steps

        if adaptation_steps > 0:
            self.adapt_sampler(adaptation_steps)

        # Current step count across all chains.
        steps = min(self.mpicomm.allgather(
            len(self._chain) if self.pool.main else sys.maxsize))

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
            self.run_sampler(steps_to_take)

            if steps % check_every == 0:
                self.checks.append(self.check(
                    burn_in=burn_in, gelman_rubin=gelman_rubin,
                    geweke=geweke, ess=ess))

            if self.directory is not None and steps % save_every == 0:
                self.write()

        if self.directory is not None and steps % save_every != 0:
            self.write()

        if self.pool.main:
            self._chain = self._chain.remove_burnin(burn_in)

        all_chains = self.chains
        if concatenate and self.mpicomm.rank == 0:
            all_chains = MCSamples.concatenate(all_chains)
        return all_chains

    def write(self):
        if self.pool.main:
            with open(self.directory / f'rng_{self.chain_id}.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)
            self._chain.write(self.directory / f'samples_{self.chain_id}.h5')
        if self.mpicomm.rank == 0:
            with open(self.directory / 'checks.json', 'w') as fstream:
                json.dump(self.checks, fstream)

    def read(self):
        if self.pool.main:
            rng_path    = self.directory / f'rng_{self.chain_id}.json'
            samples_path  = self.directory / f'samples_{self.chain_id}.h5'
            if rng_path.exists():
                with open(rng_path, 'r') as fstream:
                    self.rng = np.random.default_rng()
                    self.rng.bit_generator.state = json.load(fstream)
            if samples_path.exists():
                self._chain = MCSamples.read(samples_path)
        checks_path = self.directory / 'checks.json'
        if checks_path.exists():
            with open(self.directory / 'checks.json', 'r') as fstream:
                self.checks = json.load(fstream)


# ── Ensemble sampler ──────────────────────────────────────────────────────────

class EnsembleSampler(MarkovChainSampler):
    """Base for ensemble samplers (emcee, zeus) that run ``nwalkers`` in parallel."""

    logger = logging.getLogger('EnsembleSampler')

    def __init__(self, posterior, nchains=1, chains=None, rng=None,
                 mpicomm=None, directory=None, nwalkers=None,
                 rescale=False, covariance=None, batch_size=None):
        super().__init__(posterior, nchains=nchains, chains=chains,
                         rng=rng, mpicomm=mpicomm, directory=directory,
                         rescale=rescale, covariance=covariance, batch_size=batch_size)
        if nwalkers is None and self._chain is not None:
            nwalkers = self._chain.shape[1] if len(self._chain.shape) > 1 else None
        nwalkers_all = self.mpicomm.allgather(nwalkers)
        for nw in nwalkers_all:
            if nw is not None:
                nwalkers = nw
                break
        self.nwalkers = int(nwalkers) if nwalkers is not None else None

    def initialize_samples(self, max_init_attempts=100):
        super().initialize_samples(max_init_attempts=max_init_attempts, shape=(self.nwalkers,))