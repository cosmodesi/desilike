"""BaseProfiler — core profiling logic shared by all concrete profilers."""

from __future__ import annotations

import copy
import logging
import dataclasses
import functools
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from ..parameter import Parameter, VariableCollection
from ..samples import Profiles
from ..distributed import get_mpicomm


# ── ProfilerState ─────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ProfilerState:
    """Immutable snapshot of one optimisation problem in *rescaled* space.

    Use ``dataclasses.replace(state, start=x0)`` to vary the starting point.

    ``start``, ``flat_bounds``, and ``flat_proposals`` all use the *flat*
    parameter layout: a vector param of shape ``(n,)`` occupies ``n``
    consecutive elements.  Scalar params occupy one element each.
    """
    chi2_fn:        Callable            # chi2(x_rescaled) → scalar
    grad_fn:        Callable | None     # grad_chi2, or None
    start:          np.ndarray          # shape (flat_size,) in rescaled space
    varied_params:  VariableCollection  # per-param metadata (names, shapes, …)
    flat_bounds:    list                # [(lo, hi), …] per flat element, rescaled
    flat_proposals: list                # [proposal, …]  per flat element, rescaled
    fast:           bool                # True → skip hesse / covariance


# ── flat-layout helpers ───────────────────────────────────────────────────────

def _build_best_from_x(x_rescaled, logpost, varied_params):
    """Convert a flat rescaled result vector into a ``Profiles.best`` dict.

    Parameters
    ----------
    x_rescaled : array_like, shape ``(flat_size,)``
        Parameter values in the *rescaled* optimisation space.
    logpost : float
        Log-posterior value at this point.
    varied_params : VariableCollection
        Parameters corresponding to the elements of *x_rescaled*.

    Returns
    -------
    dict
        Keys are parameter names plus ``'logpdf'``.
        Each value has shape ``(1,)`` for scalar params or
        ``(1, *param.shape)`` for non-scalar params.
    """
    x_arr  = np.asarray(x_rescaled)
    best   = {}
    offset = 0
    for param in varied_params:
        flat_size = int(np.prod(param.shape)) if param.shape else 1
        val_flat  = x_arr[offset:offset + flat_size]
        if param.shape:
            best[param.name] = val_flat.reshape((1,) + param.shape)
        else:
            best[param.name] = np.array([float(val_flat[0])])
        offset += flat_size
    best['logpdf'] = np.array([float(logpost)])
    return best


def _build_error_from_cov(cov, varied_params):
    """Convert a covariance matrix diagonal into an ``error`` dict.

    Parameters
    ----------
    cov : array_like, shape ``(flat_size, flat_size)``
        Covariance matrix in the *original* (un-rescaled) parameter space.
    varied_params : VariableCollection

    Returns
    -------
    dict
        Each entry has shape ``(1,)`` for scalar params or
        ``(1, *param.shape)`` for non-scalar params.
    """
    error_flat = np.sqrt(np.abs(np.diag(np.asarray(cov))))
    error  = {}
    offset = 0
    for param in varied_params:
        flat_size = int(np.prod(param.shape)) if param.shape else 1
        err_flat  = error_flat[offset:offset + flat_size]
        if param.shape:
            error[param.name] = err_flat.reshape((1,) + param.shape)
        else:
            error[param.name] = np.array([float(err_flat[0])])
        offset += flat_size
    return error


def _compute_flat_bounds(transformed_params):
    """Return per-flat-element ``(lo, hi)`` bounds in transformed space.

    Parameters
    ----------
    transformed_params : iterable of Parameter

    Returns
    -------
    list of (float, float)
        One entry per flat element (vector params contribute multiple entries).
    """
    bounds = []
    for param in transformed_params:
        flat_size = int(np.prod(param.shape)) if param.shape else 1
        lo_all, hi_all = param.prior.limits
        if flat_size == 1:
            bounds.append((float(lo_all), float(hi_all)))
        else:
            lo_arr = np.broadcast_to(np.asarray(lo_all).ravel(), (flat_size,))
            hi_arr = np.broadcast_to(np.asarray(hi_all).ravel(), (flat_size,))
            for lo, hi in zip(lo_arr, hi_arr):
                bounds.append((float(lo), float(hi)))
    return bounds


def _compute_flat_proposals(transformed_params):
    """Return per-flat-element proposal step size in transformed space.

    Parameters
    ----------
    transformed_params : iterable of Parameter

    Returns
    -------
    list of float
        One entry per flat element.  The step size is each parameter's ``ref.std()``.
    """
    proposals = []
    for param in transformed_params:
        flat_size = int(np.prod(param.shape)) if param.shape else 1
        std = param.ref.std()
        p_val = float(std) if (std is not None and np.isfinite(float(std))) else 1.0
        for _ in range(flat_size):
            proposals.append(p_val)
    return proposals


def _pool_map(mpicomm, fn, items):
    """Map *fn* over *items*, distributing the independent runs across *mpicomm*
    ranks (round-robin) and gathering results to every rank.

    Each rank already holds *fn* (no pickling/dispatch): rank ``r`` runs
    ``items[r::size]`` serially, then an ``allgather`` reassembles the full
    result list in original order, identical on all ranks.
    """
    items = list(items)
    if not items:
        return []
    if mpicomm.size == 1 or len(items) == 1:
        return [fn(item) for item in items]
    local_results = [fn(item) for item in items[mpicomm.rank::mpicomm.size]]
    results = [None] * len(items)
    for worker_rank, chunk in enumerate(mpicomm.allgather(local_results)):
        results[worker_rank::mpicomm.size] = chunk
    return results


# ── BaseProfiler ──────────────────────────────────────────────────────────────

class BaseProfiler:
    """Base class for likelihood profilers.

    Subclasses implement ``_maximize_one(state, **kwargs) -> Profiles``
    working entirely in *rescaled* parameter space.

    Parameters
    ----------
    likelihood : CompiledGraph
        Compiled pipeline whose ``__call__(params_dict)`` returns the
        log-posterior scalar.
    rng : np.random.Generator, optional
        Random number generator.  If ``None``, built from *seed*.
    seed : int, optional
        Seed for ``np.random.default_rng``.
    max_tries : int
        Maximum candidate draws when searching for a finite starting point.
    profiles : Profiles or path, optional
        Existing profiles to append new results to.
    ref_scale : float
        Rescale each parameter's reference distribution by this factor
        before sampling starting points.
    rescale : bool
        Internally normalise parameters so that their expected variation
        range is ~ unity.
    covariance : array_like, optional
        ``(flat_size, flat_size)`` covariance used to set the rescaling scale.
        When ``None``, each parameter's ``ref.std()`` is used instead.
    save_fn : str or Path, optional
        If given, profiles are written here after every public method.
    mpicomm : MPI communicator, optional
        Communicator over which independent ``maximize`` runs are distributed,
        and on whose rank 0 outputs are written. Defaults to
        :func:`desilike.distributed.get_mpicomm`.

    Notes on parallelism
    --------------------
    Parallelism is used only in ``maximize`` (outer loop over independent
    starts).  ``profile`` and ``grid`` run their inner optimisations
    serially to support warm-starting.

    Distribution is SPMD over ``mpicomm``: every rank runs ``maximize`` and its
    own slice of the starts (no pickling/dispatch — each rank already holds the
    function), then results are ``allgather``-ed so every rank holds the full,
    identical result; only rank 0 writes ``save_fn``.
    """

    logger = logging.getLogger('BaseProfiler')

    #: Override in subclasses (set ``True`` to enable gradient probing).
    with_gradient: bool = False

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000,
                 profiles=None, ref_scale=1., rescale=False, covariance=None,
                 save_fn=None, mpicomm=None):

        self.likelihood = likelihood
        self.max_tries  = int(max_tries)
        self.save_fn    = save_fn
        self.mpicomm    = mpicomm if mpicomm is not None else get_mpicomm()

        # ── collect varied parameters ─────────────────────────────────────
        self.varied_params = likelihood.params.select(fixed=False)
        if not self.varied_params:
            raise ValueError('No varied parameters found in the likelihood.')
        self.logger.info('Varied parameters: %s', self.varied_params.names())

        # Apply ref_scale (copy so we don't mutate the likelihood's params)
        if ref_scale != 1.:
            ref_scaled_params = VariableCollection()
            for param in self.varied_params:
                param_copy = copy.copy(param)
                param_copy.ref = param.ref.affine_transform(
                    loc=(1. - ref_scale) * param.ref.center(),
                    scale=ref_scale,
                )
                ref_scaled_params.set(param_copy)
            self.varied_params = ref_scaled_params

        # ── flat parameter layout ─────────────────────────────────────────
        # Every parameter is flattened: shape () → 1 element, shape (n,) → n
        # elements, etc.  ``_param_slices`` maps name → slice into the flat
        # optimisation vector.
        flat_offset = 0
        self._param_slices = {}
        for param in self.varied_params:
            flat_size = int(np.prod(param.shape)) if param.shape else 1
            self._param_slices[param.name] = slice(flat_offset, flat_offset + flat_size)
            flat_offset += flat_size
        self._flat_size = flat_offset

        # ── rescaling transform ───────────────────────────────────────────
        # _loc[k] and _scale[k] are the centre and step for flat element k.
        loc_parts = []
        for param in self.varied_params:
            center = np.asarray(
                param.value if param.value is not None else param.ref.center()
            ).ravel()
            flat_size = self._param_slices[param.name].stop - self._param_slices[param.name].start
            if center.size == 1 and flat_size > 1:
                center = np.full(flat_size, float(center[0]))
            loc_parts.append(center.astype('f8'))
        self._loc = np.concatenate(loc_parts) if loc_parts else np.array([], dtype='f8')

        if rescale:
            if covariance is not None:
                self._scale = np.sqrt(np.diag(np.asarray(covariance)))
            else:
                scale_parts = []
                for param in self.varied_params:
                    std = param.ref.std()
                    if std is None or not np.isfinite(std) or std <= 0.:
                        raise ValueError(
                            f'Parameter {param.name!r}: cannot determine rescale '
                            f'scale from ref.std()={std!r}. '
                            'Provide covariance or set a proper ref distribution.'
                        )
                    flat_size = self._param_slices[param.name].stop - self._param_slices[param.name].start
                    scale_parts.append(np.full(flat_size, float(std), dtype='f8'))
                self._scale = np.concatenate(scale_parts)
        else:
            self._scale = np.ones(self._flat_size, dtype='f8')

        # Build transformed VariableCollection (limits/ref in rescaled space; the
        # rescaled step size is recovered from the transformed ref.std()).
        self._transformed_params = VariableCollection()
        for param in self.varied_params:
            slc = self._param_slices[param.name]
            loc_p   = self._loc[slc]    # shape (flat_size_of_param,)
            scale_p = self._scale[slc]  # shape (flat_size_of_param,)
            param_copy = copy.copy(param)
            if not param.shape:
                # Scalar param: element-wise scalars for prior/ref transform
                loc_s, scale_s = float(loc_p[0]), float(scale_p[0])
                param_copy.prior = param.prior.affine_transform(loc=-loc_s / scale_s, scale=1. / scale_s)
                param_copy.ref   = param.ref.affine_transform(loc=-loc_s / scale_s, scale=1. / scale_s)
            else:
                # Vector param: element-wise array transform
                loc_arr   = loc_p.reshape(param.shape)
                scale_arr = scale_p.reshape(param.shape)
                param_copy.prior = param.prior.affine_transform(loc=-loc_arr / scale_arr, scale=1. / scale_arr)
                param_copy.ref   = param.ref.affine_transform(loc=-loc_arr / scale_arr, scale=1. / scale_arr)
            self._transformed_params.set(param_copy)

        # ── existing profiles ─────────────────────────────────────────────
        if profiles is not None and not isinstance(profiles, Profiles):
            profiles = Profiles.read(profiles)
        self.profiles = profiles

        # ── random state ──────────────────────────────────────────────────
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # ── JAX compilation ───────────────────────────────────────────────
        self._jit_chi2, self._grad_chi2 = _jit_and_grad(self._chi2_rescaled, with_gradient=self.with_gradient)

    # ── coordinate transforms ─────────────────────────────────────────────────

    def _forward(self, x):
        """Rescaled → original space.  ``x_original = x * scale + loc``.

        Operates on the flat parameter vector.  JAX-safe under jit/grad.
        """
        return jnp.asarray(x) * self._scale + self._loc

    def _backward(self, x):
        """Original → rescaled space.  ``x_rescaled = (x - loc) / scale``.

        Operates on the flat parameter vector.
        """
        return (np.asarray(x) - self._loc) / self._scale

    # ── chi2 ─────────────────────────────────────────────────────────────────

    def _chi2_rescaled(self, x):
        """χ² in rescaled space — fully JAX-traceable.

        Unpacks the flat vector ``x`` into per-parameter arrays (handling
        non-trivial shapes) before calling the likelihood.
        """
        x_orig = self._forward(x)
        params_dict = {}
        for name, param in zip(self.varied_params.names(), self.varied_params):
            slc = self._param_slices[name]
            val = x_orig[slc]
            if param.shape:
                val = val.reshape(param.shape)
            else:
                val = val[0]
            params_dict[name] = val
        logpost = self.likelihood(params_dict)
        return -2.0 * logpost

    # ── starting points ───────────────────────────────────────────────────────

    def _get_starts(self, nstarts):
        """Return ``(nstarts, flat_size)`` array of finite-chi2 starts in rescaled space."""
        result = np.full((nstarts, self._flat_size), np.nan)
        valid  = np.zeros(nstarts, dtype=bool)

        for _try in range(self.max_tries):
            if valid.all():
                break
            missing = int((~valid).sum())

            cols = []
            for param in self.varied_params:
                flat_size = self._param_slices[param.name].stop - self._param_slices[param.name].start
                if param.ref.is_proper():
                    try:
                        key = jax.random.PRNGKey(int(self.rng.integers(2**31)))
                        vals = np.asarray(param.ref.sample(key, shape=(missing,)))
                        # Flatten any trailing param-shape axes: (missing, *param.shape) → (missing, flat_size)
                        vals = vals.reshape(missing, flat_size)
                    except Exception:
                        center = np.asarray(
                            param.value if param.value is not None else param.ref.center()
                        ).ravel()
                        if center.size != flat_size:
                            center = np.full(flat_size, float(center.mean()))
                        vals = np.tile(center, (missing, 1))
                else:
                    center = np.asarray(
                        param.value if param.value is not None else param.ref.center()
                    ).ravel()
                    if center.size != flat_size:
                        center = np.full(flat_size, float(center.mean()))
                    vals = np.tile(center, (missing, 1))
                cols.append(vals)  # each has shape (missing, flat_size_of_param)

            candidates_orig     = np.concatenate(cols, axis=1)            # (missing, flat_size)
            candidates_rescaled = (candidates_orig - self._loc) / self._scale

            for candidate in candidates_rescaled:
                if valid.all():
                    break
                if np.isfinite(float(self._jit_chi2(candidate))):
                    slot = np.flatnonzero(~valid)[0]
                    result[slot] = candidate
                    valid[slot]  = True

        if not valid.all():
            raise ValueError(
                f'Could not find {nstarts} valid starting points after {self.max_tries} tries. '
                'Check that the likelihood is finite somewhere in the prior region.'
            )
        return result

    # ── transform profiles back to original space ─────────────────────────────

    def _transform_back(self, profiles):
        """Apply forward transform to best/start/error in-place (rescaled → original)."""
        if profiles is None:
            return profiles
        names = self.varied_params.names()
        for slot in ('best', 'start'):
            data = getattr(profiles, slot, None)
            if data is None:
                continue
            for name, arr in data.items():
                if name == 'logpdf' or name not in names:
                    continue
                slc     = self._param_slices[name]
                scale_p = self._scale[slc]
                loc_p   = self._loc[slc]
                param   = self.varied_params[name]
                if param.shape:
                    scale_p = scale_p.reshape(param.shape)
                    loc_p   = loc_p.reshape(param.shape)
                else:
                    scale_p = float(scale_p[0])
                    loc_p   = float(loc_p[0])
                data[name] = np.asarray(arr) * scale_p + loc_p
        error = getattr(profiles, 'error', None)
        if error is not None:
            for name, arr in error.items():
                if name not in names:
                    continue
                slc     = self._param_slices[name]
                scale_p = self._scale[slc]
                param   = self.varied_params[name]
                if param.shape:
                    scale_p = scale_p.reshape(param.shape)
                else:
                    scale_p = float(scale_p[0])
                error[name] = np.asarray(arr) * scale_p
        return profiles

    def _merge_and_transform(self, raw_list):
        """Concatenate rescaled Profiles list and transform to original space."""
        raw_list = [result for result in raw_list if result is not None]
        if not raw_list:
            return Profiles()
        profiles = Profiles.concatenate(raw_list)
        return self._transform_back(profiles)

    def _add_derived(self, profiles):
        """Re-evaluate the likelihood at each run's best-fit to populate derived parameters in ``profiles.best``.

        Derived parameters (``param.derived is True``) are computed as outputs of
        the likelihood pipeline.  They are not varied during optimisation, but their
        values at the best-fit point are useful for downstream analysis.

        The likelihood is called *eagerly* (not JIT) once per run.
        """
        derived_params = [p for p in self.likelihood.params if p.derived is True]
        if not derived_params or profiles is None or profiles.best is None:
            return profiles
        names   = self.varied_params.names()
        nruns   = profiles.nruns
        derived_arrays = {}
        for run_idx in range(nruns):
            params_dict = {name: np.asarray(profiles.best[name][run_idx]) for name in names}
            _, derived_dict = self.likelihood(params_dict, return_derived=True)
            for dp in derived_params:
                val = np.asarray(derived_dict[dp.name])
                if dp.name not in derived_arrays:
                    derived_arrays[dp.name] = np.empty((nruns,) + val.shape)
                derived_arrays[dp.name][run_idx] = val
        profiles.best.update(derived_arrays)
        return profiles

    # ── public API ────────────────────────────────────────────────────────────

    def maximize(self, niterations=None, start=None, **kwargs):
        """Maximize the likelihood from independent starting points.

        Parameters
        ----------
        niterations : int, optional
            Number of independent runs.  Defaults to 1.
        start : array_like, optional
            Fixed starting point(s) in *original* parameter space,
            shape ``(flat_size,)`` or ``(niterations, flat_size)``.
        **kwargs
            Passed to ``_maximize_one``.
        """
        if start is not None:
            start_arr       = np.atleast_2d(np.asarray(start, dtype='f8'))
            niterations     = niterations or len(start_arr)
            starts_rescaled = (start_arr - self._loc) / self._scale
        else:
            niterations     = niterations or 1
            starts_rescaled = self._get_starts(niterations)

        flat_bounds   = _compute_flat_bounds(self._transformed_params)
        flat_proposals = _compute_flat_proposals(self._transformed_params)

        state_tmpl = ProfilerState(
            chi2_fn=self._jit_chi2,
            grad_fn=self._grad_chi2,
            start=starts_rescaled[0],      # placeholder; replaced per-run
            varied_params=self._transformed_params,
            flat_bounds=flat_bounds,
            flat_proposals=flat_proposals,
            fast=False,
        )

        # Bind kwargs into the function so the per-start map gets a 1-arg callable
        run_one = functools.partial(_maximize_one_worker, self, state_tmpl, kwargs)
        raw = _pool_map(self.mpicomm, run_one, list(starts_rescaled))

        profiles = self._merge_and_transform(raw)
        self._add_derived(profiles)

        # Populate profiles.start with starting points in original space.
        # Only include starts for runs that produced a best-fit result.
        if profiles is not None and profiles.best is not None:
            successful_starts = np.array([
                s for r, s in zip(raw, starts_rescaled)
                if r is not None and r.best is not None
            ])
            if successful_starts.size > 0:
                starts_orig = successful_starts * self._scale + self._loc
                start_dict = {}
                for name, param in zip(self.varied_params.names(), self.varied_params):
                    slc  = self._param_slices[name]
                    vals = starts_orig[:, slc]  # (nruns, flat_size_of_param)
                    if param.shape:
                        start_dict[name] = vals.reshape((len(successful_starts),) + param.shape)
                    else:
                        start_dict[name] = vals[:, 0]  # (nruns,)
                profiles.start = start_dict

        self.profiles = (profiles if self.profiles is None
                         else Profiles.concatenate([self.profiles, profiles]))
        if self.save_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.save_fn)
        return self.profiles

    def covariance(self, **kwargs):
        """Estimate parameter covariance at the best-fit point.

        Uses ``jax.hessian``.  Updates ``error`` in ``self.profiles`` and stores
        the full matrix in ``self.profiles.attrs['covariance']``.
        """
        if self.profiles is None or self.profiles.best is None:
            self.maximize()

        argmax = int(np.argmax(self.profiles.best['logpdf']))
        # Build best_orig as a flat vector of shape (flat_size,)
        best_orig = np.concatenate([
            np.asarray(self.profiles.best[name][argmax]).ravel()
            for name in self.varied_params.names()
        ])
        best_rescaled = self._backward(best_orig)

        # Compute Hessian of χ² w.r.t. rescaled params via JAX
        hessian = np.asarray(jax.hessian(self._jit_chi2)(best_rescaled))

        try:
            cov_rescaled = np.linalg.inv(0.5 * hessian)
        except np.linalg.LinAlgError:
            self.logger.warning('Hessian inversion failed; covariance set to NaN.')
            cov_rescaled = np.full((self._flat_size, self._flat_size), np.nan)

        # x_original = x_rescaled * scale + loc  →  cov_orig[i,j] = cov_r[i,j]*scale[i]*scale[j]
        cov_orig   = cov_rescaled * np.outer(self._scale, self._scale)
        nruns      = self.profiles.nruns

        error_dict = _build_error_from_cov(cov_orig, self.varied_params)
        if self.profiles.error is None:
            self.profiles.error = {name: np.tile(err, (nruns,) + (1,) * (err.ndim - 1))
                                   for name, err in error_dict.items()}
        else:
            for name, err in error_dict.items():
                self.profiles.error[name] = np.tile(err, (nruns,) + (1,) * (err.ndim - 1))

        # Store full covariance as a Covariance object in the dedicated slot
        from ..samples import Covariance
        self.profiles.covariance = Covariance(cov_orig, params=list(self.varied_params))

        if self.save_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.save_fn)
        return self.profiles

    def interval(self, params=None, cl=1, niterations=1, xtol=1e-3, **kwargs):
        """Compute 1-D confidence intervals via profile-likelihood bisection.

        For each requested scalar parameter, fixes it at a sequence of trial
        values and maximises over the remaining parameters, then uses
        :func:`scipy.optimize.root_scalar` to locate the two points where
        :math:`\\Delta\\chi^2 = cl^2`.

        The result is stored in ``self.profiles.interval`` as
        ``interval[name] = (lo_offset, hi_offset)`` — offsets from the
        best-fit value, each with shape ``(1,)``.

        Parameters
        ----------
        params : str or list of str, optional
            Scalar parameters to compute intervals for.
            Defaults to all varied scalar parameters.
        cl : float
            Confidence level expressed as a number of Gaussian sigmas
            (e.g. ``1`` for 68 %, ``2`` for 95 %).  If ``0 < cl < 1`` the
            value is interpreted as a probability and converted internally.
        niterations : int
            Number of independent optimizer restarts per evaluation.
        xtol : float
            Convergence tolerance for the bisection (in sigma-equivalent units).
        **kwargs
            Forwarded to ``_maximize_one``.
        """
        if 0. < cl < 1.:
            from scipy import stats
            cl = float(stats.chi2(df=1).ppf(cl)) ** 0.5   # probability → sigma

        # ── resolve params ────────────────────────────────────────────────
        if params is None:
            param_names = [name for name in self.varied_params.names()
                           if not self.varied_params[name].shape]
        elif isinstance(params, str):
            param_names = [params]
        else:
            param_names = list(params)

        for name in param_names:
            if self.varied_params[name].shape:
                raise ValueError(
                    f'interval() only supports scalar parameters; '
                    f'{name!r} has shape {self.varied_params[name].shape}.'
                )

        if self.profiles is None or self.profiles.best is None:
            self.maximize()
        if self.profiles.covariance is None:
            self.covariance()

        argmax = int(np.argmax(self.profiles.best['logpdf']))
        lp_max = float(self.profiles.best['logpdf'][argmax])

        # Best-fit in rescaled space (needed by _interval_one)
        best_orig = np.concatenate([
            np.asarray(self.profiles.best[name][argmax]).ravel()
            for name in self.varied_params.names()
        ])
        center_r = self._backward(best_orig)

        cov_orig = np.asarray(self.profiles.covariance._value)
        interval_dict = {}

        for name in param_names:
            flat_pidx = self._param_slices[name].start

            # Standard deviation in rescaled space
            cov_orig_ii = float(cov_orig[flat_pidx, flat_pidx])
            sigma_r     = float(np.sqrt(max(cov_orig_ii, 0.))) / float(self._scale[flat_pidx])

            scan_ctx = self._build_scan_setup([flat_pidx])
            lo_off, hi_off = self._interval_one(
                name, flat_pidx, center_r, sigma_r, lp_max,
                scan_ctx, cl, xtol, niterations, **kwargs,
            )
            interval_dict[name] = (np.array([lo_off]), np.array([hi_off]))

        if self.profiles.interval is None:
            self.profiles.interval = interval_dict
        else:
            self.profiles.interval.update(interval_dict)

        if self.save_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.save_fn)
        return self.profiles

    def contour(self, params=None, cl=1, niterations=1, size=50, xtol=1e-3, **kwargs):
        """Compute 2-D profile-likelihood contours.

        For each requested pair of scalar parameters, sweeps angles around
        the best-fit Gaussian ellipse and uses
        :func:`scipy.optimize.root_scalar` at each angle to find the contour
        radius where :math:`\\Delta\\chi^2_\\text{2D} = \\text{factor}`.

        Results are stored in ``self.profiles.contour[cl][(name1, name2)]``
        as ``(x1_arr, x2_arr)`` — closed contour curves in original parameter
        space, shape ``(size + 1,)`` each.

        Parameters
        ----------
        params : optional
            How to specify parameter pairs:

            * ``None`` — all unique pairs of varied scalar parameters.
            * list of str — all unique pairs drawn from that list.
            * list of 2-tuples — those specific pairs.

        cl : float
            Confidence level in sigma units (``cl >= 1``) or as a probability
            (``0 < cl < 1``).
        niterations : int
            Optimizer restarts per evaluation.
        size : int
            Number of angles sampled around the contour.
        xtol : float
            Bisection tolerance (in ellipse-radius units).
        **kwargs
            Forwarded to ``_maximize_one``.
        """
        from scipy import stats

        if 0. < cl < 1.:
            cl = float(stats.chi2(df=1).ppf(cl)) ** 0.5   # probability → sigma

        # Δχ² threshold for 2 degrees of freedom at the equivalent sigma level
        factor_2d = float(stats.chi2(df=2).ppf(stats.chi2(df=1).cdf(cl ** 2)))

        # ── resolve param pairs ───────────────────────────────────────────
        if params is None:
            scalar_names = [name for name in self.varied_params.names()
                            if not self.varied_params[name].shape]
            param_pairs = [
                (scalar_names[idx1], scalar_names[idx2])
                for idx1 in range(len(scalar_names))
                for idx2 in range(idx1 + 1, len(scalar_names))
            ]
        else:
            params_list = list(params)
            if params_list and isinstance(params_list[0], (list, tuple)):
                param_pairs = [(str(p1), str(p2)) for p1, p2 in params_list]
            else:
                str_names = [p if isinstance(p, str) else p.name for p in params_list]
                param_pairs = [
                    (str_names[idx1], str_names[idx2])
                    for idx1 in range(len(str_names))
                    for idx2 in range(idx1 + 1, len(str_names))
                ]

        for name1, name2 in param_pairs:
            for name in (name1, name2):
                if self.varied_params[name].shape:
                    raise ValueError(
                        f'contour() only supports scalar parameters; '
                        f'{name!r} has shape {self.varied_params[name].shape}.'
                    )

        if self.profiles is None or self.profiles.best is None:
            self.maximize()
        if self.profiles.covariance is None:
            self.covariance()

        argmax = int(np.argmax(self.profiles.best['logpdf']))
        lp_max = float(self.profiles.best['logpdf'][argmax])

        best_orig = np.concatenate([
            np.asarray(self.profiles.best[name][argmax]).ravel()
            for name in self.varied_params.names()
        ])
        center_r = self._backward(best_orig)
        cov_orig = np.asarray(self.profiles.covariance._value)

        contour_pairs = {}
        for name1, name2 in param_pairs:
            flat_idx1 = self._param_slices[name1].start
            flat_idx2 = self._param_slices[name2].start

            # 2×2 covariance in rescaled space
            scale_2   = self._scale[[flat_idx1, flat_idx2]]
            cov_r_2x2 = (cov_orig[np.ix_([flat_idx1, flat_idx2], [flat_idx1, flat_idx2])]
                         / np.outer(scale_2, scale_2))

            scan_ctx = self._build_scan_setup([flat_idx1, flat_idx2])
            x1_arr, x2_arr = self._contour_one(
                name1, name2, flat_idx1, flat_idx2,
                center_r, cov_r_2x2, lp_max,
                scan_ctx, factor_2d, size, xtol, niterations, **kwargs,
            )
            contour_pairs[(name1, name2)] = (x1_arr, x2_arr)

        if self.profiles.contour is None:
            self.profiles.contour = {cl: contour_pairs}
        elif cl not in self.profiles.contour:
            self.profiles.contour[cl] = contour_pairs
        else:
            self.profiles.contour[cl].update(contour_pairs)

        if self.save_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.save_fn)
        return self.profiles

    def profile(self, params=None, grid=None, size=30, cl=2, niterations=1, **kwargs):
        """Compute 1-D profile likelihoods.

        Only scalar parameters are supported.

        Scan points are processed sequentially to allow warm-starting of the
        inner maximisation.

        Parameters
        ----------
        params : str or list[str], optional
            Parameters to profile.  Defaults to all varied scalar parameters.
        grid : array_like or list, optional
            Explicit scan values in *original* space (one per parameter).
        size : int or list[int]
            Number of scan points (ignored when *grid* is given).
        cl : float or list[float]
            Half-width of the scan in units of the parameter error.
        niterations : int
            Restarts per scan point (to reduce chance of local minima).
        """
        if self.profiles is None or self.profiles.best is None:
            self.maximize()
        if self.profiles.error is None:
            self.covariance()

        all_names = self.varied_params.names()
        param_names, grids, sizes, cls_ = _parse_profile_args(params, grid, size, cl, all_names)

        # Profile likelihood is only defined for scalar parameters.
        for pname in param_names:
            param = self.varied_params[pname]
            if param.shape:
                raise ValueError(
                    f'profile() only supports scalar parameters; '
                    f'{pname!r} has shape {param.shape}.'
                )

        argmax    = int(np.argmax(self.profiles.best['logpdf']))
        profile_results = {}

        for pname, grid_vals, npoints, cl_val in zip(param_names, grids, sizes, cls_):
            flat_pidx = self._param_slices[pname].start  # flat index of this scalar param
            scan = _build_1d_grid(pname, flat_pidx, grid_vals, npoints, cl_val, argmax, self.profiles, self.varied_params)

            scan_r          = (scan - self._loc[flat_pidx]) / self._scale[flat_pidx]
            fixed_points_r  = scan_r.reshape(-1, 1)
            logposteriors   = self._scan([flat_pidx], fixed_points_r, niterations, **kwargs)
            profile_results[pname] = (scan, logposteriors)

        if self.profiles.profile is None:
            self.profiles.profile = {}
        self.profiles.profile.update(profile_results)

        if self.save_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.save_fn)
        return self.profiles

    def grid(self, params=None, grid=None, size=10, cl=2, niterations=1, **kwargs):
        """Compute best-fit log-posterior on a multi-dimensional parameter grid.

        Only scalar parameters are supported.

        Parameters
        ----------
        params : list[str], optional
            Parameters spanning the grid.
        grid : list of array_like, optional
            Explicit scan values per parameter in *original* space.
        size, cl : int/float or list thereof
        niterations : int
        """
        if self.profiles is None or self.profiles.best is None:
            self.maximize()
        if self.profiles.error is None:
            self.covariance()

        all_names = self.varied_params.names()
        param_names, grids, sizes, cls_ = _parse_profile_args(params, grid, size, cl, all_names)

        # Grid is only defined for scalar parameters.
        for pname in param_names:
            param = self.varied_params[pname]
            if param.shape:
                raise ValueError(
                    f'grid() only supports scalar parameters; '
                    f'{pname!r} has shape {param.shape}.'
                )

        flat_grid_idx = [self._param_slices[name].start for name in param_names]
        argmax        = int(np.argmax(self.profiles.best['logpdf']))
        grids1d = [
            _build_1d_grid(pname, flat_pidx, grid_vals, npoints, cl_val, argmax, self.profiles, self.varied_params)
            for pname, flat_pidx, grid_vals, npoints, cl_val
            in zip(param_names, flat_grid_idx, grids, sizes, cls_)
        ]
        mesh       = np.meshgrid(*grids1d, indexing='ij')
        grid_shape = mesh[0].shape
        flat_pts   = np.column_stack([m.ravel() for m in mesh])  # (N, n_grid)

        # Rescale fixed points (vectorised over all grid points)
        loc_grid        = self._loc[flat_grid_idx]
        scale_grid      = self._scale[flat_grid_idx]
        fixed_points_r  = (flat_pts - loc_grid) / scale_grid  # (N, n_grid)

        lp_grid = self._scan(flat_grid_idx, fixed_points_r, niterations, **kwargs).reshape(grid_shape)

        if self.profiles.grid is None:
            self.profiles.grid = {}
        self.profiles.grid.update({
            'params':  param_names,
            'grids':   [dim_grid.tolist() for dim_grid in grids1d],
            'logpdf':  lp_grid,
        })

        if self.save_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.save_fn)
        return self.profiles

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_scan_setup(self, fixed_flat_idx):
        """Precompute a reusable scan context for the given fixed flat indices.

        The context holds pre-built JIT functions, parameter collections, and
        index arrays so that ``_scan_single`` can be called repeatedly without
        recompilation.

        Parameters
        ----------
        fixed_flat_idx : list of int
            Flat-vector indices of the parameters to hold fixed.

        Returns
        -------
        dict
            Keys: ``free_params``, ``free_flat_idx``, ``free_idx_arr``,
            ``fixed_idx_arr``, ``flat_bounds_free``, ``flat_proposals_free``,
            ``jit_chi2_fixed``, ``grad_chi2_fixed``.
        """
        flat_size      = self._flat_size
        fixed_flat_set = set(fixed_flat_idx)
        free_flat_idx  = [k for k in range(flat_size) if k not in fixed_flat_set]

        # Free parameters: those whose every flat element is free.
        free_params = VariableCollection()
        for param in self._transformed_params:
            slc = self._param_slices[param.name]
            if set(range(slc.start, slc.stop)) <= set(free_flat_idx):
                free_params.set(param)

        fixed_idx_arr = np.array(fixed_flat_idx, dtype=int)
        free_idx_arr  = np.array(free_flat_idx,  dtype=int) if free_flat_idx else np.array([], dtype=int)

        all_bounds     = _compute_flat_bounds(self._transformed_params)
        all_proposals  = _compute_flat_proposals(self._transformed_params)
        flat_bounds_free    = [all_bounds[k]    for k in free_flat_idx]
        flat_proposals_free = [all_proposals[k] for k in free_flat_idx]

        jit_chi2_fixed = grad_chi2_fixed = None
        if free_flat_idx:
            def chi2_2arg(x_free, point_r):
                full = jnp.zeros(flat_size).at[free_idx_arr].set(x_free).at[fixed_idx_arr].set(point_r)
                return self._chi2_rescaled(full)
            jit_chi2_fixed, grad_chi2_fixed = _jit_and_grad(chi2_2arg, with_gradient=self.with_gradient)

        return dict(
            free_params=free_params,
            free_flat_idx=free_flat_idx,
            free_idx_arr=free_idx_arr,
            fixed_idx_arr=fixed_idx_arr,
            flat_bounds_free=flat_bounds_free,
            flat_proposals_free=flat_proposals_free,
            jit_chi2_fixed=jit_chi2_fixed,
            grad_chi2_fixed=grad_chi2_fixed,
        )

    def _scan_single(self, scan_ctx, fixed_value_r, niterations, last_free=None, **kwargs):
        """Optimise the free parameters for one fixed-point evaluation.

        Parameters
        ----------
        scan_ctx : dict
            Context returned by :meth:`_build_scan_setup`.
        fixed_value_r : array_like, shape ``(n_fixed,)``
            Rescaled values of the fixed parameters at this scan point.
        niterations : int
            Number of independent optimizer restarts.
        last_free : np.ndarray or None
            Previous best free-parameter vector used for warm-starting.

        Returns
        -------
        best_lp : float
            Best log-posterior found.
        best_free : np.ndarray or None
            Flat free-parameter vector at the best-fit (for warm-starting the
            next call), or ``None`` when there are no free parameters.
        """
        if not scan_ctx['free_flat_idx']:
            # All parameters fixed — direct chi2 evaluation, no optimisation.
            full_r = jnp.zeros(self._flat_size).at[scan_ctx['fixed_idx_arr']].set(jnp.asarray(fixed_value_r))
            return float(-0.5 * self._jit_chi2(full_r)), None

        point_r_jnp = jnp.asarray(fixed_value_r)
        chi2_fn = lambda x_free, _pt=point_r_jnp: scan_ctx['jit_chi2_fixed'](x_free, _pt)
        grad_fn = ((lambda x_free, _pt=point_r_jnp: scan_ctx['grad_chi2_fixed'](x_free, _pt))
                   if scan_ctx['grad_chi2_fixed'] is not None else None)

        full_starts = self._get_starts(niterations)
        starts_free = full_starts[:, scan_ctx['free_idx_arr']]
        if last_free is not None:
            # Warm-start: small perturbations around previous best
            spread      = (starts_free - starts_free.mean(axis=0)) * 0.1
            starts_free = spread + last_free[None, :]

        state_tmpl = ProfilerState(
            chi2_fn=chi2_fn,
            grad_fn=grad_fn,
            start=starts_free[0],
            varied_params=scan_ctx['free_params'],
            flat_bounds=scan_ctx['flat_bounds_free'],
            flat_proposals=scan_ctx['flat_proposals_free'],
            fast=True,
        )

        best_lp   = -np.inf
        best_free = None
        for start in starts_free:
            state = dataclasses.replace(state_tmpl, start=start)
            raw   = self._maximize_one(state, **kwargs)
            if raw is not None and raw.best is not None:
                lp = float(np.max(raw.best.get('logpdf', [-np.inf])))
                if lp > best_lp:
                    best_lp = lp
                    if scan_ctx['free_params']:
                        idx_best  = int(np.argmax(raw.best['logpdf']))
                        best_free = np.concatenate([
                            np.asarray(raw.best[param.name][idx_best]).ravel()
                            for param in scan_ctx['free_params']
                        ])

        return best_lp, best_free

    def _scan(self, fixed_flat_idx, fixed_points_r, niterations, **kwargs):
        """Maximise over free parameters at every point in a parameter scan.

        This is the common inner loop shared by :meth:`profile` and
        :meth:`grid`.  Fixed parameters are held constant at the flat-element
        indices *fixed_flat_idx*; the remaining elements are optimised.

        A single ``chi2_2arg(x_free, point_r)`` is JIT-compiled once using
        JAX scatter, so there is no per-scan-point recompilation.

        Parameters
        ----------
        fixed_flat_idx : list of int
            Flat-vector indices of the fixed elements.
        fixed_points_r : array_like, shape ``(n_points, n_fixed)``
            Rescaled values of the fixed elements at each scan point.
        niterations : int
            Number of independent restarts per scan point.

        Returns
        -------
        np.ndarray, shape ``(n_points,)``
            Best-fit log-posterior at each scan point.
        """
        fixed_points_r = np.asarray(fixed_points_r)
        scan_ctx       = self._build_scan_setup(fixed_flat_idx)

        last_best_free = None
        logposteriors  = []
        for point_r in fixed_points_r:
            lp, last_best_free = self._scan_single(scan_ctx, point_r, niterations, last_best_free, **kwargs)
            logposteriors.append(lp)
        return np.array(logposteriors)

    def _interval_one(self, name, flat_pidx, center_r, sigma_r, lp_max,
                      scan_ctx, cl, xtol, niterations, **kwargs):
        """Find the 1-D confidence interval for one scalar parameter.

        Uses bisection (:func:`scipy.optimize.root_scalar`) to locate the two
        :math:`\\Delta\\chi^2 = cl^2` crossing points on either side of the
        best-fit.

        Parameters
        ----------
        name : str
        flat_pidx : int
            Index into the flat parameter vector.
        center_r : np.ndarray, shape ``(flat_size,)``
            Best-fit in rescaled space.
        sigma_r : float
            Standard deviation of this parameter in rescaled space.
        lp_max : float
            Log-posterior at the best-fit.
        scan_ctx : dict
            Context from :meth:`_build_scan_setup` with this param fixed.
        cl : float
            Sigma level.
        xtol : float
            Bisection tolerance in sigma units.
        niterations : int

        Returns
        -------
        lo_offset, hi_offset : float
            Offsets from best-fit in *original* space.
            Typically ``lo_offset < 0 < hi_offset``.
        """
        from scipy.optimize import root_scalar

        factor       = cl ** 2
        center_val_r = float(center_r[flat_pidx])
        scale_p      = float(self._scale[flat_pidx])
        loc_p        = float(self._loc[flat_pidx])
        center_orig  = center_val_r * scale_p + loc_p

        lim_lo_r, lim_hi_r = self._transformed_params[name].prior.limits

        def _get_point_r(z):
            """Rescaled param value at normalized displacement z."""
            x = center_val_r + z * sigma_r
            if np.isfinite(lim_lo_r):
                x = max(x, float(lim_lo_r))
            if np.isfinite(lim_hi_r):
                x = min(x, float(lim_hi_r))
            return x

        def _to_orig(x_r):
            return x_r * scale_p + loc_p

        interval_bounds = []
        warm = {'last_free': None}   # mutable warm-start state shared across scan calls

        for sign in (-1, 1):
            warm['last_free'] = None  # fresh warm-start for each direction

            def scan(z, _sign=sign):
                point_r = np.array([_get_point_r(_sign * z)])
                lp, best_free = self._scan_single(
                    scan_ctx, point_r, niterations, warm['last_free'], **kwargs
                )
                if best_free is not None:
                    warm['last_free'] = best_free
                return -2.0 * (lp - lp_max) - factor

            # ── find bracket ──────────────────────────────────────────────
            a = 0.5
            while scan(a) > 0. and a > 1e-7:
                a *= 0.5

            if a < 1e-7:
                interval_bounds.append(np.nan)
                continue

            b = 1.2
            while scan(b) < 0. and b < 8.:
                b *= 1.1

            if b >= 8.:
                # Scan limit reached — use current boundary as a conservative bound
                interval_bounds.append(_to_orig(_get_point_r(sign * b)) - center_orig)
                continue

            result = root_scalar(scan, bracket=(a, b), xtol=xtol)
            if result.converged:
                interval_bounds.append(_to_orig(_get_point_r(sign * result.root)) - center_orig)
            else:
                interval_bounds.append(np.nan)

        return float(interval_bounds[0]), float(interval_bounds[1])

    def _contour_one(self, name1, name2, flat_idx1, flat_idx2,
                     center_r, cov_r_2x2, lp_max,
                     scan_ctx, factor_2d, size, xtol, niterations, **kwargs):
        """Compute the 2-D profile-likelihood contour for one parameter pair.

        Sweeps angles uniformly around the Gaussian-approximation ellipse.
        At each angle, bisection locates the contour radius where
        :math:`\\Delta\\chi^2_\\text{2D} = \\text{factor\\_2d}`.

        Parameters
        ----------
        name1, name2 : str
        flat_idx1, flat_idx2 : int
        center_r : np.ndarray, shape ``(flat_size,)``
        cov_r_2x2 : np.ndarray, shape ``(2, 2)``
            2×2 covariance in *rescaled* space.
        lp_max : float
        scan_ctx : dict
            Context from :meth:`_build_scan_setup` with both params fixed.
        factor_2d : float
            :math:`\\Delta\\chi^2` threshold (2 degrees of freedom).
        size : int
            Number of angular scan points.
        xtol : float
        niterations : int

        Returns
        -------
        x1_arr, x2_arr : np.ndarray, shape ``(size + 1,)``
            Closed contour coordinates in *original* parameter space.
        """
        from scipy.optimize import root_scalar

        # Eigendecompose the 2×2 covariance to get ellipse geometry
        eigenvalues, U = np.linalg.eigh(cov_r_2x2)
        s = np.sqrt(np.maximum(eigenvalues * factor_2d, 0.))   # semi-axes in rescaled space

        scale_1 = float(self._scale[flat_idx1])
        scale_2 = float(self._scale[flat_idx2])
        loc_1   = float(self._loc[flat_idx1])
        loc_2   = float(self._loc[flat_idx2])
        center_12 = center_r[[flat_idx1, flat_idx2]].copy()

        lim_lo_r1, lim_hi_r1 = self._transformed_params[name1].prior.limits
        lim_lo_r2, lim_hi_r2 = self._transformed_params[name2].prior.limits

        def _get_point_r(phi, z):
            """2D rescaled point at angle *phi* and radius *z*."""
            direction = U @ np.array([z * s[0] * np.cos(phi), z * s[1] * np.sin(phi)])
            raw       = center_12 + direction
            if np.isfinite(lim_lo_r1): raw[0] = max(raw[0], float(lim_lo_r1))
            if np.isfinite(lim_hi_r1): raw[0] = min(raw[0], float(lim_hi_r1))
            if np.isfinite(lim_lo_r2): raw[1] = max(raw[1], float(lim_lo_r2))
            if np.isfinite(lim_hi_r2): raw[1] = min(raw[1], float(lim_hi_r2))
            return raw  # shape (2,)

        def _to_orig_2(point_2r):
            return np.array([
                float(point_2r[0]) * scale_1 + loc_1,
                float(point_2r[1]) * scale_2 + loc_2,
            ])

        phis            = np.linspace(-np.pi, np.pi, size, endpoint=False)
        contour_points  = []
        warm            = {'last_free': None}   # persists across all angles

        for phi in phis:
            def scan(z, _phi=phi):
                point_2r = _get_point_r(_phi, z)
                lp, best_free = self._scan_single(
                    scan_ctx, point_2r, niterations, warm['last_free'], **kwargs
                )
                if best_free is not None:
                    warm['last_free'] = best_free
                return -2.0 * (lp - lp_max) - factor_2d

            # ── find bracket ──────────────────────────────────────────────
            a = 0.5
            while scan(a) > 0. and a > 1e-7:
                a *= 0.5

            if a < 1e-7:
                contour_points.append((np.nan, np.nan))
                continue

            b = 1.2
            while scan(b) < 0. and b < 8.:
                b *= 1.1

            if b >= 8.:
                pt = _to_orig_2(_get_point_r(phi, b))
                contour_points.append((float(pt[0]), float(pt[1])))
                continue

            result = root_scalar(scan, bracket=(a, b), xtol=xtol)
            if result.converged:
                pt = _to_orig_2(_get_point_r(phi, result.root))
                contour_points.append((float(pt[0]), float(pt[1])))
            else:
                contour_points.append((np.nan, np.nan))

        # Close the contour
        contour_points.append(contour_points[0])
        arr = np.array(contour_points).T   # shape (2, size + 1)
        return arr[0], arr[1]

    # ── abstract interface ────────────────────────────────────────────────────

    def _maximize_one(self, state: ProfilerState, **kwargs) -> Profiles:
        """One optimisation run in rescaled space.  Subclasses implement this."""
        raise NotImplementedError


# ── module-level helpers (picklable) ──────────────────────────────────────────

def _parse_profile_args(params, grid, size, cl, all_names):
    """Normalise params/grid/size/cl to equal-length lists."""
    if params is None:
        params = list(all_names)
    elif isinstance(params, str):
        params = [params]
    nparams = len(params)
    if not isinstance(size, (list, tuple)):
        size = [size] * nparams
    if not isinstance(cl, (list, tuple)):
        cl = [cl] * nparams
    if grid is None or (not isinstance(grid, (list, tuple)) or
                        (len(grid) > 0 and np.ndim(grid[0]) == 0)):
        grid = [grid] * nparams
    return list(params), list(grid), list(size), list(cl)


def _build_1d_grid(name, flat_pidx, grid, size, cl, argmax, profiles, varied_params):
    """Return a 1-D scan array in *original* parameter space."""
    if grid is not None:
        return np.asarray(grid, dtype='f8')
    best = float(profiles.best[name][argmax])
    err  = None
    if profiles.error is not None and name in profiles.error:
        err_val = float(profiles.error[name].ravel()[argmax])
        if np.isfinite(err_val) and err_val > 0.:
            err = err_val
    if err is None:
        param = varied_params[name]
        std   = param.ref.std()
        if std is None or not np.isfinite(std) or std <= 0.:
            raise ValueError(
                f'Cannot build grid for {name!r}: no finite error or ref.std().'
            )
        err = float(std)
    lo, hi = best - cl * err, best + cl * err
    lim = varied_params[name].prior.limits
    if np.isfinite(lim[0]):
        lo = max(lo, float(lim[0]))
    if np.isfinite(lim[1]):
        hi = min(hi, float(lim[1]))
    return np.linspace(lo, hi, int(size))


def _jit_and_grad(fn, with_gradient=False):
    """Return ``(jit_fn, grad_fn)`` for *fn*.

    Parameters
    ----------
    fn : callable
        A JAX-traceable scalar function ``f(x) -> scalar``.
    with_gradient : bool
        When ``True``, also compile ``jax.grad(fn)``; otherwise *grad_fn*
        is ``None``.

    Returns
    -------
    jit_fn : callable
        ``jax.jit(fn)``
    grad_fn : callable or None
        ``jax.jit(jax.grad(fn))`` if *with_gradient*, else ``None``.
    """
    jit_fn  = jax.jit(fn)
    grad_fn = jax.jit(jax.grad(fn)) if with_gradient else None
    return jit_fn, grad_fn


def _maximize_one_worker(profiler, state_tmpl, kwargs, x0):
    """Top-level (picklable) worker for pool.map in maximize()."""
    state = dataclasses.replace(state_tmpl, start=x0)
    return profiler._maximize_one(state, **kwargs)
