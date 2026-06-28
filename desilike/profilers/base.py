"""BaseProfiler — core profiling logic shared by all concrete profilers."""

from __future__ import annotations

import logging
import dataclasses
import functools

import numpy as np
import jax
import jax.numpy as jnp

from ..parameter import VariableCollection
from ..samples import Profiles, Covariance
from ..distributed import default_mpicomm, get_mpicomm
from ..conditioning import AffineConditioner


# ── Kernel ────────────────────────────────────────────────────────────────────

class Kernel:
    """Abstract base class for profiler kernels.

    A kernel encapsulates an optimisation algorithm.  It knows nothing about
    MPI, rescaling, or parameter bookkeeping — those live in :class:`Profiler`.
    """

    logger = logging.getLogger('Kernel')

    #: Set ``True`` for gradient-based kernels; the profiler will then compile
    #: ``jax.grad(chi2)`` and pass it to :meth:`run`.
    with_gradient: bool = False

    def init(self):
        """Lightweight one-time setup (e.g. dependency check).

        Called eagerly in :meth:`Profiler.__init__` before any chi2 is
        available.  The default implementation does nothing.
        """

    def run(self, state: 'ProfilerState', chi2, grad=None, **kwargs) -> 'ProfilerState':
        """Run one optimisation.

        Parameters
        ----------
        state : ProfilerState
            Input state: ``start`` (flat rescaled vector), ``bounds``,
            ``proposals``, ``fast``.  The result fields
            (``best``, ``logpdf``, ``cov``) are ``None`` on entry.
        chi2 : callable
            JIT-compiled ``chi2(x_rescaled) → scalar``.
        grad : callable or None
            JIT-compiled gradient of ``chi2``, or ``None`` for derivative-free
            kernels.
        **kwargs
            Algorithm-specific keyword arguments (e.g. ``max_iterations``).

        Returns
        -------
        ProfilerState
            Same object, with ``best``, ``logpdf``, and optionally ``cov``
            filled in (or left ``None`` when the optimisation failed).
        """
        raise NotImplementedError


# ── ProfilerState ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ProfilerState:
    """Mutable snapshot of one optimisation problem in *rescaled* space.

    ``start``, ``bounds``, and ``proposals`` use the *flat* parameter layout:
    a vector param of shape ``(n,)`` occupies ``n`` consecutive elements.
    Scalar params occupy one element each.

    The result fields (``best``, ``logpdf``, ``cov``) are ``None`` before
    :meth:`Kernel.run` fills them in.
    """
    start:     np.ndarray          # shape (flat_size,) in rescaled space
    bounds:    list                # [(lo, hi), …] per flat element, rescaled
    proposals: list                # [proposal, …]  per flat element, rescaled
    fast:      bool                # True → skip hesse / covariance
    # Results (filled by Kernel.run)
    best:   np.ndarray | None = None
    logpdf: float | None = None
    cov:    np.ndarray | None = None   # (flat_size, flat_size) or None


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


def _state_to_profiles(state: ProfilerState, varied_params) -> 'Profiles | None':
    """Convert a completed ProfilerState to a Profiles object (rescaled space)."""
    if state.best is None:
        return None
    profiles = Profiles()
    profiles.best = _build_best_from_x(state.best, state.logpdf, varied_params)
    profiles.logpdf = np.array([state.logpdf])
    if state.cov is not None and not state.fast:
        profiles.error = _build_error_from_cov(state.cov, varied_params)
        profiles.covariance = Covariance(state.cov, list(varied_params))
    return profiles


# ── Profiler ──────────────────────────────────────────────────────────────────

class Profiler:
    """Likelihood profiler: maximize, profile, grid, covariance, interval, contour.

    Parameters
    ----------
    likelihood : CompiledGraph
        Compiled pipeline whose ``__call__(params_dict)`` returns the
        log-posterior scalar.
    kernel : Kernel
        Optimisation kernel (e.g. ``Minuit()``, ``Scipy()``, ``BOBYQA()``).
    rng : np.random.Generator or int, optional
        Random number generator or integer seed.  When ``None`` a fresh
        unseeded generator is used.
    max_tries : int
        Maximum candidate draws when searching for a finite starting point.
    profiles : Profiles or path, optional
        Existing profiles to append new results to.
    rescale : bool
        Internally normalise parameters so that their expected variation
        range is ~ unity.
    covariance : array_like, optional
        ``(flat_size, flat_size)`` covariance used to set the rescaling scale.
        When ``None``, each parameter's ``ref.std()`` is used instead.
    output_fn : str or Path, optional
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
    identical result; only rank 0 writes ``output_fn``.
    """

    logger = logging.getLogger('Profiler')

    @default_mpicomm
    def __init__(self, likelihood, kernel, rng=None, max_tries=1000,
                 profiles=None, conditioning=None,
                 output_fn=None, mpicomm=None):

        self.likelihood = likelihood
        self.kernel     = kernel
        self.max_tries  = int(max_tries)
        self.output_fn    = output_fn
        self.mpicomm    = mpicomm

        # ── collect varied parameters ─────────────────────────────────────
        self.varied_params = likelihood.params.select(varied=True, derived=False)
        if not self.varied_params:
            raise ValueError('No varied parameters found in the likelihood.')
        if self.mpicomm.rank == 0:
            self.logger.info('Varied parameters: %s', self.varied_params.names())
            if self.output_fn is not None:
                self.logger.info('Profiles will be written to: %s', self.output_fn)

        # ── flat parameter layout ─────────────────────────────────────────
        flat_offset = 0
        self._param_slices = {}
        for param in self.varied_params:
            flat_size = int(np.prod(param.shape)) if param.shape else 1
            self._param_slices[param.name] = slice(flat_offset, flat_offset + flat_size)
            flat_offset += flat_size
        self._flat_size = flat_offset

        # ── conditioning transform ─────────────────────────────────────────
        if conditioning is None:
            conditioning = AffineConditioner()
        self.conditioning = conditioning
        self.conditioning.init(self.varied_params)

        # ── existing profiles ─────────────────────────────────────────────
        if profiles is not None and not isinstance(profiles, Profiles):
            profiles = Profiles.read(profiles)
        self.profiles = profiles

        # ── random state ──────────────────────────────────────────────────
        self.rng = np.random.default_rng(rng)

        # ── JAX compilation ───────────────────────────────────────────────
        self._jit_chi2, self._grad_chi2 = _jit_and_grad(self._chi2_rescaled, with_gradient=kernel.with_gradient)

        # ── kernel setup ──────────────────────────────────────────────────
        kernel.init()

    # ── chi2 ─────────────────────────────────────────────────────────────────

    def _chi2_rescaled(self, x):
        """χ² in conditioned space — fully JAX-traceable.

        Unpacks the flat vector ``x`` into per-parameter arrays (handling
        non-trivial shapes) before calling the likelihood.
        """
        x_orig = self.conditioning.forward(x)
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
            candidates_rescaled = np.asarray(self.conditioning.inverse(candidates_orig))

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

    def _conditioned_bounds_and_proposals(self):
        """Return ``(bounds, proposals)`` in conditioned parameter space.

        *bounds* is a list of ``(lo, hi)`` tuples, one per flat element.
        *proposals* is a list of step-size floats (``ref.std() / scale``).
        """
        prior_bounds = self.conditioning.prior_bounds()  # (ndim, 2)
        bounds = [(float(prior_bounds[k, 0]), float(prior_bounds[k, 1]))
                  for k in range(self._flat_size)]
        _scale = self.conditioning._scale
        proposals = []
        for param in self.varied_params:
            slc = self._param_slices[param.name]
            flat_size = slc.stop - slc.start
            std = param.ref.std()
            scale_val = float(_scale[slc.start])
            p_val = (float(std) / scale_val) if (std is not None and np.isfinite(float(std)) and float(std) > 0.) else 1.0
            proposals.extend([p_val] * flat_size)
        return bounds, proposals

    # ── transform profiles back to original space ─────────────────────────────

    def _transform_back(self, profiles):
        """Apply forward transform to best/start/error in-place (rescaled → original)."""
        if profiles is None:
            return profiles
        names = self.varied_params.names()
        _scale = self.conditioning._scale
        _loc = self.conditioning._loc
        for slot in ('best', 'start'):
            data = getattr(profiles, slot, None)
            if data is None:
                continue
            for name, arr in data.items():
                if name not in names:
                    continue
                slc     = self._param_slices[name]
                scale_p = _scale[slc]
                loc_p   = _loc[slc]
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
                scale_p = _scale[slc]
                param   = self.varied_params[name]
                if param.shape:
                    scale_p = scale_p.reshape(param.shape)
                else:
                    scale_p = float(scale_p[0])
                error[name] = np.asarray(arr) * scale_p
        covariance = getattr(profiles, 'covariance', None)
        if covariance is not None:
            cov_array = np.asarray(covariance) * _scale[:, None] * _scale[None, :]
            profiles.covariance = Covariance(cov_array, list(self.varied_params))
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
        if profiles is None:
            return profiles
        derived_params = [p for p in self.likelihood.params if p.derived]
        # Attach parameter metadata (priors, latex, …) so downstream consumers
        # (to_stats, select, plotting) have access to it.
        profiles.params = self.likelihood.params
        # Carry likelihood bookkeeping (ndata, nvaried) into the result.
        ndata = getattr(self.likelihood.root, 'ndata', None)
        if ndata is not None:
            nvaried = sum(int(np.prod(param.shape, dtype='intp')) for param in self.likelihood.params.select(input=True, varied=True))
            profiles.attrs.update(ndata=ndata, nvaried=nvaried)
        if not derived_params or profiles.best is None:
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
            starts_rescaled = np.asarray(self.conditioning.inverse(start_arr))
        else:
            niterations     = niterations or 1
            starts_rescaled = self._get_starts(niterations)

        bounds, proposals = self._conditioned_bounds_and_proposals()

        state_tmpl = ProfilerState(
            start=starts_rescaled[0],      # placeholder; replaced per run
            bounds=bounds,
            proposals=proposals,
            fast=False,
        )

        run_one = functools.partial(_run_one_worker, self.kernel, state_tmpl, self._jit_chi2, self._grad_chi2, kwargs)
        raw = _pool_map(self.mpicomm, run_one, list(starts_rescaled))

        profiles = self._merge_and_transform([_state_to_profiles(state, self.varied_params) for state in raw])
        self._add_derived(profiles)

        # Populate profiles.start with starting points in original space.
        # Only include starts for runs that produced a best-fit result.
        if profiles is not None and profiles.best is not None:
            successful_starts = np.array([
                s for state, s in zip(raw, starts_rescaled)
                if state.best is not None
            ])
            if successful_starts.size > 0:
                starts_orig = np.asarray(self.conditioning.forward(successful_starts))
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
        if self.output_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.output_fn)
        return self.profiles

    def covariance(self, **kwargs):
        """Estimate parameter covariance at the best-fit point.

        Uses ``jax.hessian``.  Updates ``error`` in ``self.profiles`` and stores
        the full matrix in ``self.profiles.attrs['covariance']``.
        """
        if self.profiles is None or self.profiles.best is None:
            self.maximize()

        argmax = int(np.argmax(self.profiles.logpdf))
        # Build best_orig as a flat vector of shape (flat_size,)
        best_orig = np.concatenate([
            np.asarray(self.profiles.best[name][argmax]).ravel()
            for name in self.varied_params.names()
        ])
        best_rescaled = np.asarray(self.conditioning.inverse(best_orig))

        # Compute Hessian of χ² w.r.t. conditioned params via JAX
        hessian = np.asarray(jax.hessian(self._jit_chi2)(best_rescaled))

        try:
            cov_rescaled = np.linalg.inv(0.5 * hessian)
        except np.linalg.LinAlgError:
            self.logger.warning('Hessian inversion failed; covariance set to NaN.')
            cov_rescaled = np.full((self._flat_size, self._flat_size), np.nan)

        # cov_orig[i,j] = cov_r[i,j] * scale[i] * scale[j]  (diagonal conditioning only)
        _scale = self.conditioning._scale
        cov_orig = cov_rescaled * np.outer(_scale, _scale)
        nruns      = self.profiles.nruns

        error_dict = _build_error_from_cov(cov_orig, self.varied_params)
        if self.profiles.error is None:
            self.profiles.error = {name: np.tile(err, (nruns,) + (1,) * (err.ndim - 1))
                                   for name, err in error_dict.items()}
        else:
            for name, err in error_dict.items():
                self.profiles.error[name] = np.tile(err, (nruns,) + (1,) * (err.ndim - 1))

        # Store full covariance as a Covariance object in the dedicated slot.
        # Clone params with value = best-fit so that Covariance.center returns the best-fit point.
        from ..samples import Covariance
        best_params = []
        for param in self.varied_params:
            best_val = self.profiles.best[param.name][argmax]
            best_params.append(param.clone(value=best_val))
        self.profiles.covariance = Covariance(cov_orig, params=best_params)

        if self.output_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.output_fn)
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
        if self.conditioning.is_mixing:
            raise ValueError(
                'interval() is not supported when the conditioner mixes parameter dimensions '
                '(AffineConditioner with rescale="full" and a non-diagonal covariance). '
                'Use AffineConditioner(rescale="diag") instead.')

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

        argmax = int(np.argmax(self.profiles.logpdf))
        lp_max = float(self.profiles.logpdf[argmax])

        # Best-fit in rescaled space (needed by _interval_one)
        best_orig = np.concatenate([
            np.asarray(self.profiles.best[name][argmax]).ravel()
            for name in self.varied_params.names()
        ])
        center_r = np.asarray(self.conditioning.inverse(best_orig))

        cov_orig = np.asarray(self.profiles.covariance.value)
        interval_dict = {}

        for name in param_names:
            flat_pidx = self._param_slices[name].start

            # Standard deviation in conditioned space
            cov_orig_ii = float(cov_orig[flat_pidx, flat_pidx])
            sigma_r     = float(np.sqrt(max(cov_orig_ii, 0.))) / float(self.conditioning._scale[flat_pidx])

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

        if self.output_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.output_fn)
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
        if self.conditioning.is_mixing:
            raise ValueError(
                'contour() is not supported when the conditioner mixes parameter dimensions '
                '(AffineConditioner with rescale="full" and a non-diagonal covariance). '
                'Use AffineConditioner(rescale="diag") instead.')

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

        argmax = int(np.argmax(self.profiles.logpdf))
        lp_max = float(self.profiles.logpdf[argmax])

        best_orig = np.concatenate([
            np.asarray(self.profiles.best[name][argmax]).ravel()
            for name in self.varied_params.names()
        ])
        center_r = np.asarray(self.conditioning.inverse(best_orig))
        cov_orig = np.asarray(self.profiles.covariance.value)

        contour_pairs = {}
        for name1, name2 in param_pairs:
            flat_idx1 = self._param_slices[name1].start
            flat_idx2 = self._param_slices[name2].start

            # 2×2 covariance in conditioned space
            scale_2   = self.conditioning._scale[[flat_idx1, flat_idx2]]
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

        if self.output_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.output_fn)
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
        if self.conditioning.is_mixing:
            raise ValueError(
                'profile() is not supported when the conditioner mixes parameter dimensions '
                '(AffineConditioner with rescale="full" and a non-diagonal covariance). '
                'Use AffineConditioner(rescale="diag") instead.')

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

        argmax    = int(np.argmax(self.profiles.logpdf))
        profile_results = {}

        for pname, grid_vals, npoints, cl_val in zip(param_names, grids, sizes, cls_):
            flat_pidx = self._param_slices[pname].start  # flat index of this scalar param
            scan = _build_1d_grid(pname, flat_pidx, grid_vals, npoints, cl_val, argmax, self.profiles, self.varied_params)

            scan_r = (scan - self.conditioning._loc[flat_pidx]) / self.conditioning._scale[flat_pidx]
            fixed_points_r  = scan_r.reshape(-1, 1)
            logposteriors   = self._scan([flat_pidx], fixed_points_r, niterations, **kwargs)
            profile_results[pname] = (scan, logposteriors)

        if self.profiles.profile is None:
            self.profiles.profile = {}
        self.profiles.profile.update(profile_results)

        if self.output_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.output_fn)
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
        argmax        = int(np.argmax(self.profiles.logpdf))
        grids1d = [
            _build_1d_grid(pname, flat_pidx, grid_vals, npoints, cl_val, argmax, self.profiles, self.varied_params)
            for pname, flat_pidx, grid_vals, npoints, cl_val
            in zip(param_names, flat_grid_idx, grids, sizes, cls_)
        ]
        mesh       = np.meshgrid(*grids1d, indexing='ij')
        grid_shape = mesh[0].shape
        flat_pts   = np.column_stack([m.ravel() for m in mesh])  # (N, n_grid)

        # Condition fixed points (vectorised over all grid points)
        loc_grid        = self.conditioning._loc[flat_grid_idx]
        scale_grid      = self.conditioning._scale[flat_grid_idx]
        fixed_points_r  = (flat_pts - loc_grid) / scale_grid  # (N, n_grid)

        lp_grid = self._scan(flat_grid_idx, fixed_points_r, niterations, **kwargs).reshape(grid_shape)

        if self.profiles.grid is None:
            self.profiles.grid = {}
        self.profiles.grid.update({
            'params':  param_names,
            'grids':   [dim_grid.tolist() for dim_grid in grids1d],
            'logpdf':  lp_grid,
        })

        if self.output_fn is not None and self.mpicomm.rank == 0:
            self.profiles.write(self.output_fn)
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
        for param in self.varied_params:
            slc = self._param_slices[param.name]
            if set(range(slc.start, slc.stop)) <= set(free_flat_idx):
                free_params.set(param)

        fixed_idx_arr = np.array(fixed_flat_idx, dtype=int)
        free_idx_arr  = np.array(free_flat_idx,  dtype=int) if free_flat_idx else np.array([], dtype=int)

        all_bounds, all_proposals = self._conditioned_bounds_and_proposals()
        flat_bounds_free    = [all_bounds[k]    for k in free_flat_idx]
        flat_proposals_free = [all_proposals[k] for k in free_flat_idx]

        jit_chi2_fixed = grad_chi2_fixed = None
        if free_flat_idx:
            def chi2_2arg(x_free, point_r):
                full = jnp.zeros(flat_size).at[free_idx_arr].set(x_free).at[fixed_idx_arr].set(point_r)
                return self._chi2_rescaled(full)
            jit_chi2_fixed, grad_chi2_fixed = _jit_and_grad(chi2_2arg, with_gradient=self.kernel.with_gradient)

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
            start=starts_free[0],  # placeholder; replaced per start
            bounds=scan_ctx['flat_bounds_free'],
            proposals=scan_ctx['flat_proposals_free'],
            fast=True,
        )

        best_lp   = -np.inf
        best_free = None
        for start in starts_free:
            state = dataclasses.replace(state_tmpl, start=start)
            state = self.kernel.run(state, chi2_fn, grad=grad_fn, **kwargs)
            if state.logpdf is not None:
                lp = float(state.logpdf)
                if lp > best_lp:
                    best_lp = lp
                    best_free = state.best

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
        scale_p      = float(self.conditioning._scale[flat_pidx])
        loc_p        = float(self.conditioning._loc[flat_pidx])
        center_orig  = center_val_r * scale_p + loc_p

        prior_bounds = self.conditioning.prior_bounds()
        lim_lo_r, lim_hi_r = float(prior_bounds[flat_pidx, 0]), float(prior_bounds[flat_pidx, 1])

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

        scale_1 = float(self.conditioning._scale[flat_idx1])
        scale_2 = float(self.conditioning._scale[flat_idx2])
        loc_1   = float(self.conditioning._loc[flat_idx1])
        loc_2   = float(self.conditioning._loc[flat_idx2])
        center_12 = center_r[[flat_idx1, flat_idx2]].copy()

        prior_bounds = self.conditioning.prior_bounds()
        lim_lo_r1, lim_hi_r1 = float(prior_bounds[flat_idx1, 0]), float(prior_bounds[flat_idx1, 1])
        lim_lo_r2, lim_hi_r2 = float(prior_bounds[flat_idx2, 0]), float(prior_bounds[flat_idx2, 1])

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


def _run_one_worker(kernel, state_tmpl, chi2, grad, kwargs, x0):
    """Top-level (picklable) worker for _pool_map in Profiler.maximize()."""
    state = dataclasses.replace(state_tmpl, start=x0)
    return kernel.run(state, chi2, grad=grad, **kwargs)
