"""ScipyProfiler — wraps scipy.optimize.minimize."""

import logging

import numpy as np

from ..samples import Profiles
from .base import BaseProfiler, ProfilerState, _build_best_from_x, _build_error_from_cov


class ScipyProfiler(BaseProfiler):
    """Profiler backed by :func:`scipy.optimize.minimize`.

    Supports all scipy methods; gradient-based methods benefit from setting
    ``gradient=True`` (requires a JAX-traceable likelihood).

    Parameters
    ----------
    method : str, optional
        Scipy solver name, e.g. ``'L-BFGS-B'``, ``'Nelder-Mead'``,
        ``'BFGS'``.  When ``None`` scipy picks automatically.
    gradient : bool
        If ``True``, pass the JAX-computed gradient to scipy (only
        effective for gradient-supporting methods).
    *args, **kwargs
        Forwarded to :class:`BaseProfiler`.
    """

    logger = logging.getLogger('ScipyProfiler')

    name = 'scipy'

    def __init__(self, *args, method=None, gradient=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.with_gradient = bool(gradient)

    def _maximize_one(self, state: ProfilerState, max_iterations=int(1e5), tol=None, **kwargs):
        from scipy import optimize

        bounds = [
            (None if np.isinf(lo) else lo, None if np.isinf(hi) else hi)
            for lo, hi in state.flat_bounds
        ]
        call_kw = {}
        if state.grad_fn is not None:
            call_kw['jac'] = state.grad_fn

        try:
            result = optimize.minimize(
                fun=state.chi2_fn,
                x0=state.start,
                method=self.method,
                bounds=bounds,
                tol=tol,
                options={'maxiter': max_iterations, **kwargs},
                **call_kw,
            )
        except Exception as exc:
            self.logger.warning('scipy.minimize raised %r', exc)
            return Profiles()

        if not result.success:
            self.logger.warning('scipy.minimize: %s', result.message)

        logpost  = float(-0.5 * result.fun)
        profiles = Profiles()
        profiles.best = _build_best_from_x(result.x, logpost, state.varied_params)
        profiles.logpdf = np.array([logpost])

        # Covariance from inverse Hessian (L-BFGS-B / BFGS provide this)
        hess_inv = getattr(result, 'hess_inv', None)
        if hess_inv is not None and not state.fast:
            try:
                cov = np.asarray(getattr(hess_inv, 'todense', lambda: hess_inv)())
                profiles.error = _build_error_from_cov(cov, state.varied_params)
            except Exception:
                pass

        return profiles
