"""Scipy profiler kernel — wraps scipy.optimize.minimize."""

import logging

import numpy as np

from .base import Kernel, ProfilerState


class Scipy(Kernel):
    """Optimisation kernel backed by :func:`scipy.optimize.minimize`.

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
    """

    logger = logging.getLogger('Scipy')

    def __init__(self, method=None, gradient=False):
        self.method = method
        self.with_gradient = bool(gradient)

    def run(self, state: ProfilerState, chi2, grad=None, max_iterations=int(1e5), tol=None, **kwargs) -> ProfilerState:
        from scipy import optimize

        bounds = [
            (None if np.isinf(lo) else lo, None if np.isinf(hi) else hi)
            for lo, hi in state.bounds
        ]
        call_kw = {}
        if grad is not None:
            call_kw['jac'] = grad

        try:
            result = optimize.minimize(
                fun=chi2,
                x0=state.start,
                method=self.method,
                bounds=bounds,
                tol=tol,
                options={'maxiter': max_iterations, **kwargs},
                **call_kw,
            )
        except Exception as exc:
            self.logger.warning('scipy.minimize raised %r', exc)
            return state

        if not result.success:
            self.logger.warning('scipy.minimize: %s', result.message)

        state.logpdf = np.asarray(-0.5 * result.fun)
        state.best = np.asarray(result.x)

        hess_inv = getattr(result, 'hess_inv', None)
        if hess_inv is not None and not state.fast:
            try:
                state.cov = np.asarray(getattr(hess_inv, 'todense', lambda: hess_inv)())
            except Exception:
                pass

        return state
