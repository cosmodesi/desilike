"""BOBYQAProfiler — wraps pybobyqa (derivative-free, bound-constrained)."""

import logging

import numpy as np

from ..samples import Profiles
from .base import BaseProfiler, ProfilerState, _build_best_from_x, _build_error_from_cov


class BOBYQAProfiler(BaseProfiler):
    """Derivative-free profiler using `Py-BOBYQA
    <https://github.com/numericalalgorithmsgroup/pybobyqa>`_.

    BOBYQA is well-suited for noisy or expensive likelihoods where
    automatic differentiation is unavailable or unreliable.

    *args, **kwargs
        Forwarded to :class:`BaseProfiler`.
    """

    logger = logging.getLogger('BOBYQAProfiler')

    name = 'bobyqa'

    @classmethod
    def install(cls, installer):
        installer.pip('pybobyqa')

    def _maximize_one(self, state: ProfilerState, max_iterations=int(1e5), **kwargs):
        import pybobyqa

        _INF_PROXY = 1e20   # pybobyqa requires finite bounds
        bounds = np.array([
            [
                -_INF_PROXY if np.isinf(lo) else lo,
                 _INF_PROXY if np.isinf(hi) else hi,
            ]
            for lo, hi in state.flat_bounds
        ]).T   # shape (2, flat_size)

        profiles = Profiles()
        try:
            # pybobyqa is chatty; silence it
            logging.getLogger('pybobyqa').setLevel(logging.WARNING)

            result = pybobyqa.solve(
                objfun=state.chi2_fn,
                x0=state.start,
                bounds=tuple(bounds),
                maxfun=max_iterations,
                **kwargs,
            )
        except Exception as exc:
            self.logger.warning('pybobyqa.solve raised %r', exc)
            return profiles

        if result.flag != result.EXIT_SUCCESS:
            self.logger.warning('pybobyqa finished with flag %s: %s', result.flag, result.msg)

        logpost = float(-0.5 * result.f)
        profiles.best = _build_best_from_x(result.x, logpost, state.varied_params)

        # Covariance from Hessian approximation
        try:
            cov = np.linalg.inv(result.hessian)
            profiles.error = _build_error_from_cov(cov, state.varied_params)
        except Exception:
            pass

        return profiles
