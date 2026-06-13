"""BOBYQA profiler kernel — wraps pybobyqa (derivative-free, bound-constrained)."""

import logging

import numpy as np

from .base import Kernel, ProfilerState


class BOBYQA(Kernel):
    """Derivative-free optimisation kernel using `Py-BOBYQA
    <https://github.com/numericalalgorithmsgroup/pybobyqa>`_.

    Well-suited for noisy or expensive likelihoods where automatic
    differentiation is unavailable or unreliable.
    """

    logger = logging.getLogger('BOBYQA')

    @classmethod
    def install(cls, installer):
        installer.pip('pybobyqa')

    def init(self):
        try:
            import pybobyqa  # noqa: F401
        except ImportError:
            raise ImportError("'pybobyqa' is required but not installed. Run: pip install pybobyqa")

    def run(self, state: ProfilerState, chi2, grad=None, max_iterations=int(1e5), **kwargs) -> ProfilerState:
        import pybobyqa

        _INF_PROXY = 1e20   # pybobyqa requires finite bounds
        bounds = np.array([
            [
                -_INF_PROXY if np.isinf(lo) else lo,
                 _INF_PROXY if np.isinf(hi) else hi,
            ]
            for lo, hi in state.bounds
        ]).T   # shape (2, flat_size)

        try:
            logging.getLogger('pybobyqa').setLevel(logging.WARNING)
            result = pybobyqa.solve(
                objfun=chi2,
                x0=state.start,
                bounds=tuple(bounds),
                maxfun=max_iterations,
                **kwargs,
            )
        except Exception as exc:
            self.logger.warning('pybobyqa.solve raised %r', exc)
            return state

        if result.flag != result.EXIT_SUCCESS:
            self.logger.warning('pybobyqa finished with flag %s: %s', result.flag, result.msg)

        state.logpdf = np.asarray(-0.5 * result.f)
        state.best = np.asarray(result.x)

        try:
            state.cov = np.linalg.inv(result.hessian)
        except Exception:
            pass

        return state
