"""Minuit profiler kernel — wraps iminuit."""

import logging

import numpy as np

from .base import Kernel, ProfilerState


class Minuit(Kernel):
    """Optimisation kernel backed by `iminuit <https://github.com/scikit-hep/iminuit>`_.

    Minuit is the de-facto standard minimiser for high-energy physics
    likelihoods.  It provides reliable parabolic errors (HESSE).

    Only scalar parameters are supported (iminuit uses named scalar values).

    Parameters
    ----------
    gradient : bool
        If ``True``, pass the JAX-computed gradient to Minuit (can speed up
        convergence significantly).
    precision : float or None
        Relative numerical precision of the cost function.  When ``None``
        (default) iminuit assumes machine precision (~2e-16), which causes
        HESSE to use finite-difference step sizes of order 1e-8 — too small
        for emulator-backed likelihoods whose GP interpolation noise is
        typically ~1e-4 to 1e-5.  Setting this to the emulator noise level
        (e.g. ``1e-4``) makes HESSE use correspondingly larger steps, giving
        reliable parabolic error estimates.  For COMET-emu a value around
        ``1e-4`` is appropriate (the GP chi2 noise-to-signal ratio is ~3e-5).
    """

    logger = logging.getLogger('Minuit')

    def __init__(self, gradient=False, precision=None):
        self.with_gradient = bool(gradient)
        self.precision = precision

    @classmethod
    def install(cls, installer):
        installer.pip('iminuit')

    def init(self):
        try:
            import iminuit  # noqa: F401
        except ImportError:
            raise ImportError("'iminuit' is required but not installed. Run: pip install iminuit")

    def run(self, state: ProfilerState, chi2, grad=None, max_iterations=int(1e5), **kwargs) -> ProfilerState:
        import iminuit

        def chi2m(*values):
            return chi2(np.asarray(values))

        minuit_kw = {}
        if grad is not None:
            def gradm(*values):
                return np.asarray(grad(np.asarray(values)))
            minuit_kw['grad'] = gradm

        minuit = iminuit.Minuit(chi2m, *state.start, **minuit_kw)
        minuit.errordef = 1.0
        minuit.strategy = 0 if state.fast else 1
        if self.precision is not None:
            minuit.precision = float(self.precision)

        for param_idx, ((lo, hi), proposal) in enumerate(zip(state.bounds, state.proposals)):
            minuit.limits[param_idx] = (None if np.isinf(lo) else lo,
                                        None if np.isinf(hi) else hi)
            if np.isfinite(proposal):
                minuit.errors[param_idx] = proposal

        try:
            minuit.migrad(ncall=max_iterations)
        except RuntimeError as exc:
            self.logger.warning('migrad failed: %r', exc)
            return state

        if not state.fast:
            try:
                minuit.hesse()
            except RuntimeError as exc:
                self.logger.warning('hesse failed: %r', exc)

        state.logpdf = np.array(-0.5 * minuit.fval)
        state.best = np.array(list(minuit.values))

        if not state.fast and minuit.covariance is not None:
            state.cov = np.asarray(minuit.covariance)

        return state
