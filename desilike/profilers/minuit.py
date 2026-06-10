"""MinuitProfiler — wraps iminuit.Minuit."""

import logging

import numpy as np

from ..samples import Profiles
from .base import BaseProfiler, ProfilerState, _build_best_from_x, _build_error_from_cov


class MinuitProfiler(BaseProfiler):
    """Profiler backed by `iminuit <https://github.com/scikit-hep/iminuit>`_.

    Minuit is the de-facto standard minimiser for high-energy physics
    likelihoods.  It provides reliable parabolic errors (HESSE) and MINOS
    confidence intervals.

    Only scalar parameters are supported (iminuit uses named scalar values).

    Parameters
    ----------
    gradient : bool
        If ``True`` and JAX differentation of the likelihood succeeds,
        pass the gradient to Minuit (can significantly speed up convergence).
    *args, **kwargs
        Forwarded to :class:`BaseProfiler`.
    """

    logger = logging.getLogger('MinuitProfiler')

    name = 'minuit'

    def __init__(self, *args, gradient=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_gradient = bool(gradient)

    @classmethod
    def install(cls, installer):
        installer.pip('iminuit')

    def _maximize_one(self, state: ProfilerState, max_iterations=int(1e5), **kwargs):
        import iminuit

        names = list(state.varied_params.names())

        # iminuit requires scalar-valued parameters.
        for param in state.varied_params:
            if param.shape:
                raise ValueError(
                    f'MinuitProfiler does not support non-scalar parameters; '
                    f'{param.name!r} has shape {param.shape}. '
                    'Use ScipyProfiler or BOBYQAProfiler instead.'
                )

        # iminuit expects a function of positional args
        def chi2m(*values):
            return state.chi2_fn(np.asarray(values))

        minuit_kw = {'name': names}
        if state.grad_fn is not None:
            def gradm(*values):
                return np.asarray(state.grad_fn(np.asarray(values)))
            minuit_kw['grad'] = gradm

        minuit = iminuit.Minuit(
            chi2m,
            **dict(zip(names, state.start)),
            **minuit_kw,
        )
        minuit.errordef = 1.0
        minuit.strategy = 0 if state.fast else 1

        # One flat element per scalar param; flat_bounds/flat_proposals align 1:1.
        for param, (lo, hi), proposal in zip(state.varied_params, state.flat_bounds, state.flat_proposals):
            minuit.values[param.name] = float(state.start[names.index(param.name)])
            minuit.limits[param.name] = (None if np.isinf(lo) else lo,
                                         None if np.isinf(hi) else hi)
            if np.isfinite(proposal):
                minuit.errors[param.name] = proposal

        profiles = Profiles()

        try:
            minuit.migrad(ncall=max_iterations)
        except RuntimeError as exc:
            self.logger.warning('migrad failed: %r', exc)
            return profiles

        # HESSE for accurate errors and covariance
        if not state.fast:
            try:
                minuit.hesse()
            except RuntimeError as exc:
                self.logger.warning('hesse failed: %r', exc)

        logpost = float(-0.5 * minuit.fval)
        result_x = np.array([float(minuit.values[name]) for name in names])
        profiles.best = _build_best_from_x(result_x, logpost, state.varied_params)
        profiles.logpdf = np.array([logpost])

        if not state.fast:
            if minuit.covariance is not None:
                from ..samples import Covariance
                cov = np.asarray(minuit.covariance)
                profiles.error = _build_error_from_cov(cov, state.varied_params)
                profiles.covariance = Covariance(cov, list(state.varied_params))
            else:
                profiles.error = {
                    name: np.array([float(minuit.errors[name])]) for name in names
                }
        return profiles
