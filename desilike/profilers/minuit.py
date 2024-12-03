import numpy as np

from desilike.samples.profiles import Profiles, ParameterArray, Samples, ParameterContours, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


def _get_options(name, **kwargs):
    if name in kwargs:
        toret = kwargs[name]
        if toret is None: toret = {}
        return toret
    return None


class MinuitProfiler(BaseProfiler):

    """
    Wrapper for minuit profiler, used by the high-energy physics community for likelihood profiling.

    Reference
    ---------
    - https://github.com/scikit-hep/iminuit
    - https://ui.adsabs.harvard.edu/abs/1975CoPhC..10..343J/abstract
    """
    name = 'minuit'

    def __init__(self, *args, gradient=False, **kwargs):
        """
        Initialize profiler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        rng : np.random.RandomState, default=None
            Random state. If ``None``, ``seed`` is used to set random state.

        seed : int, default=None
            Random seed.

        max_tries : int, default=1000
            A :class:`ValueError` is raised after this number of likelihood (+ prior) calls without finite posterior.

        profiles : str, Path, Profiles
            Path to or profiles, to which new profiling results will be added.

        ref_scale : float, default=1.
            Rescale parameters' :attr:`Parameter.ref` reference distribution by this factor

        rescale : bool, default=False
            If ``True``, internally rescale parameters such their variation range is ~ unity.
            Provide ``covariance`` to take parameter variations from;
            else parameters' :attr:`Parameter.proposal` will be used.

        covariance : str, Path, ParameterCovariance, Chain, default=None
            If ``rescale``, path to or covariance or chain, which is used for rescaling parameters.
            If ``None``, parameters' :attr:`Parameter.proposal` will be used instead.

        gradient : bool, default=False
            If ``True``, try to take the likelihood gradient (requires jax).

        save_fn : str, Path, default=None
            If not ``None``, save profiles to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`
        """
        super(MinuitProfiler, self).__init__(*args, **kwargs)
        self.with_gradient = bool(gradient)

    def _get_minuit(self, start, chi2, varied_params, gradient=None):

        def chi2m(*values):
            return chi2(values)

        import iminuit
        minuit_params = {}
        minuit_params['name'] = parameter_names = [str(param) for param in varied_params]

        if gradient is not None:

            def gradientm(*values):
                return gradient(values)

            minuit_params['grad'] = gradientm

        minuit = iminuit.Minuit(chi2m, **dict(zip(parameter_names, [param.value for param in varied_params])), **minuit_params)
        minuit.errordef = 1.0
        for param, value in zip(varied_params, start):
            minuit.values[str(param)] = value
            minuit.limits[str(param)] = tuple(None if np.isinf(lim) else lim for lim in param.prior.limits)
            if param.ref.is_proper():
                minuit.errors[str(param)] = param.proposal
        return minuit

    def maximize(self, *args, **kwargs):
        r"""
        Maximize :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.start`
        - :attr:`Profiles.bestfit`
        - :attr:`Profiles.error`  # parabolic errors at best fit
        - :attr:`Profiles.covariance`  # parameter covariance at best fit

        One will typically run several independent likelihood maximizations in parallel,
        on number of MPI processes - 1 ranks (1 if single process), to make sure the global maximum is found.

        Parameters
        ----------
        niterations : int, default=None
            Number of iterations, i.e. of runs of the profiler from independent starting points.
            If ``None``, defaults to :attr:`mpicomm.size - 1` (if > 0, else 1).

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.
        """
        return super(MinuitProfiler, self).maximize(*args, **kwargs)

    def _maximize_one(self, start, chi2, varied_params, max_iterations=int(1e5), gradient=None):
        minuit = self._get_minuit(start, chi2, varied_params, gradient=gradient)
        profiles = Profiles()
        try:
            minuit.migrad(ncall=max_iterations)
        except RuntimeError as exc:
            if self.mpicomm.rank == 0:
                self.log_warning('maximize failed: {}'.format(exc))
            return profiles
        try:
            minuit.hesse()
        except RuntimeError as exc:
            if self.mpicomm.rank == 0:
                self.log_warning('hesse failed: {}'.format(exc))
        bestfit_attrs = {name: getattr(minuit.fmin, name) for name in ['nfcn', 'ngrad', 'is_valid', 'is_above_max_edm', 'has_reached_call_limit', 'time']}
        covariance_attrs = {name: getattr(minuit.fmin, name) for name in ['has_accurate_covar', 'has_posdef_covar', 'has_made_posdef_covar']}
        profiles.set(bestfit=ParameterBestFit([np.atleast_1d(minuit.values[str(param)]) for param in varied_params] + [- 0.5 * np.atleast_1d(minuit.fval)], params=varied_params + ['logposterior'], attrs=bestfit_attrs))
        profiles.set(error=Samples([np.atleast_1d(minuit.errors[str(param)]) for param in varied_params], params=varied_params, attrs=covariance_attrs))
        if minuit.covariance is not None:
            profiles.set(covariance=ParameterCovariance(np.array(minuit.covariance), params=varied_params, attrs=covariance_attrs))
        return profiles

    def interval(self, *args, **kwargs):
        """
        Compute confidence intervals for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.interval`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to estimate confidence intervals.

        cl : float, int, default=None
            Confidence level for the confidence interval.
            If not set or None, a standard 68.3 % confidence interval is produced.
            If 0 < cl < 1, the value is interpreted as the confidence level (a probability).
            If cl >= 1, it is interpreted as number of standard deviations. For example, cl = 3 produces a 3 sigma interval.
        """
        return super(MinuitProfiler, self).interval(*args, **kwargs)

    def _interval_one(self, start, chi2, varied_params, param, max_iterations=int(1e5), cl=None, gradient=None):
        minuit = self._get_minuit(start, chi2, varied_params, gradient=gradient)
        profiles = Profiles()
        name = str(param)
        try:
            minuit.minos(name, ncall=max_iterations, cl=cl)  # minimum not found
        except RuntimeError as exc:
            if self.mpicomm.rank == 0:
                self.log_warning('interval failed: {}'.format(exc))
            return profiles
        merrors = minuit.merrors[name]
        interval = np.array([merrors.lower, merrors.upper])
        attrs = {name: getattr(merrors, name) for name in ['is_valid', 'lower_valid', 'upper_valid', 'at_lower_limit', 'at_upper_limit', 'at_lower_max_fcn', 'at_upper_max_fcn',
                                                           'lower_new_min', 'upper_new_min', 'nfcn', 'min']}
        profiles.set(interval=Samples([interval], params=[param], attrs={name: attrs}))
        return profiles

    def contour(self, *args, **kwargs):
        """
        Compute 2D contours for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.contour`

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            List of tuples of parameters for which to compute 2D contours.
            If a list of parameters is provided instead, contours are computed for unique tuples of parameters.

        cl : float, int, default=1
            Confidence level for the confidence contour.
            If not set or None, a standard 68.3 % confidence contour is produced.
            If 0 < cl < 1, the value is interpreted as the confidence level (a probability).
            If cl >= 1, it is interpreted as number of standard deviations. For example, cl = 3 produces a 3 sigma contour.

        size : int, default=100
            Number of points on the contour to find. Increasing this makes the contour smoother, but requires more computation time.

        interpolated : int, default=0
            Number of interpolated points on the contour. If you set this to a value larger than size,
            cubic spline interpolation is used to generate a smoother curve and the interpolated coordinates are returned.
            Values smaller than size are ignored. Good results can be obtained with size=20, interpolated=200.
        """
        return super(MinuitProfiler, self).contour(*args, **kwargs)

    def _contour_one(self, start, chi2, varied_params, params, cl=None, size=40, interpolated=0, gradient=None, **kwargs):
        param1, param2 = params
        minuit = self._get_minuit(start, chi2, varied_params, gradient=gradient)
        profiles = Profiles()
        try:
            x1x2 = minuit.mncontour(str(param1), str(param2), cl=cl, size=size, interpolated=interpolated, **kwargs)
            if not len(x1x2):
                raise RuntimeError('mncontour is empty')
        except RuntimeError as exc:
            if self.mpicomm.rank == 0:
                self.log_warning('contour failed: {}'.format(exc))
            return profiles
        x1, x2 = x1x2.T
        profiles.set(contour=ParameterContours({cl: [(ParameterArray(x1, param1, copy=True), ParameterArray(x2, param2, copy=True))]}))
        return profiles

    def profile(self, *args, **kwargs):
        """
        Compute 1D profiles for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.profile`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to compute 1D profiles.

        grid : array, list, default=None
            Parameter values on which to compute the profile, for each parameter. If grid is set, size and bound are ignored.

        size : int, list, default=30
            Number of scanning points. Ignored if grid is set. Can be specified for each parameter.

        cl : int, list, default=2
            If bound is a number, it specifies an interval of N sigmas symmetrically around the minimum.
            Ignored if grid is set. Can be specified for each parameter.

        niterations : int, default=1
            Number of iterations, i.e. of runs of the profiler from independent starting points.

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.
        """
        return super(MinuitProfiler, self).profile(*args, **kwargs)

    def grid(self, *args, **kwargs):
        """
        Compute best fits on grid for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.grid`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to compute 1D profiles.

        grid : array, list, dict, default=None
            Parameter values on which to compute the profile, for each parameter. If grid is set, size and bound are ignored.

        size : int, list, dict, default=1
            Number of scanning points. Ignored if grid is set. Can be specified for each parameter.

        cl : int, list, dict, default=2
            If bound is a number, it specifies an interval of N sigmas symmetrically around the minimum.
            Ignored if grid is set. Can be specified for each parameter.

        niterations : int, default=1
            Number of iterations, i.e. of runs of the profiler from independent starting points.

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.
        """
        return super(MinuitProfiler, self).grid(*args, **kwargs)

    @classmethod
    def install(cls, config):
        config.pip('iminuit')
