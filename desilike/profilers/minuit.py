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

    name = 'minuit'

    def __init__(self, *args, **kwargs):
        """Wrapper for minuit profiler, used by the high-energy physics community for likelihood profiling."""
        super(MinuitProfiler, self).__init__(*args, **kwargs)
        import iminuit
        minuit_params = {}
        minuit_params['name'] = parameter_names = [str(param) for param in self.varied_params]
        self.minuit = iminuit.Minuit(self.chi2, **dict(zip(parameter_names, [param.value for param in self.varied_params])), **minuit_params)
        self.minuit.errordef = 1.0
        for param in self.varied_params:
            self.minuit.limits[str(param)] = tuple(None if np.isinf(lim) else lim for lim in param.prior.limits)
            if param.ref.is_proper():
                self.minuit.errors[str(param)] = param.proposal

    def chi2(self, *values):
        return super(MinuitProfiler, self).chi2(values)

    def _set_start(self, start):
        for param, value in zip(self.varied_params, start):
            self.minuit.values[str(param)] = value

    def _maximize_one(self, start, max_iterations=int(1e5)):
        self._set_start(start)
        self.minuit.migrad(ncall=max_iterations)
        profiles = Profiles()
        profiles.set(start=Samples(start, params=self.varied_params))
        profiles.set(bestfit=ParameterBestFit([self.minuit.values[str(param)] for param in self.varied_params] + [- 0.5 * self.minuit.fval], params=self.varied_params + ['logposterior']))
        profiles.set(error=Samples([self.minuit.errors[str(param)] for param in self.varied_params], params=self.varied_params))
        profiles.set(covariance=ParameterCovariance(np.array(self.minuit.covariance), params=self.varied_params))
        return profiles

    def _interval_one(self, start, param, max_iterations=int(1e5), cl=None):
        """
        Parameters
        ----------
        cl : float, int, default=None
            Confidence level for the confidence interval.
            If not set or None, a standard 68.3 % confidence interval is produced.
            If 0 < cl < 1, the value is interpreted as the confidence level (a probability).
            If cl >= 1, it is interpreted as number of standard deviations. For example, cl = 3 produces a 3 sigma interval.
        """
        self._set_start(start)
        profiles = Profiles()
        name = str(param)
        self.minuit.minos(name, ncall=max_iterations, cl=cl)
        interval = (self.minuit.merrors[name].lower, self.minuit.merrors[name].upper)
        profiles.set(interval=Samples([interval], params=[param]))

        return profiles

    def _profile_one(self, start, param, size=30, grid=None, **kwargs):
        """
        Parameters
        ----------
        size : int, default=30
            Number of scanning points. Ignored if grid is set.

        bound : tuple, int, default=2
            If bound is tuple, (left, right) scanning bound. If bound is a number, it specifies an interval of N sigmas
            symmetrically around the minimum. Ignored if grid is set.

        grid : array, default=None
            Parameter values on which to compute the profile. If grid is set, size and bound are ignored.
        """
        self._set_start(start)
        profiles = Profiles()
        if 'cl' in kwargs:
            kwargs['bound'] = kwargs.pop('cl')
        if not np.isinf(param.prior.limits).any():
            kwargs.setdefault('bound', param.prior.limits)
        x, chi2 = self.minuit.mnprofile(param.name, size=size, grid=grid, **kwargs)[:2]
        profiles.set(profile=Samples([(x, chi2)], params=[param]))

        return profiles

    def _contour_one(self, start, param1, param2, cl=None, size=100, interpolated=0):
        """
        cl : float, int, default=None
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
        self._set_start(start)
        profiles = Profiles()
        x1, x2 = self.minuit.mncontour(str(param1), str(param2), cl=cl, size=size, interpolated=interpolated)
        profiles.set(profile=ParameterContours([(ParameterArray(x1, param1), ParameterArray(x2, param2))]))
        return profiles

    @classmethod
    def install(cls, config):
        config.pip('iminuit')
