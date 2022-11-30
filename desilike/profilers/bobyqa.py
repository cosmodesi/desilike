import numpy as np

from desilike import utils
from desilike.samples.profiles import Profiles, Samples, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


class BOBYQAProfiler(BaseProfiler):

    name = 'bobyqa'
    """
    Designed for solving bound-constrained general objective minimization, without requiring derivatives of the objective.
    See https://github.com/numericalalgorithmsgroup/pybobyqa.
    """
    def _maximize_one(self, start, max_iterations=int(1e5), **kwargs):
        r"""
        Maximize.

        Parameters
        ----------
        npt : int, default=None
            The number of interpolation points to use; default is 2 * ndim + 1.
            Py-BOBYQA requires ndim + 1 <= npt <= (ndim + 1)(ndim + 2)/2. Larger values are particularly useful for noisy problems.

        rhobeg : float, default=None
            The initial value of the trust region radius default is :math:`0.1 max(|x_0|_{\infty}, 1)`.

        rhoend : float, default=1e-8
            Minimum allowed value of trust region radius, which determines when a successful termination occurs.

        seek_global_minimum : bool, default=False
            A flag to indicate whether to search for a global minimum, rather than a local minimum.
            This is used to set some sensible default parameters, all of which can be overridden by the values provided in user_params.
            If True, both upper and lower bounds must be set. Note that Py-BOBYQA only implements a heuristic method,
            so there are no guarantees it will find a global minimum. However, by using this flag,
            it is more likely to escape local minima if there are better values nearby.
            The method used is a multiple restart mechanism, where we repeatedly re-initialize Py-BOBYQA from the best point found so far,
            but where we use a larger trust reigon radius each time (note: this is different to more common multi-start approach to global optimization).
        """
        import pybobyqa
        infs = [- 1e20, 1e20]  # pybobyqa defaults
        bounds = np.array([[inf if np.isinf(lim) else lim for lim, inf in zip(param.prior.limits, infs)] for param in self.varied_params]).T
        result = pybobyqa.solve(objfun=self.chi2, x0=start, bounds=bounds, maxfun=max_iterations, **kwargs)
        success = result.flag == result.EXIT_SUCCESS
        profiles = Profiles()
        if not success and self.mpicomm.rank == 0:
            self.log_error('Finished unsuccessfully.')
            return profiles
        profiles.set(bestfit=ParameterBestFit(list(result.x) + [- 0.5 * result.f], params=self.varied_params + ['logposterior']))
        cov = utils.inv(result.hessian)
        profiles.set(error=Samples(np.diag(cov)**0.5, params=self.varied_params))
        profiles.set(covariance=ParameterCovariance(cov, params=self.varied_params))
        print(profiles.error)
        return profiles

    @classmethod
    def install(cls, config):
        config.pip('pybobyqa')
