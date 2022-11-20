import numpy as np

from desilike.samples.profiles import Profiles, Samples, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


class ScipyProfiler(BaseProfiler):

    def __init__(self, *args, method=None, **kwargs):
        """
        Wrapper for collection of scipy's samplers.

        Parameters
        ----------
        method : str, default=None
            Type of solver. Should be one of ‘Nelder-Mead’, ‘Powell’, ‘CG', ‘BFGS’, ‘Newton-CG’, ‘L-BFGS-B’, ‘TNC’,
            ‘COBYLA’, ‘SLSQP’ , ‘trust-constr’, ‘dogleg’, ‘trust-ncg’, ‘trust-exact’, ‘trust-krylov’.
            If not given, chosen to be one of ‘BFGS‘, ‘L-BFGS-B‘, ‘SLSQP‘, depending on whether or not the problem has constraints or bounds.
        """
        super(ScipyProfiler, self).__init__(*args, **kwargs)
        self.method = method

    def _maximize_one(self, start, max_iterations=int(1e5), tol=None, **kwargs):
        """
        Parameters
        ----------
        tol : float, default=None
            Tolerance for termination. When tol is specified, the selected minimization algorithm sets some relevant solver-specific tolerance(s)
            equal to tol. For detailed control, use solver-specific options.

        kwargs : dict
            Solver-specific options.
        """
        from scipy import optimize
        bounds = [tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in self.varied_params]
        result = optimize.minimize(fun=self.chi2, x0=start, method=self.method, bounds=bounds, tol=tol, options={'maxiter': max_iterations, **kwargs})
        if not result.success and self.mpicomm.rank == 0:
            self.log_error('Finished unsuccessfully.')
        profiles = Profiles()
        profiles.set(bestfit=ParameterBestFit(list(result.x) + [- 0.5 * result.fun], params=self.varied_params + ['logposterior']))
        if getattr(result, 'hess_inv', None) is not None:
            cov = result.hess_inv.todense()
            profiles.set(error=Samples(np.diag(cov)**0.5, params=self.varied_params))
            profiles.set(covariance=ParameterCovariance(cov, params=self.varied_params))
        return profiles
