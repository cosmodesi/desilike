import numpy as np

from desilike import utils
from desilike.samples.profiles import Profiles, Samples, ParameterBestFit, ParameterCovariance
from desilike.jax import numpy_jax

from .base import BaseProfiler


class ScipyProfiler(BaseProfiler):

    """
    Wrapper for the collection of scipy's profilers (including curve_fit).

    Reference
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    """

    def __init__(self, *args, method=None, **kwargs):
        """
        Initialize profiler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        method : str, default=None
            Type of solver.
            Should be one of ‘Nelder-Mead’, ‘Powell’, ‘CG', ‘BFGS’, ‘Newton-CG’, ‘L-BFGS-B’, ‘TNC’,
            ‘COBYLA’, ‘SLSQP’, ‘trust-constr’, ‘dogleg’, ‘trust-ncg’, ‘trust-exact’, ‘trust-krylov’.
            If not given, chosen to be one of ‘BFGS‘, ‘L-BFGS-B‘, ‘SLSQP‘, depending on whether or not sompe parameters have bounded priors.

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

        save_fn : str, Path, default=None
            If not ``None``, save profiles to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`
        """
        super(ScipyProfiler, self).__init__(*args, **kwargs)
        self.method = method

    def maximize(self, *args, **kwargs):
        r"""
        Maximize :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.start`
        - :attr:`Profiles.bestfit`
        - :attr:`Profiles.error`  # parabolic errors at best fit (if made available by the solver)
        - :attr:`Profiles.covariance`  # parameter covariance at best fit (if made available by the solver).

        One will typically run several independent likelihood maximizations in parallel,
        on number of MPI processes - 1 ranks (1 if single process), to make sure the global maximum is found.

        Parameters
        ----------
        niterations : int, default=None
            Number of iterations, i.e. of runs of the profiler from independent starting points.
            If ``None``, defaults to :attr:`mpicomm.size - 1` (if > 0, else 1).

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.

        tol : float, default=None
            Tolerance for termination. When ``tol`` is specified, the selected minimization algorithm sets some relevant solver-specific tolerance(s)
            equal to ``tol``. For detailed control, use solver-specific options.

        kwargs : dict
            Solver-specific options.
        """
        return super(ScipyProfiler, self).maximize(*args, **kwargs)

    def _maximize_one(self, start, chi2, varied_params, max_iterations=int(1e5), tol=None, **kwargs):
        from scipy import optimize
        bounds = [tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in varied_params]
        try:
            result = optimize.minimize(fun=chi2, x0=start, method=self.method, bounds=bounds, tol=tol, options={'maxiter': max_iterations, **kwargs})
        except RuntimeError:
            if self.mpicomm.rank == 0:
                self.log_warning('Finished unsuccessfully.')
            return profiles
        if not result.success and self.mpicomm.rank == 0:
            self.log_error('Finished unsuccessfully.')
        profiles = Profiles()
        attrs = {name: getattr(result, name) for name in ['success', 'status', 'message', 'nit']}
        profiles.set(bestfit=ParameterBestFit([np.atleast_1d(xx) for xx in result.x] + [- 0.5 * np.atleast_1d(result.fun)], params=varied_params + ['logposterior']), attrs=attrs)
        if getattr(result, 'hess_inv', None) is not None:
            cov = result.hess_inv.todense()
            profiles.set(error=Samples(np.diag(cov)**0.5, params=varied_params, attrs=attrs))
            profiles.set(covariance=ParameterCovariance(cov, params=varied_params, attrs=attrs))
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
        return super(ScipyProfiler, self).profile(*args, **kwargs)

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
        return super(ScipyProfiler, self).grid(*args, **kwargs)