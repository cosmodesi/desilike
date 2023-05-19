import numpy as np

from desilike import utils
from desilike.samples.profiles import Profiles, Samples, ParameterBestFit, ParameterCovariance

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
            If input likelihood is a :class:`BaseGaussianLikelihood` instance, or a :class:`SumLikelihood` of such instances,
            :func:`scipy.optimize.curve_fit` can be used; in this case ``method`` can be ‘lsq‘ (i.e. least-squares), or specifically
            choose the :func:`scipy.optimize.curve_fit` algorithm, `lm`, `trf`, or `dogbox`. In this case, priors that have a scale are taken as Gaussian.
            Else, should be one of ‘Nelder-Mead’, ‘Powell’, ‘CG', ‘BFGS’, ‘Newton-CG’, ‘L-BFGS-B’, ‘TNC’,
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

        use_curve_fit = self.method in ['lsq', 'lm', 'trf', 'dogbox']
        method = self.method
        bounds = [tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in varied_params]

        if use_curve_fit:
            if self.method == 'lsq':
                if all(bound == (None, None) for bound in bounds):
                    method = 'lm'
                else:
                    method = 'trf'
            use_curve_fit = {}
            likelihoods = getattr(self.likelihood, 'likelihoods', [self.likelihood])
            from desilike.likelihoods import BaseGaussianLikelihood
            is_gaussian = all(isinstance(likelihood, BaseGaussianLikelihood) for likelihood in likelihoods)
            if not is_gaussian:
                raise ValueError('Cannot choose {} method with non-Gaussian likelihood'.format(method))

            solved_params = self.pipeline.params.select(solved=True)
            if solved_params and self.mpicomm.rank == 0:
                self.log_warning('Analytic marginalization for parameters {} does not work with curve_fit yet.'.format(solved_params.names()))

            params_with_scale = [param for param in self.varied_params + solved_params if getattr(param.prior, 'scale', None) is not None]
            covariance_params = np.array([param.prior.scale**2 for param in params_with_scale])
            center_params = {param.name: param.prior.center() for param in params_with_scale}

            is_2d = any(likelihood.precision.ndim == 2 for likelihood in likelihoods)
            if is_2d:
                covariances = [utils.inv(likelihood.precision) if likelihood.precision.ndim == 2 else np.diag(1. / likelihood.precision) for likelihood in likelihoods]
                size = sum(cov.shape[0] for cov in covariances) + len(covariance_params)
                sigma = np.zeros((size, size), dtype='f8')
                start = 0
                for cov in covariances:
                    stop = start + cov.shape[0]
                    sl = slice(start, stop)
                    sigma[sl, sl] = cov
                    start = stop
                sigma[start:, start:] = np.diag(covariance_params)
            else:
                sigma = np.concatenate([1. / likelihood.precision for likelihood in likelihoods] + [covariance_params])**0.5  # if 1D, give sigma

            def f(x, *values):
                chi2(values)
                return np.concatenate([likelihood.flatdiff for likelihood in likelihoods] + [[self.pipeline.param_values[param.name] - center_params[param.name] for param in params_with_scale]])

            profiles = Profiles()
            bounds = list(zip(*[tuple(param.prior.limits) for param in varied_params]))
            try:
                if tol is not None:
                    kwargs = {'xtol': tol, 'ftol': tol, 'gtol': tol, **kwargs}
                if self.method == 'lm':
                    kwargs.setdefault('maxfev', max_iterations)
                else:
                    kwargs.setdefault('max_nfev', max_iterations)
                popt, pcov = optimize.curve_fit(f, xdata=np.linspace(0., 1., sigma.shape[0]), ydata=np.zeros(sigma.shape[0], dtype='f8'), p0=start, sigma=sigma,
                                                check_finite=True, bounds=bounds, method=self.method, jac=None, **kwargs)
            except RuntimeError:
                if self.mpicomm.rank == 0:
                    self.log_error('Finished unsuccessfully.')
            else:
                profiles.set(bestfit=ParameterBestFit(list(popt) + [- 0.5 * self.chi2(popt)], params=varied_params + ['logposterior']))
                profiles.set(error=Samples(np.diag(pcov)**0.5, params=varied_params))
                profiles.set(covariance=ParameterCovariance(pcov, params=varied_params))
        else:
            bounds = [tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in varied_params]
            result = optimize.minimize(fun=chi2, x0=start, method=method, bounds=bounds, tol=tol, options={'maxiter': max_iterations, **kwargs})
            if not result.success and self.mpicomm.rank == 0:
                self.log_error('Finished unsuccessfully.')
            profiles = Profiles()
            profiles.set(bestfit=ParameterBestFit(list(result.x) + [- 0.5 * result.fun], params=varied_params + ['logposterior']))
            if getattr(result, 'hess_inv', None) is not None:
                cov = result.hess_inv.todense()
                profiles.set(error=Samples(np.diag(cov)**0.5, params=varied_params))
                profiles.set(covariance=ParameterCovariance(cov, params=varied_params))
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