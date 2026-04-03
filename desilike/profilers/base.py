"""Base class for profilers."""

# this is a rough first draft
# TODO: properly implement pool
# TODO: implement other minimizers
# TODO: add plotting functions
# TODO: expand functionality such as warm starts
# TODO: add IO
# TODO: add remove_duplicates

import numpy as np
from scipy.optimize import minimize
from functools import partial

from desilike import Samples
from desilike.pool import MPIPool
from desilike.utils import BaseClass


class Profiler(BaseClass):
    """Profiler used to find maximum likelihood or posterior profiles."""

    def __init__(self, likelihood, posterior=True, rng=None, directory=None):
        """Initialize the profiler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to profile.
        posterior : bool, optional
            If ``True``, profile the posterior. Otherwise, profile the
            likelihood. Default is ``True``.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this folder. Default is ``None``.

        """
        self.likelihood = likelihood
        if posterior:
            self.neg_cost_key = 'log_posterior'
        else:
            self.neg_cost_key = 'log_likelihood'
        self.params = likelihood.varied_params.names()
        self.limits = {param.name: (param.limits[0], param.limits[1]) for param
                       in likelihood.varied_params}

        self.samples = Samples()
        self.pool = MPIPool()

        self.rng = rng

    def add_global(self):
        """Add finding the global optimum."""
        samples = Samples(**{key: [np.nan, ] for key in self.params})
        self._add_samples(samples)

    def add_sample(self, sample):
        """Add parameter combination to profile.

        Parameters
        ----------
        sample : dict
            Single parameter combination to profile.

        Raises
        ------
        ValueError
            If a parameter is not described in the likelihood.

        """
        for key in sample.keys():
            if key not in self.params:
                raise ValueError(f"Unkown parameter '{key}'.")
        samples = Samples(
            **{key: [value, ] for key, value in sample.items()},
            fixed=[list(sample.keys()), ])
        for key in self.params:
            if key not in sample.keys():
                samples[key] = [np.nan, ]
        self._add_samples(samples)

    def add_grid(self, grid):
        """Add parameter grid to profile.

        Parameters
        ----------
        grid : dict
            Parameter grid to profile, i.e., ``dict(a=[0, 1, 2])`` implies
            that the maximum likelihood is found for :math:`a=0`, :math:`a=1`,
            and :math:`a=2`. If multiple parameters are specified, all
            combinations are profiled.

        """
        data = dict()
        n = 0
        for key, values in grid.items():
            for other_key in data.keys():
                data[other_key] = np.repeat(data[other_key], len(values))
            data[key] = np.tile(values, max(n, 1))
            n = len(data[key])
        samples = Samples(fixed=[list(grid.keys())] * n, **data)
        for key in self.params:
            if key not in grid.keys():
                samples[key] = np.repeat(np.nan, len(samples))
        self._add_samples(samples)

    def _add_samples(self, samples):
        """Add samples to profile."""
        samples[self.neg_cost_key] = np.repeat(-np.inf, len(samples))
        self.samples.append(samples)
        # self._remove_duplicates()
        self.fixed_params = self.samples._get_fixed()

    def _remove_duplicates(self):
        """Remove duplicate parameter combinations to profile over."""
        points = np.zeros((len(self.samples), len(self.params)))
        for i in range(len(points)):
            for k, key in enumerate(self.params):
                if key in self.samples['profiled'][i].split(','):
                    points[i, k] = self.samples[key][i]
                else:
                    points[i, k] = np.nan
        self.samples = self.samples[
            np.unique(points, axis=0, return_index=True)[1]]

    def _vector_to_params(self, vector, index=0):
        """Convert an array of varied parameters to a (complete) dictionary.

        Parameters
        ----------
        vector : numpy.ndarray
            Array of varied parameters normalized to [0, 1].
        index : int, optional
            Index of the fixed parameters.

        Raises
        ------
        ValueError
            If ``vector`` has the wrong length.

        Returns
        -------
        params : dict
            Dictionary including varied and fixed parameters.

        """
        if len(vector) != len(self.params) - len(self.fixed_params[index]):
            raise ValueError("Incorrect number of parameters.")

        varied_params = [p for p in self.params if p not in
                         self.fixed_params[index].keys()]
        vector = vector.copy()
        for i, key in enumerate(varied_params):
            vector[i] = (
                vector[i] * (self.limits[key][1] - self.limits[key][0]) +
                self.limits[key][0])
        return dict(zip(varied_params, vector)) | self.fixed_params[index]

    def _cost_function(self, vector, index=0):
        """Cost function to optimize.

        Parameters
        ----------
        vector : numpy.ndarray or dict
            Array of varied parameters normalized to [0, 1]. Alternatively,
            can be a dictionary listing all parameters.
        index : int, optional
            Index of the fixed parameters.

        Returns
        -------
        float
            Cost function value.

        """
        if not isinstance(vector, dict):
            params = self._vector_to_params(vector, index=index)
        else:
            params = vector

        if self.neg_cost_key == 'log_likelihood':
            return - self.likelihood(params)
        else:
            return - (self.likelihood(params) +
                      self.likelihood.all_params.prior(**params))

    def _get_start(self, max_init_attempts=100):
        """Generate cold-start samples.

        Parameters
        ----------
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each sample. Default is
            100.

        Raises
        ------
        ValueError
            If a finite cost function value cannot be found for all samples
            after ``max_init_attempts``.

        """
        x0 = [None] * len(self.samples)
        cost = np.repeat(np.inf, len(self.samples))

        for _ in range(max_init_attempts):

            for i in range(len(self.samples)):
                if np.isfinite(cost[i]):
                    pass
                n_free = len(self.params) - len(self.fixed_params[i])
                x0[i] = np.random.uniform(size=n_free)

            args = [self._vector_to_params(x, i) for i, (x, c) in enumerate(
                zip(x0, cost)) if not np.isfinite(c)]
            new_cost = list(map(self._cost_function, args))
            cost[~np.isfinite(cost)] = new_cost

            if np.all(np.isfinite(cost)):
                break

        if not np.all(np.isfinite(cost)):
            raise ValueError('Could not find finite posterior '
                             f'after {max_init_attempts:d} attempts.')

        return x0

    def _run_minimizer(self, index_and_x0):
        index, x0 = index_and_x0
        cost_function = partial(self._cost_function, index=index)
        if len(x0) == 0:
            return cost_function(x0)
        res = minimize(cost_function, x0=x0, bounds=[(0, 1)] * len(x0))
        return res.x, res.fun

    def run(self, max_iter=100, tol=1e-3, warm_start=False,
            max_init_attempts=100):
        """Run the profiler.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations. At each iteration, all samples
            are optimized. Default is 100.
        tol : float, optional
            Optimization stops if maximum improvement accross all samples
            drops below ``tol``. Default is 1e-2.
        warm_start : bool, optional
            If True, starting positions are derived from interpolating
            previous points. This can only be done if the profiler was
            run with ``warm_start=False`` before.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each sample. Default is
            100.

        Raises
        ------
        ValueError
            If trying to run the profiler without having added samples.

        Returns
        -------
        samples : Samples
            Maxima found by the profiler.

        """
        if len(self.samples) == 0:
            raise ValueError("Cannot run profiler without samples.")

        for _ in range(max_iter):
            x0 = self._get_start(max_init_attempts=max_init_attempts)
            result = list(map(self._run_minimizer, enumerate(x0)))
            x = [r[0] for r in result]
            cost = np.array([r[1] for r in result])
            update = cost < -self.samples[self.neg_cost_key]
            d_cost = np.amax(-self.samples[self.neg_cost_key] - cost)
            for i in np.arange(len(self.samples))[update]:
                params = self.samples[i]
                params.update(self._vector_to_params(x[i], i))
                params[self.neg_cost_key] = -cost[i]
                self.samples[i] = params
            if d_cost < tol:
                break

        return self.samples
