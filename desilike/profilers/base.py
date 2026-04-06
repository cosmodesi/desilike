"""Base class for profilers."""
# TODO: implement other optimizers
# TODO: add plotting functions
# TODO: expand functionality such as warm starts

import json
from functools import partial
from pathlib import Path

import numpy as np

from desilike import Samples
from desilike.pool import MPIPool
from desilike.utils import BaseClass
from .optimizers import scipy_dual_annealing


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

        self.pool = MPIPool()
        for name in ['_cost_function', '_run_optimizer']:
            setattr(self, name, self.pool.save_function(
                getattr(self, name), name))

        if directory is not None:
            directory = Path(directory)
            if directory.suffix:
                raise ValueError("The directory cannot have a suffix.")
            if self.pool.main:
                directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

        if self.directory is not None:
            try:
                self.load()
            except FileNotFoundError:
                pass

        if not hasattr(self, 'samples'):
            self.samples = Samples()

        if not hasattr(self, 'rng'):
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            self.rng = rng

    def save(self):
        """Save all results to disk."""
        if self.pool.main:
            self.samples.save(self.directory / 'samples.npz')
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def load(self):
        """Load internal calculations from disk."""
        if self.pool.main:
            self.samples = Samples.load(self.directory / 'samples.npz')
            with open(self.directory / 'rng.json', 'r') as fstream:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = json.load(fstream)

    def add_global(self):
        """Add finding the global optimum."""
        samples = Samples(**{key: [np.nan, ] for key in self.params},
                          fixed=[[], ])
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
        self._remove_duplicates()
        self.fixed_params = self.samples._get_fixed()

    def _remove_duplicates(self):
        """Remove duplicate parameter combinations to profile over."""
        samples = np.zeros((len(self.samples), len(self.params)))
        fixed_params = self.samples._get_fixed()
        for i in range(len(self.samples)):
            for k, key in enumerate(self.params):
                if key in fixed_params[i]:
                    samples[i, k] = self.samples[key][i]
                else:
                    samples[i, k] = np.inf  # np.nan doesn't work with unique
        self.samples = self.samples[
            np.unique(samples, axis=0, return_index=True)[1]]

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
            return - (self.likelihood(params) -
                      self.likelihood.all_params.prior(**params))
        else:
            return - self.likelihood(params)

    def _get_start(self, n, max_init_attempts=100):
        """Generate cold-start samples.

        This should only be called by the main process while the others are
        waiting.

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

        Returns
        -------
        index : numpy.ndarray
            Indices corresponding to the sample.
        x_0 : list of numpy.ndarray
            Starting positions.

        """
        index = np.repeat(np.arange(len(self.samples)), n)
        x_0 = [None] * len(index)
        cost = np.repeat(np.inf, len(x_0))

        for _ in range(max_init_attempts):

            for i in range(len(x_0)):
                if np.isfinite(cost[i]):
                    pass
                n_free = len(self.params) - len(self.fixed_params[index[i]])
                x_0[i] = self.rng.uniform(size=n_free)

            args = [self._vector_to_params(x, i) for i, x, c in zip(
                index, x_0, cost) if not np.isfinite(c)]
            new_cost = self.pool.map(self._cost_function, args)
            cost[~np.isfinite(cost)] = new_cost

            if np.all(np.isfinite(cost)):
                break

        if not np.all(np.isfinite(cost)):
            raise ValueError("Could not find finite likelihood/posterior "
                             f"after {max_init_attempts:d} attempts.")

        return index, x_0

    def _run_optimizer(self, optimizer, args, **kwargs):
        index, x_0, rng = args
        cost_function = partial(self._cost_function, index=index)
        if len(x_0) == 0:
            return x_0, cost_function(x_0), True
        return optimizer(cost_function, x_0, rng, **kwargs)

    def run(self, n_per_iter=10, max_iter=10, tol=1e-3, warm_start=False,
            max_init_attempts=100, optimizer=scipy_dual_annealing,
            optimizer_kwargs=None):
        """Run the profiler.

        Parameters
        ----------
        n_per_iter : int, optional
            Independent optimizations per sample at each iteration. Default is
            10.
        max_iter : int, optional
            Maximum number of iterations. Default is 10.
        tol : float, optional
            Optimization stops if maximum improvement accross all samples
            drops below ``tol`` between optimizations. Default is 1e-2.
        warm_start : bool, optional
            If True, starting positions are derived from interpolating
            previous points. This can only be done if the profiler was
            run with ``warm_start=False`` before.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each sample. Default is
            100.
        optimizer : callable, optional
            Optimizer function from ``desilike.profilers.optimizers``. Default
            is ``desilike.profilers.scipy_dual_annealing``.
        optimizer_kwargs : dict, optional
            Optional keyword arguments passed to the optimizer. Default is
            ``None``.

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

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        run_optimizer = partial(self._run_optimizer, optimizer,
                                **optimizer_kwargs)

        if self.pool.main:

            for _ in range(max_iter):
                index, x_0 = self._get_start(
                    n_per_iter, max_init_attempts=max_init_attempts)
                result = self.pool.map(
                    run_optimizer, zip(index, x_0, self.rng.spawn(len(x_0))))

                impr = np.zeros(len(self.samples))
                for i, (x_min, f_min, success) in zip(index, result):
                    if f_min < -self.samples[self.neg_cost_key][i]:
                        impr[i] = -self.samples[self.neg_cost_key][i] - f_min
                        params = self.samples[i]
                        params.update(self._vector_to_params(x_min, i))
                        params[self.neg_cost_key] = -f_min
                        self.samples[i] = params

                if self.directory is not None:
                    self.save()

                if np.amax(impr) < tol:
                    break

            self.pool.stop_wait()
        else:
            self.pool.wait()

        return self.pool.bcast(self.samples)
