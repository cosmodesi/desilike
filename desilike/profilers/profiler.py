"""Base class for profilers."""
# TODO: should fail if points added are outside limits

import json
from functools import partial
from pathlib import Path

import numpy as np

from desilike import Samples
from desilike.pool import MPIPool, from_main
from desilike.utils import BaseClass

from .optimize import optimize_dual_annealing


class Profiler(BaseClass):
    """Profiler used to compute likelihood and posterior profiles."""

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
        for name in ['_cost_function', '_run_optimize']:
            setattr(self, name, self.pool.cache_function(
                getattr(self, name), name))

        if directory is not None:
            directory = Path(directory)
            if directory.suffix:
                msg = "The directory cannot have a suffix."
                raise ValueError(msg)
            if self.pool.main:
                directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

        if self.directory is not None:
            try:
                self._load()
            except FileNotFoundError:
                pass

        if not hasattr(self, 'samples'):
            self.samples = Samples()

        if not hasattr(self, 'rng'):
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            self.rng = rng

    def _save(self):
        """Save all results to disk."""
        if self.pool.main:
            self.samples.save(self.directory / 'samples.npz')
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def _load(self):
        """Load internal calculations from disk."""
        if self.pool.main:
            self.samples = Samples.load(self.directory / 'samples.npz')
            with open(self.directory / 'rng.json', 'r') as fstream:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = json.load(fstream)

    def _add_samples(self, samples):
        """Add samples to profile."""
        samples[self.neg_cost_key] = -np.inf
        self.samples.append(samples)

        # Remove duplicate parameter combinations. Use a complex number as a
        # placeholder for optimized parameters since np.nan is not treated
        # as equal if present in arrays.
        x = np.column_stack([np.where(
            self.samples.get_flag('optimize', param), 1j,
            self.samples[param]) for param in self.params])
        self.samples = self.samples[np.unique(x, axis=0, return_index=True)[1]]

        # Get a list of dictionaries of fixed parameters.
        self.fixed_params = []
        for i in range(len(self.samples)):
            self.fixed_params.append({})
            for param in self.params:
                if not self.samples.get_flag('optimize', param)[i]:
                    self.fixed_params[i][param] = self.samples[i][param]

    def add_single_sample(self, param_dict):
        """Add a parameter combination to optimize.

        Parameters
        ----------
        sample : dict
            Single parameter combination to profile.

        Raises
        ------
        ValueError
            If a parameter is not described in the likelihood.

        """
        for param in param_dict:
            if param not in self.params:
                msg = f"Unkown parameter '{param}'."
                raise ValueError(msg)

        samples = Samples(**{key: [value, ] for key, value in
                             param_dict.items()})
        for param in self.params:
            if param not in param_dict:
                samples[param] = [np.nan, ]
                samples.set_flag('optimize', param, True)
            else:
                samples.set_flag('optimize', param, False)

        self._add_samples(samples)

    def add_optimize_all(self):
        """Add finding the global optimum."""
        self.add_single_sample({})

    def add_manual_grid(self, param_dict_grid):
        """Manually add parameter grid to optimize.

        Parameters
        ----------
        param_dict_grid : dict
            Parameter grid to profile, i.e., ``dict(a=[0, 1, 2])`` implies
            that the maximum likelihood is found for :math:`a=0`, :math:`a=1`,
            and :math:`a=2`. If multiple parameters are specified, all
            combinations are profiled.

        Raises
        ------
        ValueError
            If no parameter is specificed or a parameter is not described in
            the likelihood.

        """
        if not param_dict_grid:
            msg = "You must specify at least one parameter."
            raise ValueError(msg)
        for param in param_dict_grid:
            if param not in self.params:
                msg = f"Unkown parameter '{param}'."
                raise ValueError(msg)

        # Get all combinations.
        samples = dict(zip(param_dict_grid.keys(),
                           np.meshgrid(*param_dict_grid.values())))
        samples = {key: value.flatten() for key, value in samples.items()}

        samples = Samples(**samples)
        for param in self.params:
            if param not in samples.params:
                samples[param] = np.nan
                samples.set_flag('optimize', param, True)
            else:
                samples.set_flag('optimize', param, False)

        self._add_samples(samples)

    def _normalize(self, theta_i, param, inverse=False):
        a = self.limits[param][0]
        b = self.limits[param][1] - self.limits[param][0]
        if inverse:
            return a + b * theta_i
        else:
            return (theta_i - a) / b

    def _vector_to_params(self, theta_var, index):
        """Convert an array of varied parameters to a (complete) dictionary.

        Parameters
        ----------
        theta_var : numpy.ndarray
            Array of varied parameters normalized to [0, 1].
        index : int
            Index of the fixed parameters.

        Returns
        -------
        param_dict : dict
            Dictionary of varied and fixed parameters. The parameters are not
            normalized.

        Raises
        ------
        ValueError
            If ``theta`` has the wrong length.

        """
        if len(theta_var) != len(self.params) - len(self.fixed_params[index]):
            msg = "Incorrect number of parameters."
            raise ValueError(msg)

        varied_params = [p for p in self.params if p not in
                         self.fixed_params[index]]
        theta_var = [self._normalize(theta_i, key, inverse=True) for
                     theta_i, key in zip(theta_var, varied_params)]
        return dict(zip(varied_params, theta_var)) | self.fixed_params[index]

    def _cost_function(self, theta_var, index=0):
        """Cost function to optimize.

        Parameters
        ----------
        theta_var : numpy.ndarray or dict
            Array of varied parameters normalized to [0, 1]. Alternatively,
            can be a dictionary listing all parameters.
        index : int, optional
            Index of the fixed parameters.

        Returns
        -------
        float
            Cost function value.

        """
        if isinstance(theta_var, dict):
            param_dict = theta_var
        else:
            param_dict = self._vector_to_params(theta_var, index)

        if self.neg_cost_key == 'log_likelihood':
            return - (self.likelihood(param_dict) -
                      self.likelihood.all_params.prior(**param_dict))

        return - self.likelihood(param_dict)

    def _get_start(self, n, warm=False, max_init_attempts=100):
        """Generate starting positions for all samples.

        This should only be called by the main process while the others are
        waiting.

        Parameters
        ----------
        n : int
            Number of starts per sample.
        warm : bool, optional
            If False, starting samples are drawn uniformly from within the
            limits over which the likelihood is defined. If True, they are
            instead drawn uniformly from the range of best-fit values. Default
            is False.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each sample. Default is
            100.

        Returns
        -------
        idx : numpy.ndarray
            Sample indeces.
        theta_var : list of numpy.ndarray
            Starting positions.

        Raises
        ------
        ValueError
            If a finite cost function value cannot be found for all samples
            after ``max_init_attempts``. If ``warm_start=True`` but not all
            samples have been optimized from a cold start.

        """
        idx = np.repeat(np.arange(len(self.samples)), n)
        theta_var = [None] * len(idx)
        cost = np.repeat(np.inf, len(theta_var))

        if not warm:
            limits = {param: (0, 1) for param in self.params}
        else:
            if np.any(self.samples[self.neg_cost_key] == -np.inf):
                msg = "Not all samples initialized. Warm start not available."
                raise ValueError(msg)
            limits = {param: (
                self._normalize(np.amin(self.samples[param]), param),
                self._normalize(np.amax(self.samples[param]), param)) for
                param in self.params}

        for _ in range(max_init_attempts):

            for i in range(len(theta_var)):
                if np.isfinite(cost[i]):
                    pass
                theta_var[i] = []
                for param in self.params:
                    if param not in self.fixed_params[idx[i]]:
                        theta_var[i].append(self.rng.uniform(*limits[param]))

            args = [self._vector_to_params(t, i) for t, i, c in
                    zip(theta_var, idx, cost) if not np.isfinite(c)]
            new_cost = self.pool.map(self._cost_function, args)
            cost[~np.isfinite(cost)] = new_cost

            if np.all(np.isfinite(cost)):
                break

        if not np.all(np.isfinite(cost)):
            msg = ("Could not find finite likelihood/posterior after "
                   f"{max_init_attempts:d} attempts.")
            raise ValueError(msg)

        return idx, theta_var

    def _run_optimize(self, optimize, args, **kwargs):
        theta_var, index, rng = args
        cost_function = partial(self._cost_function, index=index)
        if len(theta_var) == 0:
            return theta_var, cost_function(theta_var), True
        return optimize(cost_function, theta_var, rng, **kwargs)

    @from_main
    def run(self, n_per_iter=10, max_iter=10, tol=1e-3, warm_start=False,
            max_init_attempts=100, optimize=None, optimize_kwargs=None):
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
            drops below ``tol`` between iterations. Default is 1e-2.
        warm_start : bool, optional
            If True, starting positions are limited to the range of
            best-fit parameters thus far. This can only be done if the profiler
            was run before.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each sample. Default is
            100.
        optimize : callable or None, optional
            Optimize function from ``desilike.profilers.optimize``. If
            ``None``, default to
            ``desilike.profilers.optimize.optimize_dual_annealing``. Default
            is ``None``.
        optimize_kwargs : dict or None, optional
            Optional keyword arguments passed to the optimize function.
            Default is ``None``.

        Returns
        -------
        samples : desilike.statistics.samples.Samples
            Maxima found by the profiler.

        Raises
        ------
        ValueError
            If trying to run the profiler without having added samples.

        """
        if len(self.samples) == 0:
            msg = "Cannot run profiler without samples."
            raise ValueError(msg)

        if optimize is None:
            optimize = optimize_dual_annealing

        if optimize_kwargs is None:
            optimize_kwargs = {}

        run_optimize = partial(self._run_optimize, optimize, **optimize_kwargs)

        for _ in range(max_iter):
            idx, theta_var = self._get_start(
                n_per_iter, warm=warm_start,
                max_init_attempts=max_init_attempts)
            result = self.pool.map(
                run_optimize, zip(theta_var, idx, self.rng.spawn(len(idx))))

            impr = np.zeros(len(self.samples))
            for i, (x_min, f_min, success) in zip(idx, result):
                if f_min < -self.samples[self.neg_cost_key][i]:
                    impr[i] = -self.samples[self.neg_cost_key][i] - f_min
                    params = self.samples[i]
                    params.update(self._vector_to_params(x_min, i))
                    params[self.neg_cost_key] = -f_min
                    self.samples[i] = params

            if self.directory is not None:
                self._save()

            if np.amax(impr) < tol:
                break

        return self.samples
