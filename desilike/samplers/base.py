"""
Base classes for posterior samplers.

This module defines common functions and classes that are inherited by
specialized classes implementing specific samplers such as `emcee` or
`dynesty`.
"""

# TODO: Properly implement abstract classes and methods.

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from desilike.samples import Chain, diagnostics
from desilike.utils import BaseClass
from .pool import MPIPool


def update_kwargs(user_kwargs, sampler, **desilike_kwargs):
    """
    Update the keyword arguments passed to a sampler.

    desilike homogenizes the interface to several samplers. In some cases, this
    requires overwriting keyword arguments the user tries to pass to the
    sampler explicitly.

    Parameters
    ----------
    user_kwargs : dict
        Keyword arguments received from the user.
    sampler : str
        Name of the sampler. This is used to make warnings informative.
    desilike_kwargs : dict
        Keyword arguments enforced by desilike.

    Returns
    -------
    dict
        Updated keyword arguments.

    """
    kwargs = user_kwargs.copy()
    for key, value in desilike_kwargs.items():
        if key in user_kwargs:
            warnings.warn(
                f"The keyword argument '{key}' passed to {sampler} is "
                "overwritten.")
        kwargs[key] = value
    return kwargs


class BaseSampler(BaseClass):
    """Abstract class defining common functions used by all samplers."""

    def __init__(self, likelihood, rng=None, directory=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this folder. Default is ``None``.

        """
        self.likelihood = likelihood
        self.n_dim = len(self.likelihood.varied_params)

        if isinstance(rng, int) or rng is None:
            rng = np.random.default_rng(seed=rng)

        self.rng = rng
        if directory is not None:
            directory = Path(directory)
            if directory.suffix:
                raise ValueError("The directory cannot have a suffix.")
            directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.mpicomm = likelihood.mpicomm
        self.pool = MPIPool(comm=self.mpicomm)
        for name, f in zip(
                ['prior_transform', 'compute_prior', 'compute_posterior',
                 'compute_likelihood'],
                [self.prior_transform, self.compute_prior,
                 self.compute_posterior, self.compute_likelihood]):
            setattr(self, name, self.pool.save_function(f, name))

        self.params = (self.likelihood.varied_params +
                       self.likelihood.params.select(derived=True))
        self.derived = {key: [] for key in self.params.keys()}

    def prior_transform(self, sample):
        """Transform from the unit cube to parameter space using the prior.

        Parameters
        ----------
        sample : numpy.ndarray of shape (n_dim, )
            Sample for which to perform the prior transform.

        Returns
        -------
        numpy.ndarray of shape (n_dim, )
            Prior transformation of the input sample.

        """
        return np.array([param.prior.ppf(x) for param, x in zip(
            self.likelihood.varied_params, sample)])

    def compute_prior(self, samples):
        """
        Compute the natural logarithm of the prior.

        Parameters
        ----------
        samples : numpy.ndarray of shape (n_samples, n_dim) or (n_dim, )
            Sample(s) for which to compute the prior.

        Returns
        -------
        log_prior : numpy.ndarray of shape (n_samples, ) or float
            Natural logarithm of the prior.

        """
        if not isinstance(samples, dict):
            samples = dict(
                zip(self.likelihood.varied_params.names(), samples.T))
        return self.likelihood.all_params.prior(**samples)

    def compute_posterior(self, sample, save_derived=True,
                          return_derived=False):
        """Compute the natural logarithm of the posterior.

        Parameters
        ----------
        sample : numpy.ndarray of shape (n_dim, ) or dict
            Sample for which to compute the likelihood.
        save_derived : bool, optional
            Whether to save derived parameters internally. Default is True.
        return_derived : bool, optional
            Whether to return the derived parameters.

        Returns
        -------
        log_post : float
            Natural logarithm of the posterior.
        derived : numpy.ndarray, optional
            Derived parameters. Only returned if ``return_derived`` is True.

        """
        if not isinstance(sample, dict):
            sample = dict(zip(self.likelihood.varied_params.names(), sample.T))
        log_post, derived = self.likelihood(sample, return_derived=True)

        if save_derived:
            for i, key in enumerate(self.params.keys()):
                if i < self.n_dim:
                    self.derived[key].append(sample[key])
                else:
                    self.derived[key].append(derived[key])

        if return_derived:
            return log_post, [derived[key] for key in self.params[self.n_dim:]]
        else:
            return log_post

    def compute_likelihood(self, sample, save_derived=False,
                           return_derived=True):
        """Compute the natural logarithm of the likelihood.

        Note that this function also saves all derived parameters internally.

        Parameters
        ----------
        sample : numpy.ndarray of shape (n_dim, )
            sample for which to compute the likelihood.
        save_derived : bool, optional
            Whether to save parameters internally. Default is True.
        return_derived : bool, optional
            Whether to return the derived parameters. Default is True.

        Returns
        -------
        log_l : float
            Natural logarithm of the likelihood.
        derived : numpy.ndarray, optional
            Derived parameters. Only returned if ``return_derived`` is True.

        """
        log_prior = self.compute_prior(sample)
        log_post, derived = self.compute_posterior(
            sample, save_derived=save_derived, return_derived=True)
        log_l = log_post - log_prior

        if return_derived:
            return log_l, derived
        else:
            return log_l

    def gather_derived(self):
        """Gather all derived parameters in the main process."""
        for key in self.params.keys():
            if len(self.derived[key]) > 0:
                self.derived[key] = np.concatenate(
                    self.derived[key], axis=None)

        derived = self.mpicomm.gather(self.derived)
        if self.mpicomm.rank == 0:
            self.derived = {
                key: list(np.concatenate(
                    [c[key] for c in derived], axis=None))
                for key in self.params.keys()}
        else:
            self.derived = {key: [] for key in self.params.keys()}

    def write(self):
        """Write internal calculations to disk."""
        self.gather_derived()
        if self.mpicomm.rank == 0:
            np.savez(self.directory / 'derived.npz',
                     **self.derived)

    def read(self):
        """Read internal calculations from disk."""
        if self.mpicomm.rank == 0:
            self.derived = dict(np.load(self.directory / 'derived.npz'))

    def augment(self, samples, **kwargs):
        """Convert a sample into a desilike chain and add optional parameters.

        Parameters
        ----------
        samples : numpy.ndaray of shape (n_samples, n_dim)
            Samples to which internal results should be matched and added.
        **kwargs : dict, optional
            Additional parameters passed to the results. Each must have length
            n_samples.

        Raises
        ------
        ValueError
            If not all samples could be associated with internal calculations.

        Returns
        -------
        desilike.samples.Chain
            Samples with added parameters.

        """
        self.gather_derived()
        if self.mpicomm.rank == 0:
            samples = Chain(
                [*samples.T] + [*kwargs.values()],
                params=self.params[:self.n_dim] + list(kwargs.keys()))
            keys = self.params.keys()
            params = self.params
            derived = Chain([self.derived[key] for key in keys], params=params)
            for key in set(self.derived.keys()) - set(self.params.keys()):
                derived[key] = self.derived[key]

            # Check if there are derived parameters not explicitly passed.
            # Obtain them from the internal results.
            success = True
            if set(self.derived.keys()) - set(kwargs.keys()) - set(
                    self.params.keys()[:self.n_dim]):
                idx_s, idx_d = derived.match(
                    samples, params=params[:self.n_dim])
                # TODO: Find out why the first index from match is needed.
                if len(idx_s[0]) != len(samples):
                    success = False
                samples = samples[idx_s[0]]
                samples.update(derived[idx_d[0]])
        else:
            samples = None
            success = True

        if not self.mpicomm.bcast(success, root=0):
            raise ValueError("Not all derived results could be found.")

        return self.mpicomm.bcast(samples, root=0)


class StaticSampler(BaseSampler):
    """Class defining common functions used by static samplers."""

    def __init__(self, likelihood, rng=None, directory=None):
        """Initialize the static sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this folder. Default is ``None``.

        """
        super().__init__(likelihood, rng=rng, directory=directory)

        if self.directory is not None:
            try:
                self.read()
            except FileNotFoundError:
                pass

    def get_samples(self, **kwargs):
        """Abstract method to get the samples to be evaluated.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_dim)
            Samples in parameter space to evaluate.
        """
        pass

    def run(self, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to the ``get_samples`` method.

        Returns
        -------
        results : desilike.samples.Chain
            Sampler results.

        """
        samples = self.get_samples(**kwargs)

        if not self.mpicomm.bcast(hasattr(self, 'log_post'), root=0):
            # Do the calculations.
            if self.mpicomm.rank == 0:
                self.log_prior = self.pool.map(self.compute_prior, samples)
                self.log_post = self.pool.map(self.compute_posterior, samples)
                self.pool.stop_wait()
            else:
                self.pool.wait()

            if self.directory is not None:
                self.write()

        results = self.augment(samples)
        if self.mpicomm.rank == 0:
            results[results._logprior] = self.log_prior
            results.logposterior = self.log_post
            results[results._loglikelihood] = np.array(
                self.log_post) - np.array(self.log_prior)
            results.aweight = np.exp(self.log_post - logsumexp(self.log_post))

        return self.mpicomm.bcast(results, root=0)

    def write(self):
        """Write internal calculations to disk."""
        super().write()
        if self.mpicomm.rank == 0:
            np.save(self.directory / 'logprior.npy', self.log_prior)
            np.save(self.directory / 'logposterior.npy', self.log_post)

    def read(self):
        """Read internal calculations from disk."""
        super().read()
        if self.mpicomm.rank == 0:
            self.log_prior = np.load(self.directory / 'logprior.npy')
            self.log_post = np.load(self.directory / 'logposterior.npy')


class PopulationSampler(BaseSampler):
    """Class defining common functions used by population samplers."""

    def run_sampler(self, **kwargs):
        """Abstract method to run the sampler from the main MPI process.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to sampler's run method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Sampler results.
        extras : dict
            Extra parameters such as weights and derived parameters.

        """
        pass

    def run(self, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to the run function of the sampler.

        Returns
        -------
        results : desilike.samples.Chain
            Sampler results.

        """
        if self.mpicomm.rank == 0:
            samples, extras = self.run_sampler(**kwargs)
            self.pool.stop_wait()
        else:
            self.pool.wait()
            samples = None
            extras = {}

        return self.augment(samples, **extras)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    def __init__(self, likelihood, n_chains=4, burn_in=0.2, rng=None,
                 directory=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        burn_in : float or int, optional
            Fraction of samples to remove from each chain before doing
            convergence checks or returning results. If an integer, number of
            iterations (steps) to remove. Default is 0.2.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        ValueError
            If ``burn_in`` is a float and larger than unity.

        """
        super().__init__(likelihood, rng=rng, directory=directory)
        self.n_chains = n_chains
        self.chains = None
        self.log_post = None
        self.checks = None

        if isinstance(burn_in, float):
            if burn_in > 1:
                raise ValueError(
                    f"'burn_in' cannot be a float and bigger than 1. Received "
                    f"{burn_in}.")
        self.burn_in = burn_in

        if self.directory is not None:
            try:
                self.read()
            except FileNotFoundError:
                pass

    def run_sampler(self, n_steps, **kwargs):
        """Abstract method to run the sampler from the main MPI process.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        n_steps : int
            How many additional steps to run.
        kwargs: dict, optional
            Extra keyword arguments passed to sampler's run method.

        """
        pass

    def initialize_chains(self, n_init=100):
        """Initialize the chains.

        Parameters
        ----------
        n_init : int, optional
            Maximum number of attempts for each chain. If None, there is
            no limit. Default is 100.

        Raises
        ------
        ValueError
            If no finite posterior has been found after ``n_init`` attempts.

        """
        chains = np.zeros((self.n_chains, self.n_dim))
        log_post = np.repeat(-np.inf, self.n_chains)
        n_try = 0

        while n_try < n_init and not np.all(np.isfinite(log_post)):
            use = ~np.isfinite(log_post)
            for i, param in enumerate(self.likelihood.varied_params):
                if param.ref.is_proper():
                    chains[use, i] = param.ref.sample(
                        size=np.sum(use), random_state=self.rng)
                else:
                    chains[use, i] = np.full(np.sum(use), param.value)

            log_post[use] = self.pool.map(self.compute_posterior, chains[use])
            n_try += 1

        if not np.all(np.isfinite(log_post)):
            raise ValueError('Could not find finite posterior '
                             f'after {n_init:d} attempts.')

        self.chains = chains[:, np.newaxis, :]
        self.log_post = log_post[:, np.newaxis]

    @property
    def chains_without_burn_in(self):
        """Return the chains without burn in."""
        if isinstance(self.burn_in, float):
            burn_in = round(self.burn_in * len(self.chains[0]))
        else:
            burn_in = self.burn_in
        return [chain[burn_in:] for chain in self.chains]

    def check(self, gelman_rubin=1.1, geweke=None, ess=None, quiet=False):
        """Check the status of the sampling.

        This function will also output the status of the analysis to the log.

        Parameters
        ----------
        gelman_rubin : float or None
            If given, the maximum value of the Gelman-Rubin statistic. Default
            is 1.1.
        geweke : float or None
            If given, the maximum value of the Geweke statistic. Default is
            None.
        ess : float or None
            If given, the minimum effective sample size per chain. The
            effective sample size is the number of chain elements divided
            by the autocorrelation time. Default is None.
        quiet : bool, optional
            If True, do not log results. Default is False.

        Returns
        -------
        bool
            Whether the chains passed convergence checks.

        """

        if isinstance(self.burn_in, float):
            burn_in = round(self.burn_in * len(self.chains[0]))
        else:
            burn_in = self.burn_in
        chains = [Chain([*chain[burn_in:].T], params=self.params[:self.n_dim])
                  for chain in self.chains_without_burn_in]

        if not quiet:
            self.log_info('Diagnostics:')

        gelman_rubin_value = np.amax(diagnostics.gelman_rubin(chains))
        try:
            geweke_value = np.amax(
                diagnostics.geweke(chains, first=0.1, last=0.5))
        except ValueError:
            geweke_value = float('inf')

        iact = diagnostics.integrated_autocorrelation_time(chains)
        ess_value = len(chains[0]) / iact.max()

        passed_all = True

        for name, threshold, upper, value in zip(
                ["Gelman-Rubin", "Geweke", "Effective Sample Size"],
                [gelman_rubin, geweke, ess], [True, True, False],
                [gelman_rubin_value, geweke_value, ess_value]):
            if not quiet:
                self.log_info(f"{name}: {value:.3g}")
            if threshold is not None:
                passed = value < threshold if upper else value >= threshold
                passed_all = passed_all and passed
                if not quiet:
                    self.log_info(
                        f"{value:.3g} {'<' if value < threshold else '>='} "
                        f"{threshold:.3g} ({'' if passed else 'not '}passed)")

        return passed_all

    def is_converged(self, min_iterations=0, max_iterations=sys.maxsize,
                     checks_passed=10):
        """Check whether sampling should stop.

        Parameters
        ----------
        min_iterations : int, optional
            Minimum number of steps to run. Default is 0.
        max_iterations : int, optional
            Maximum number of steps to run. Default is infinity.
        checks_passed : int, optional
            Threshold for the number of successive successful convergence
            checks. If fulfilled (and the minimum number of iterations is
            reached), the sampling will stop. Default is 10.

        Returns
        -------
        bool
            If True, sampling should stop.

        """
        if self.mpicomm.rank == 0:
            converged = (len(self.chains[0]) >= max_iterations or
                         (len(self.chains[0]) >= min_iterations and
                          len(self.checks) >= checks_passed and
                          all(self.checks[-checks_passed:])))
        else:
            converged = False

        return self.mpicomm.bcast(converged, root=0)

    def run(self, burn_in=0.2, min_iterations=0, max_iterations=sys.maxsize,
            check_every=10, checks_passed=10, gelman_rubin=1.1, geweke=None,
            ess=None, flatten_chains=True, save_every=10, n_init=100):
        """Run the sampler.

        Parameters
        ----------
        burn_in: float or int, optional
            Fraction of samples to remove from each chain. If an integer,
            number of iterations(steps) to remove. Default is 0.2.
        min_iterations: int, optional
            Minimum number of steps to run. Default is 0.
        max_iterations: int, optional
            Maximum number of steps to run. Default is infinity.
        check_every: int, optional
            After how many steps convergence is checked. Default is 10.
        checks_passed: int, optional
            Threshold for the number of successive successful convergence
            checks. If fulfilled ( and the minimum number of iterations is
            reached), the sampling will stop. Default is 10.
        gelman_rubin: float or None
            Used to asses convergence. If given, the maximum value of the
            Gelman-Rubin statistic. Default is 1.1.
        geweke: float or None
            Used to asses convergence. If given, the maximum value of the
            Geweke statistic. Default is None.
        ess: float or None
            Used to asses convergence.  If given, the minimum effective sample
            size per chain. The effective sample size is the number of chain
            elements divided by the autocorrelation time. Default is None.
        flatten_chains: bool, optional
            Whether to concatenate individual chains into one chain.
        save_every: int, optional
            After how many steps results are saved. Default is 10.
        n_init: int, optional
            Maximum number of attempts to initialize each chain. Default is
            100.

        Returns
        -------
        desilike.samples.Chain or list of desilike.samples.Chain
            Sampler results.

        """
        if self.mpicomm.rank == 0:
            # Initialize the chains, if necessary.
            if self.chains is None:
                self.initialize_chains(n_init=n_init)
                self.checks = []
            self.pool.stop_wait()
        else:
            self.pool.wait()

        if self.directory is None:
            save_every = check_every  # Don't stop to save.

        # Run the chain until convergence.
        while not self.is_converged(
                min_iterations=min_iterations, max_iterations=max_iterations,
                checks_passed=checks_passed):

            # Advance the sampler and do convergence checks.
            if self.mpicomm.rank == 0:
                n_steps_tot = len(self.chains[0])
                n_steps = min(check_every - (n_steps_tot % check_every),
                              save_every - (n_steps_tot % save_every),
                              max_iterations - n_steps_tot)
                self.run_sampler(n_steps)
                n_steps_tot += n_steps
                if n_steps_tot % check_every == 0:
                    self.checks.append(self.check(
                        gelman_rubin=gelman_rubin, geweke=geweke, ess=ess))
                self.pool.stop_wait()
            else:
                self.pool.wait()

            # Write results.
            if self.directory is not None and n_steps_tot % save_every == 0:
                self.write()

        if self.mpicomm.rank == 0:
            chains = self.chains_without_burn_in
        else:
            chains = [None] * self.n_chains

        chains = [self.augment(chain) for chain in chains]

        if flatten_chains:
            return Chain.concatenate(chains)
        else:
            return chains

    def write(self):
        """Write all results to disk."""
        super().write()
        if self.mpicomm.rank == 0:
            for i, chain in enumerate(self.chains):
                np.save(self.directory / f'chain_{i + 1}.npy', chain)
            np.save(self.directory / 'logposterior.npy', self.log_post)
            np.save(self.directory / 'checks.npy', self.checks)
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def read(self):
        """Read internal calculations from disk."""
        super().read()
        if self.mpicomm.rank == 0:
            self.chains = [np.load(self.directory / f'chain_{i + 1}.npy') for
                           i in range(self.n_chains)]
            self.log_post = np.load(self.directory / 'logposterior.npy')
            self.checks = list(np.load(self.directory / 'checks.npy'))
            with open(self.directory / 'rng.json', 'r') as fstream:
                self.rng.bit_generator.state = json.load(fstream)
