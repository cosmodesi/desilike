"""
Base classes for posterior samplers.

This module defines common functions and classes that are inherited by
specialized classes implementing specific samplers such as ``emcee`` or
``dynesty``.
"""

import json
import sys
import warnings
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path

import numpy as np

from desilike import Samples
from desilike.pool import MPIPool
from desilike.statistics import diagnostics
from desilike.utils import BaseClass


def update_parameters(user_kwargs, sampler, **desilike_kwargs):
    """
    Update the parameter passed to a sampler.

    desilike homogenizes the interface to several samplers. In some cases, this
    requires overwriting parameters the user tries to pass to the sampler
    explicitly.

    Parameters
    ----------
    user_kwargs : dict
        Keyword arguments received from the user.
    sampler : str
        Name of the sampler. This is used to make warnings informative.
    **desilike_kwargs : dict, optional
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


class BaseSamplerMeta(type(BaseClass), ABCMeta):
    """Metaclass combining BaseClass metaclass and ABCMeta."""

    pass


class BaseSampler(BaseClass, ABC, metaclass=BaseSamplerMeta):
    """Abstract class defining common functions used by all samplers."""

    def __init__(self, likelihood, rng=None, directory=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this folder. Default is ``None``.

        """
        self.likelihood = likelihood
        self.varied_params = likelihood.varied_params.names()
        self.n_dim = len(self.varied_params)

        params = self.likelihood.all_params.select(derived=True)
        params = [param for param in params if param.name not in
                  ['loglikelihood', 'logprior']]
        self.derived_params = [param.name for param in params]
        self.derived_shapes = [param.shape for param in params]
        self.n_derived = int(sum(
            np.prod(shape) for shape in self.derived_shapes))

        self.pool = MPIPool()
        for name, f in zip(
                ['prior_transform', 'compute_prior', 'compute_posterior',
                 'compute_likelihood'],
                [self.prior_transform, self.compute_prior,
                 self.compute_posterior, self.compute_likelihood]):
            setattr(self, name, self.pool.save_function(f, name))

        if directory is not None:
            directory = Path(directory)
            if directory.suffix:
                raise ValueError("The directory cannot have a suffix.")
            if self.pool.main:
                directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

        if self.directory is not None:
            try:
                self.read()
            except FileNotFoundError:
                pass

        if hasattr(self, 'rng') and rng is None:
            pass
        else:
            # Overwrite the RNG that may be read.
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            self.rng = rng

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

    def compute_prior(self, sample):
        """
        Compute the natural logarithm of the prior.

        Parameters
        ----------
        sample : numpy.ndarray of shape (n_dim, ) or dict
            Sample for which to perform the prior transform.

        Returns
        -------
        log_prior : float
            Natural logarithm of the prior.

        """
        if not isinstance(sample, dict):
            sample = dict(zip(self.varied_params, sample))
        return self.likelihood.all_params.prior(**sample)

    def compute_posterior(self, sample):
        """Compute the natural logarithm of the posterior.

        Parameters
        ----------
        sample : numpy.ndarray of shape (n_dim, ) or dict
            Sample for which to compute the likelihood.

        Returns
        -------
        log_post : float
            Natural logarithm of the posterior.
        derived : numpy.ndarray
            Derived parameters.

        """
        if not isinstance(sample, dict):
            sample = dict(zip(self.varied_params, sample))
        log_post, derived = self.likelihood(sample, return_derived=True)
        derived = np.concatenate([
            np.asarray(derived[key]).flatten() for key in self.derived_params])

        return float(log_post), derived

    def compute_likelihood(self, sample):
        """Compute the natural logarithm of the likelihood.

        Parameters
        ----------
        sample : numpy.ndarray of shape (n_dim, ) or dict
            Sample for which to compute the likelihood.

        Returns
        -------
        log_l : float
            Natural logarithm of the likelihood.
        derived : numpy.ndarray
            Derived parameters.

        """
        log_prior = self.compute_prior(sample)
        log_post, derived = self.compute_posterior(sample)

        return log_post - log_prior, derived

    def array_to_samples(self, samples, derived, **kwargs):
        """Convert NumPy arrays to desilike samples.

        Parameters
        ----------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray of shape (n_samples, n_derived)
            Samples of derived parameters.
        **kwargs : dict, optional
            Extra parameters such as weights.

        Returns
        -------
        samples : desilike.Samples
            Samples with all derived parameters, weights, etc.

        """
        samples = dict(zip(self.varied_params, samples.T))

        derived = np.split(derived, np.cumsum([
            int(np.prod(shape)) for shape in self.derived_shapes])[:-1],
            axis=1)
        derived = [derived[i].reshape((-1, ) + shape) for i, shape in
                   enumerate(self.derived_shapes)]
        derived = dict(zip(self.derived_params, derived))

        samples = Samples(**(samples | derived))
        for key, value in kwargs.items():
            samples[key] = value

        return samples

    def write(self):
        """Write all results to disk."""
        if self.pool.main:
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def read(self):
        """Read internal calculations from disk."""
        if self.pool.main:
            with open(self.directory / 'rng.json', 'r') as fstream:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = json.load(fstream)


class StaticSampler(BaseSampler):
    """Class defining common functions used by static samplers."""

    @abstractmethod
    def get_samples(self, **kwargs):
        """Abstract method to get the samples to be evaluated.

        Parameters
        ----------
        **kwargs: dict, optional
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
        **kwargs : dict, optional
            Keyword arguments passed to the ``get_samples`` method.

        Returns
        -------
        samples : desilike.Samples
            Posterior samples.

        """
        if not self.pool.bcast(hasattr(self, 'results')):
            # Do the calculations.
            if self.pool.main:
                samples = self.get_samples(**kwargs)
                log_prior = np.array(self.pool.map(
                    self.compute_prior, samples))
                results = self.pool.map(
                    self.compute_posterior, samples)
                log_posterior = np.array([r[0] for r in results])
                derived = np.array([r[1] for r in results])

                self.results = self.array_to_samples(
                    samples, derived, log_posterior=log_posterior,
                    log_weight=log_posterior, log_prior=log_prior)

                self.pool.stop_wait()
            else:
                self.pool.wait()

        if self.directory is not None:
            self.write()

        return self.pool.bcast(self.results if self.pool.main else None)

    def write(self):
        """Write internal calculations to disk."""
        if self.pool.main:
            self.results.save(self.directory / 'results.npz')

    def read(self):
        """Read internal calculations from disk."""
        if self.pool.main:
            self.results = Samples.load(self.directory / 'results.npz')


class PopulationSampler(BaseSampler):
    """Class defining common functions used by population samplers."""

    @abstractmethod
    def run_sampler(self, **kwargs):
        """Abstract method to run the sampler from the main MPI process.

        Parameters
        ----------
        **kwargs: dict, optional
            Extra keyword arguments passed to sampler's run method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray
            Samples of derived parameters.
        extras : dict
            Extra parameters such as weights.

        """
        pass

    def run(self, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments passed to the run function of the sampler.

        Returns
        -------
        samples : desilike.Samples
            Posterior samples.

        """
        if self.pool.main:
            samples, derived, extras = self.run_sampler(**kwargs)
            results = self.array_to_samples(samples, derived, **extras)
            self.pool.stop_wait()
        else:
            self.pool.wait()

        return self.pool.bcast(results if self.pool.main else None)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    default_adaptation_steps = 0

    def __init__(self, likelihood, n_chains=4, chains=None, rng=None,
                 directory=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        chains : list of desilike.samples.Chain, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        ValueError
            If ``burn_in`` is a float and larger than unity.

        """
        if chains is None:
            self.n_chains = n_chains
        else:
            self.n_chains = len(chains)

        super().__init__(likelihood, rng=rng, directory=directory)

        if chains is not None:
            self.chains = chains
            self.checks = []

        if not hasattr(self, 'chains'):
            self.chains = []
            self.checks = []

    @abstractmethod
    def run_sampler(self, steps):
        """Abstract method to run the sampler from the main MPI process.

        Parameters
        ----------
        steps : int
            How many additional steps to run.

        """
        pass

    @abstractmethod
    def adapt_sampler(self, steps):
        """Abstract method to adapt the sampler from the main MPI process.

        Parameters
        ----------
        steps : int
            How steps to run for the adaptation.

        """
        pass

    def initialize_chains(self, max_init_attempts=100):
        """Initialize the chains.

        Parameters
        ----------
        max_init_attempts : int or None, optional
            Maximum number of attempts per chain. If ``None``, there is no
            limit. Default is 100.

        Raises
        ------
        ValueError
            If no finite posterior has been found after ``max_init_attempts``
            attempts.

        """
        if max_init_attempts is None:
            max_init_attempts = sys.maxsize

        if self.pool.main:

            for _ in range(max_init_attempts):

                # Draw random samples.
                samples = np.zeros((self.n_chains, self.n_dim))
                for i, param in enumerate(self.likelihood.varied_params):
                    if param.ref.is_proper():
                        samples[:, i] = param.ref.sample(
                            size=self.n_chains, random_state=self.rng)
                    else:
                        samples[:, i] = np.full(self.n_chains, param.value)

                results = self.pool.map(self.compute_posterior, samples)
                log_post = np.array([r[0] for r in results])
                derived = np.array([r[1] for r in results])

                # Accept those with finite posterior.
                for i in np.arange(self.n_chains)[np.isfinite(log_post)]:
                    chain = self.array_to_samples(
                        np.atleast_2d(samples[i]), np.atleast_2d(derived[i]),
                        log_posterior=np.atleast_1d(log_post[i]))
                    self.chains.append(chain)

                if len(self.chains) >= self.n_chains:
                    break

            self.pool.stop_wait()
        else:
            self.pool.wait()

        if self.pool.bcast(len(self.chains) < self.n_chains):
            raise ValueError('Could not find finite posterior '
                             f'after {max_init_attempts:d} attempts.')

    @property
    def state(self):
        """Return the current state of the chains as NumPy arrays.

        Returns
        -------
        samples : numpy.ndarray of shape (n_chains, n_dim)
            Current position of the chains.
        derived : numpy.ndarray of shape (n_chain, n_derived)
            Current derived paramters.
        log_post : numpy.ndarray of shape (n_chains, )
            Current logarithm of the posterior.

        """
        samples = [[chain[key][-1] for key in self.varied_params] for chain in
                   self.chains]
        derived = [np.concatenate([
            np.asarray(chain[key][-1]).flatten() for key in
            self.derived_params]) for chain in self.chains]
        log_post = [chain['log_posterior'][-1] for chain in self.chains]
        return np.array(samples), np.array(derived), np.array(log_post)

    def extend(self, samples, derived, log_post):
        """Extend the sampler chains.

        Parameters
        ----------
        samples : numpy.ndarray of shape (n_chains, n_steps, n_dim)
            Positions in parameter space.
        derived : numpy.ndarray of shape (n_chains, n_steps, ...)
            Blobs returned from the posterior.
        log_post : numpy.ndarray of shape (n_chains, n_steps)
            Logarithm of the posterior.

        """
        for i in range(self.n_chains):
            chain = self.array_to_samples(
                samples[i], derived[i], log_posterior=log_post[i])
            self.chains[i].append(chain)

    def check(self, burn_in=0.2, gelman_rubin=1.1, ess=None):
        """Check the status of the sampling.

        This function will also output the status of the analysis to the log.

        Parameters
        ----------
        burn_in: float or int, optional
            Fraction of samples to remove from each chain. If an integer,
            number of iterations(steps) to remove. Default is 0.2.
        gelman_rubin : float or None
            If given, the maximum value of the Gelman-Rubin statistic. Default
            is 1.1.
        ess : float or None
            If given, the minimum effective sample size per chain. The
            effective sample size is the number of chain elements divided
            by the autocorrelation time. Default is ``None``.

        Returns
        -------
        passed : bool
            Whether the chains passed all convergence checks.

        """
        if isinstance(burn_in, float):
            burn_in = int(burn_in) * len(self.chains[0])
        chains = [chain[burn_in:] for chain in self.chains]

        self.log_info('Diagnostics:')

        gelman_rubin_value = max(diagnostics.gelman_rubin(
            chains, keys=self.varied_params).values())

        tau = max(diagnostics.integrated_autocorrelation_time(
            chains, keys=self.varied_params).values())
        ess_value = len(chains[0]) / tau

        passed_all = True

        for name, threshold, upper, value in zip(
                ["Gelman-Rubin", "Effective Sample Size"], [gelman_rubin, ess],
                [True, False], [gelman_rubin_value, ess_value]):
            self.log_info(f"{name}: {value:.3g}")
            if threshold is not None:
                passed = value < threshold if upper else value >= threshold
                passed_all = passed_all and passed
                self.log_info(
                    f"{value:.3g} {'<' if value < threshold else '>='} "
                    f"{threshold:.3g} ({'' if passed else 'not '}passed)")

        return passed_all

    def is_converged(self, min_steps=0, max_steps=sys.maxsize,
                     checks_passed=10):
        """Check whether sampling should stop.

        Parameters
        ----------
        min_steps : int, optional
            Minimum number of steps to run. Default is 0.
        max_steps : int, optional
            Maximum number of steps to run. Default is infinity.
        checks_passed : int, optional
            Threshold for the number of successive successful convergence
            checks. If fulfilled (and the minimum number of iterations is
            reached), the sampling will stop. Default is 10.

        Returns
        -------
        bool
            If ``True``, sampling should stop.

        """
        if self.pool.main:
            converged = (len(self.chains[0]) >= max_steps or
                         (len(self.chains[0]) >= min_steps and
                          len(self.checks) >= checks_passed and
                          all(self.checks[-checks_passed:])))

        return self.pool.bcast(converged if self.pool.main else None)

    def run(self, burn_in=0.2, min_steps=0, max_steps=None,
            adaptation_steps=None, check_every=300, checks_passed=2,
            gelman_rubin=1.1, ess=None, flatten_chains=True, save_every=300,
            max_init_attempts=100):
        """Run the sampler.

        Parameters
        ----------
        burn_in: float or int, optional
            Fraction of samples to remove from each chain. If an integer,
            number of iterations(steps) to remove. Default is 0.2.
        min_steps: int, optional
            Minimum number of steps to run. Default is 0.
        max_steps: int or None, optional
            Maximum number of steps to run. If ``None``, no limit is applied.
            Default is ``None``.
        adaptation_steps: int, optional
            Number of learning steps for samplers that can learn effective
            hyperparameters online. These samplers include Metropolis-Hastings
            MCMC, HMC, NUTS, and MCLMC. If ``None``, use the sampler-specific
            default value. Default is ``None``.
        check_every: int, optional
            After how many steps convergence is checked. Default is 300.
        checks_passed: int, optional
            Threshold for the number of successive successful convergence
            checks. If fulfilled (and the minimum number of iterations is
            reached), the sampling will stop. Default is 2.
        gelman_rubin: float or None
            Used to asses convergence. If given, the maximum value of the
            Gelman-Rubin statistic. Default is 1.1.
        ess: float or None
            Used to asses convergence.  If given, the minimum effective sample
            size per chain. The effective sample size is the number of chain
            elements divided by the autocorrelation time. Default is ``None``.
        flatten_chains: bool, optional
            Whether to concatenate individual chains into one chain. Default is
            ``True``.
        save_every: int, optional
            After how many steps results are saved. Default is 300.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each chain. Default is
            100.

        Returns
        -------
        samples : desilike.Samples or list of desilike.Samples
            Posterior chains.

        """
        if self.pool.bcast(len(self.chains) == 0):
            self.initialize_chains(max_init_attempts=max_init_attempts)

        if self.directory is None:
            save_every = check_every  # Don't stop to save.

        if adaptation_steps is None:
            adaptation_steps = self.default_adaptation_steps
        self.adaptation_steps = adaptation_steps  # only used for MH MCMC

        if self.pool.main and adaptation_steps > 0:
            self.adapt_sampler(adaptation_steps)

        # Run the chain until convergence.
        steps = self.pool.bcast(len(self.chains[0]) if self.pool.main else 0)

        if max_steps is None:
            max_steps = sys.maxsize

        while not self.is_converged(
                min_steps=min_steps, max_steps=max_steps,
                checks_passed=checks_passed):

            # Advance the sampler and do convergence checks.
            steps_to_take = min(check_every - (steps % check_every),
                                save_every - (steps % save_every),
                                max_steps - steps)
            steps += steps_to_take
            if self.pool.main:
                self.run_sampler(steps_to_take)
                if steps % check_every == 0:
                    self.checks.append(self.check(
                        burn_in=burn_in, gelman_rubin=gelman_rubin, ess=ess))
                self.pool.stop_wait()
            else:
                self.pool.wait()

            # Write results.
            if self.directory is not None and steps % save_every == 0:
                self.write()

        # Write results in case it wasn't written in the last iteration.
        if self.directory is not None and steps % save_every != 0:
            self.write()

        if self.pool.main:
            if isinstance(burn_in, float):
                burn_in = int(burn_in) * len(self.chains[0])
            chains = [chain[burn_in:] for chain in self.chains]

        chains = self.pool.bcast(chains if self.pool.main else None)

        if flatten_chains:
            return Samples.concatenate(chains)
        else:
            return chains

    def write(self):
        """Write all results to disk."""
        super().write()
        if self.pool.main:
            for i, chain in enumerate(self.chains):
                chain.save(self.directory / f'chain_{i + 1}.npz')
            np.save(self.directory / 'checks.npy', self.checks)

    def read(self):
        """Read internal calculations from disk."""
        super().read()
        if self.pool.main:
            self.chains = [Samples.load(self.directory / f'chain_{i + 1}.npz')
                           for i in range(self.n_chains)]
            self.checks = list(np.load(self.directory / 'checks.npy'))
