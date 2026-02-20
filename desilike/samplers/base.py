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

        self.mpicomm = likelihood.mpicomm
        self.pool = MPIPool(comm=self.mpicomm)
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
            if self.mpicomm.rank == 0:
                directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

        if self.directory is not None:
            try:
                self.read()
            except FileNotFoundError:
                pass

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
            sample = dict(
                zip(self.likelihood.varied_params.names(), sample))
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
            sample = dict(zip(self.likelihood.varied_params.names(), sample))
        log_post, derived = self.likelihood(sample, return_derived=True)
        derived = [float(derived[key]) for key in
                   self.likelihood.params.select(derived=True)]

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

    def write(self):
        """Abstract method to write internal calculations to disk.

        This needs to be implemented by the subclass.
        """
        pass

    def read(self):
        """Abstract method to  read internal calculations from disk.

        This needs to be implemented by the subclass.
        """
        pass


class StaticSampler(BaseSampler):
    """Class defining common functions used by static samplers."""

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
        if not self.mpicomm.bcast(hasattr(self, 'results'), root=0):
            # Do the calculations.
            if self.mpicomm.rank == 0:
                samples = self.get_samples(**kwargs)
                log_prior = np.array(self.pool.map(
                    self.compute_prior, samples))
                results = self.pool.map(
                    self.compute_posterior, samples)
                log_post = np.array([r[0] for r in results])
                derived = np.array([r[1] for r in results])

                self.results = Chain(
                    data=np.hstack([samples, derived]).T,
                    params=self.likelihood.params)
                self.results.aweight = np.exp(log_post - logsumexp(log_post))
                self.results.logposterior = log_post
                self.results[self.results._logprior] = log_prior

                self.pool.stop_wait()
            else:
                self.results = None
                self.pool.wait()

        if self.directory is not None:
            self.write()

        return self.mpicomm.bcast(self.results, root=0)

    def write(self):
        """Write internal calculations to disk."""
        if self.mpicomm.rank == 0:
            self.results.save(self.directory / 'results.npz')

    def read(self):
        """Read internal calculations from disk."""
        if self.mpicomm.rank == 0:
            self.results = Chain.load(self.directory / 'results.npz')


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
        kwargs : dict, optional
            Keyword arguments passed to the run function of the sampler.

        Returns
        -------
        results : desilike.samples.Chain
            Sampler results.

        """
        if self.mpicomm.rank == 0:
            samples, derived, extras = self.run_sampler(**kwargs)
            results = Chain(data=np.hstack([samples, derived]).T,
                            params=self.likelihood.params)
            for key in extras.keys():
                setattr(results, key, extras[key])
            self.pool.stop_wait()
        else:
            results = None
            self.pool.wait()

        return self.mpicomm.bcast(results, root=0)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    default_adaptation_steps = 0

    def __init__(self, likelihood, n_chains=4, rng=None, directory=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
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
        self.n_chains = n_chains
        self.chains = []
        self.checks = []
        super().__init__(likelihood, rng=rng, directory=directory)

    def run_sampler(self, steps):
        """Abstract method to run the sampler from the main MPI process.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        steps : int
            How many additional steps to run.

        """
        pass

    def adapt_sampler(self, steps):
        """Abstract method to adapt the sampler from the main MPI process.

        This needs to be implemented by the subclass.

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
        max_init_attempts : int, optional
            Maximum number of attempts per chain. If ``None``, there is no
            limit. Default is 100.

        Raises
        ------
        ValueError
            If no finite posterior has been found after ``max_init_attempts``
            attempts.

        """
        if self.mpicomm.rank == 0:

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
                    chain = Chain(
                        data=np.concatenate((samples[i], derived[i])),
                        params=self.likelihood.params)
                    chain.logposterior = log_post[i]
                    chain.shape = (1, )
                    self.chains.append(chain)

                if len(self.chains) >= self.n_chains:
                    break

            self.pool.stop_wait()
        else:
            self.wait()

        if self.mpicomm.bcast(len(self.chains) < self.n_chains, root=0):
            raise ValueError('Could not find finite posterior '
                             f'after {max_init_attempts:d} attempts.')

    @property
    def state(self):
        """Return the current state of the chains as NumPy arrays."""
        samples = np.zeros((self.n_chains, self.n_dim))
        derived = np.zeros(
            (self.n_chains, len(self.likelihood.params.select(derived=True))))
        log_post = np.zeros(self.n_chains)
        for i in range(self.n_chains):
            samples[i] = [self.chains[i][key][-1].value for key in
                          self.likelihood.varied_params]
            derived[i] = [self.chains[i][key][-1].value for key in
                          self.likelihood.params.select(derived=True)]
            log_post[i] = self.chains[i].logposterior[-1]
        return samples, derived, log_post

    def extend(self, samples, derived, log_post):
        for i in range(self.n_chains):
            chain = Chain(data=np.hstack([samples[i], derived[i]]).T,
                          params=self.likelihood.params)
            chain.logposterior = log_post[i]
            self.chains[i] = Chain.concatenate(self.chains[i], chain)

    def check(self, burn_in=0.2, gelman_rubin=1.1, geweke=None, ess=None,
              quiet=False):
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
        geweke : float or None
            If given, the maximum value of the Geweke statistic. Default is
            ``None``.
        ess : float or None
            If given, the minimum effective sample size per chain. The
            effective sample size is the number of chain elements divided
            by the autocorrelation time. Default is ``None``.
        quiet : bool, optional
            If True, do not log results. Default is False.

        Returns
        -------
        bool
            Whether the chains passed convergence checks.

        """
        chains = [chain.remove_burnin(burn_in) for chain in self.chains]

        if not quiet:
            self.log_info('Diagnostics:')

        gelman_rubin_value = np.amax(diagnostics.gelman_rubin(
            chains, method='diag'))
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
            If True, sampling should stop.

        """
        if self.mpicomm.rank == 0:
            converged = (len(self.chains[0]) >= max_steps or
                         (len(self.chains[0]) >= min_steps and
                          len(self.checks) >= checks_passed and
                          all(self.checks[-checks_passed:])))
        else:
            converged = False

        return self.mpicomm.bcast(converged, root=0)

    def run(self, burn_in=0.2, min_steps=0, max_steps=None,
            adaptation_steps=None, check_every=10, checks_passed=10,
            gelman_rubin=1.1, geweke=None, ess=None, flatten_chains=True,
            save_every=10, max_init_attempts=100):
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
            Geweke statistic. Default is ``None``.
        ess: float or None
            Used to asses convergence.  If given, the minimum effective sample
            size per chain. The effective sample size is the number of chain
            elements divided by the autocorrelation time. Default is ``None``.
        flatten_chains: bool, optional
            Whether to concatenate individual chains into one chain. Default is
            True.
        save_every: int, optional
            After how many steps results are saved. Default is 10.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each chain. Default is
            100.

        Returns
        -------
        desilike.samples.Chain or list of desilike.samples.Chain
            Sampler results.

        """
        if self.mpicomm.bcast(len(self.chains) == 0, root=0):
            self.initialize_chains(max_init_attempts=max_init_attempts)

        if self.directory is None:
            save_every = check_every  # Don't stop to save.

        if adaptation_steps is None:
            adaptation_steps = self.default_adaptation_steps
        self.adaptation_steps = adaptation_steps  # only used for MH MCMC

        if self.mpicomm.rank == 0 and adaptation_steps > 0:
            self.adapt_sampler(adaptation_steps)

        # Run the chain until convergence.
        steps = self.mpicomm.bcast(
            len(self.chains[0]) if self.pool.main else 0, root=0)

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
            if self.mpicomm.rank == 0:
                self.run_sampler(steps_to_take)
                if steps % check_every == 0:
                    self.checks.append(self.check(
                        burn_in=burn_in, gelman_rubin=gelman_rubin,
                        geweke=geweke, ess=ess))
                self.pool.stop_wait()
            else:
                self.pool.wait()

            # Write results.
            if self.directory is not None and steps % save_every == 0:
                self.write()

        if self.mpicomm.rank == 0:
            chains = [chain.remove_burnin(burn_in) for chain in self.chains]
        else:
            chains = [None] * self.n_chains

        if flatten_chains:
            return Chain.concatenate(chains)
        else:
            return chains

    def write(self):
        """Write all results to disk."""
        if self.mpicomm.rank == 0:
            for i, chain in enumerate(self.chains):
                chain.save(self.directory / f'chain_{i + 1}.npy')
            np.save(self.directory / 'checks.npy', self.checks)
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def read(self):
        """Read internal calculations from disk."""
        if self.mpicomm.rank == 0:
            self.chains = [Chain.load(self.directory / f'chain_{i + 1}.npy')
                           for i in range(self.n_chains)]
            self.checks = list(np.load(self.directory / 'checks.npy'))
            with open(self.directory / 'rng.json', 'r') as fstream:
                self.rng.bit_generator.state = json.load(fstream)
