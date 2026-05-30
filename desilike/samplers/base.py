"""
Base classes for posterior samplers.

This module defines common functions and classes that are inherited by
specialized classes implementing specific samplers such as `emcee` or
`dynesty`.
"""

import json
import sys
import warnings
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from desilike import ParameterArray
from desilike.samples import Chain, diagnostics
from desilike.utils import BaseClass
from .pool import MPIPool


def _main(func):
    """Execute function only from the main process."""

    def wrapper(self, *args, **kwargs):
        exception = None
        if self.pool.main:
            try:
                result = func(self, *args, **kwargs)
            except Exception as exc:
                exception = exc
            finally:
                try:
                    self.pool.stop_wait()
                except:
                    self.mpicomm.Abort(1)
        else:
            self.pool.wait()

        exception = self.mpicomm.bcast(exception)
        if exception:
            raise exception

        return self.mpicomm.bcast(None if not self.pool.main else result)

    return wrapper


def _update_parameters(user_kwargs, sampler, **desilike_kwargs):
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
            msg = f"Overwriting keyword argument '{key}' passed to {sampler} "
            warnings.warn(msg)
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
        self.varied_params = self.likelihood.varied_params
        self.derived_params = (
            self.likelihood.all_params.select(derived=True) +
            self.likelihood.all_params.select(solved=True))
        self.n_dim = len(self.varied_params)
        self.n_derived = len(self.derived_params)
        # Derived parameters can be multi-dimensional.
        self.n_derived = int(np.sum([np.prod(param.shape) for param in
                                     self.derived_params]))

        self.mpicomm = likelihood.mpicomm
        self.pool = MPIPool(comm=self.mpicomm)
        for name, f in zip(
                ['_prior_transform', '_compute_prior', '_compute_posterior',
                 '_compute_likelihood'],
                [self._prior_transform, self._compute_prior,
                 self._compute_posterior, self._compute_likelihood]):
            setattr(self, name, self.pool.cache_function(f, name))

        if directory is not None:
            directory = Path(directory)
            if directory.suffix:
                raise ValueError("The directory cannot have a suffix.")
            if self.mpicomm.rank == 0:
                directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

        if self.directory is not None:
            try:
                self._read()
            except FileNotFoundError:
                pass

        if hasattr(self, 'rng') and rng is None:
            pass
        else:
            # Overwrite the RNG that may be read.
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            self.rng = rng

        self._jit_likelihood()

    def _jit_likelihood(self):
        """JIT the likelihood with JAX, if possible."""
        rng = np.random.default_rng(seed=42)

        def get_start(size=1):
            toret = {}
            for param in self.varied_params:
                if param.ref.is_proper():
                    value = param.ref.sample(size=size, random_state=rng)
                else:
                    value = np.full(size, param.value)
                toret[param.name] = value
            return toret

        self.likelihood()  # initialize before jit
        try:
            import jax
            likelihood = jax.jit(
                self.likelihood, static_argnames=['return_derived'])
            likelihood(get_start())
            likelihood(get_start(), return_derived=True)
            self._likelihood = likelihood
            if self.mpicomm.rank == 0:
                self.log_info("Successfully jit input likelihood.")
        except:
            if self.mpicomm.rank == 0:
                self.log_info("Could *not* jit input likelihood.")

    def _prior_transform(self, sample):
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
            self.varied_params, sample)])

    def _compute_prior(self, sample):
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
            sample = dict(zip(self.varied_params.names(), sample))
        return self.likelihood.all_params.prior(**sample)

    def _compute_posterior(self, sample):
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
            sample = dict(zip(self.varied_params.names(), sample))
        log_post, derived = self._likelihood(sample, return_derived=True)
        derived = np.concatenate([
            np.asarray(derived[key]).flatten() for key in self.derived_params])

        return float(log_post), derived

    def _compute_likelihood(self, sample):
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
        log_prior = self._compute_prior(sample)
        log_post, derived = self._compute_posterior(sample)

        return log_post - log_prior, derived

    def _array_to_chain(self, samples, derived, **kwargs):
        """Convert NumPy arrays to desilike chains.

        Parameters
        ----------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray of shape (n_samples, n_derived)
            Samples of derived parameters.
        **kwargs
            Extra parameters such as weights.

        """
        params = self.varied_params
        samples = [ParameterArray(samples[:, i], param=param) for i, param in
                   enumerate(params)]
        params = self.derived_params
        derived = np.split(derived, np.cumsum([
            int(np.prod(param.shape)) for param in params])[:-1], axis=1)
        derived = [derived[i].reshape((-1, ) + param.shape) for i, param in
                   enumerate(params)]
        derived = [ParameterArray(derived[i], param=param) for i, param in
                   enumerate(params)]

        chain = Chain(samples + derived)
        for key, value in kwargs.items():
            setattr(chain, key, value)

        return chain

    def _write(self):
        """Write all results to disk."""
        if self.mpicomm.rank == 0:
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def _read(self):
        """Read internal calculations from disk."""
        if self.mpicomm.rank == 0:
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
        **kwargs
            Extra keyword arguments.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_dim)
            Samples in parameter space to evaluate.

        """
        pass

    @_main
    def run(self, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the ``get_samples`` method.

        Returns
        -------
        results : desilike.samples.Chain
            Sampler results.

        """
        if not hasattr(self, 'results'):
            # Do the calculations.
            samples = self.get_samples(**kwargs)
            log_prior = np.array(self.pool.map(self._compute_prior, samples))
            results = self.pool.map(self._compute_posterior, samples)
            log_post = np.array([r[0] for r in results])
            derived = np.array([r[1] for r in results])

            self.results = self._array_to_chain(
                samples, derived, logposterior=log_post,
                aweight=np.exp(log_post - logsumexp(log_post)))
            self.results[self.results._logprior] = log_prior

        if self.directory is not None:
            self._write()

        return self.results

    def _write(self):
        """Write internal calculations to disk."""
        if self.mpicomm.rank == 0:
            self.results.save(self.directory / 'results.npz')

    def _read(self):
        """Read internal calculations from disk."""
        if self.mpicomm.rank == 0:
            self.results = Chain.load(self.directory / 'results.npz')


class PopulationSampler(BaseSampler):
    """Class defining common functions used by population samplers."""

    @abstractmethod
    def _run(self, **kwargs):
        """Run a specific sampler.

        Parameters
        ----------
        **kwargs
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

    @_main
    def run(self, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the run function of the sampler.

        Returns
        -------
        results : desilike.samples.Chain
            Sampler results.

        """
        samples, derived, extras = self._run(**kwargs)
        return self._array_to_chain(samples, derived, **extras)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    default_adaptation_steps = 0

    def __init__(self, likelihood, n_chains=1, chains=None, rng=None,
                 directory=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of **independent** chains. Default is 1.
        chains : list of desilike.samples.Chain or None, optional
            If given on rank 0, continue the chains. In that case, we will
            ignore what was read from disk. Default is ``None``.
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
    def _run(self, n_steps):
        """Run a specific sampler.

        Parameters
        ----------
        n_steps : int
            How many additional steps to run.

        """
        pass

    @abstractmethod
    def adapt_sampler(self, n_steps):
        """Adapt a specific sampler.

        Parameters
        ----------
        n_steps : int
            How steps to run for the adaptation.

        """
        pass

    def _initialize(self, max_init_attempts=100):
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

        for i in range(1, max_init_attempts + 1):

            # Draw random samples.
            samples = np.zeros((self.n_chains, self.n_dim))
            for i, param in enumerate(self.varied_params):
                if param.ref.is_proper():
                    samples[:, i] = param.ref.sample(
                        size=self.n_chains, random_state=self.rng)
                else:
                    samples[:, i] = np.full(self.n_chains, param.value)

            results = self.pool.map(self._compute_posterior, samples)
            log_post = np.array([r[0] for r in results])
            derived = np.array([r[1] for r in results])

            # Accept those with finite posterior.
            for i in np.arange(self.n_chains)[np.isfinite(log_post)]:
                chain = self._array_to_chain(
                    np.atleast_2d(samples[i]), np.atleast_2d(derived[i]),
                    logposterior=np.atleast_1d(log_post[i]))
                self.chains.append(chain)

            if len(self.chains) >= self.n_chains:
                break

            if i == max_init_attempts:
                msg = f"Could not find finite posterior after {i} attempts."
                raise ValueError(msg)

    @property
    def _state(self):
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
        samples = [[chain[key][-1].value for key in
                    self.varied_params] for chain in self.chains]
        derived = [np.concatenate([
            np.asarray(chain[key][-1].value).flatten() for key in
            self.derived_params]) for chain in self.chains]
        log_post = [chain.logposterior[-1] for chain in self.chains]
        return np.array(samples), np.array(derived), np.array(log_post)

    def _extend(self, samples, derived, log_post):
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
            chain = self._array_to_chain(
                samples[i], derived[i], logposterior=log_post[i])
            self.chains[i] = Chain.concatenate(self.chains[i], chain)

    def _check(self, burn_in=0.2, gelman_rubin=1.1, geweke=None, ess=None,
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

        if len(chains) == 1:
            nsplits = 4
        elif len(chains) < 4:
            nsplits = 2
        else:
            nsplits = None

        gelman_rubin_value = np.amax(diagnostics.gelman_rubin(
            chains, nsplits=nsplits, method='diag'))
        try:
            geweke_value = np.amax(
                diagnostics.geweke(chains, first=0.1, last=0.5))
        except ValueError:
            geweke_value = float('inf')

        iact = diagnostics.integrated_autocorrelation_time(
            chains, check_valid='ignore')
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

    @_main
    def run(self, burn_in=0.2, min_steps=0, max_steps=None,
            adaptation_steps=None, check_every=300, checks_passed=2,
            gelman_rubin=1.1, geweke=None, ess=None, concatenate=True,
            save_every=300, max_init_attempts=100):
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
        geweke: float or None
            Used to asses convergence. If given, the maximum value of the
            Geweke statistic. Default is ``None``.
        ess: float or None
            Used to asses convergence.  If given, the minimum effective sample
            size per chain. The effective sample size is the number of chain
            elements divided by the autocorrelation time. Default is ``None``.
        concatenate: bool, optional
            Whether to concatenate individual chains into one chain. Default is
            ``True``.
        save_every: int, optional
            After how many steps results are saved. Default is 300.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each chain. Default is
            100.

        Returns
        -------
        desilike.samples.Chain or list of desilike.samples.Chain
            Sampler results.

        """
        if len(self.chains) == 0:
            self._initialize(max_init_attempts=max_init_attempts)

        if self.directory is None:
            save_every = check_every  # Don't stop to save.

        if adaptation_steps is None:
            adaptation_steps = self.default_adaptation_steps
        self.adaptation_steps = adaptation_steps  # only used for MH MCMC

        if adaptation_steps > 0:
            self.adapt_sampler(adaptation_steps)

        # Run the chain until convergence.
        n_steps = len(self.chains[0])

        if max_steps is None:
            max_steps = sys.maxsize

        while n_steps < max_steps:

            if (n_steps >= min_steps and len(self.checks) >= checks_passed and
                    all(self.checks[-checks_passed:])):
                break

            # Advance the sampler and do convergence checks.
            n_steps_next = min(check_every - (n_steps % check_every),
                               save_every - (n_steps % save_every),
                               max_steps - n_steps)
            n_steps += n_steps_next
            self._run(n_steps_next)
            if n_steps % check_every == 0:
                self.checks.append(self._check(
                    burn_in=burn_in, gelman_rubin=gelman_rubin,
                    geweke=geweke, ess=ess))

            # Write results.
            if self.directory is not None and n_steps % save_every == 0:
                self._write()

        # Write results in case it wasn't written in the last iteration.
        if self.directory is not None and n_steps % save_every != 0:
            self._write()

        chains = [chain.remove_burnin(burn_in) for chain in self.chains]

        if concatenate:
            chains = Chain.concatenate(chains)

        return chains

    def _write(self):
        """Write all results to disk."""
        super()._write()
        if self.mpicomm.rank == 0:
            for i, chain in enumerate(self.chains):
                chain.save(self.directory / f'chain_{i + 1}.npy')
            np.save(self.directory / 'checks.npy', self.checks)

    def _read(self):
        """Read internal calculations from disk."""
        super()._read()
        if self.mpicomm.rank == 0:
            self.chains = [Chain.load(self.directory / f'chain_{i + 1}.npy')
                           for i in range(self.n_chains)]
            self.checks = list(np.load(self.directory / 'checks.npy'))
