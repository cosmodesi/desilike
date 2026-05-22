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
from .pool import MPIPool, FunctionWrapper


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
        self.varied_params = self.likelihood.varied_params
        self.derived_params = self.likelihood.all_params.select(derived=True) + self.likelihood.all_params.select(solved=True)
        self.n_derived = int(np.sum([np.prod(param.shape) for param in self.derived_params]))

        self.mpicomm = likelihood.mpicomm
        self.set_pool(mpicomm=self.mpicomm)
        self.jit_likelihood(self.likelihood)

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
        self.set_rng(rng=rng)

    def set_rng(self, rng):
        """Set random number generator."""
        if hasattr(self, 'rng') and rng is None:
            pass
        else:
            # Overwrite the RNG that may be read.
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            self.rng = rng

    @property
    def n_dim(self):
        return len(self.varied_params)

    def jit_likelihood(self, likelihood):
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

        likelihood()  # initialize before jit
        try:
            import jax
            jitted_likelihood = jax.jit(likelihood, static_argnames=['return_derived'])
            jitted_likelihood(get_start())
            jitted_likelihood(get_start(), return_derived=True)
            #raise ValueError
        except:
            if self.mpicomm.rank == 0:
                self.log_info('Could *not* jit input likelihood.')
            jitted_likelihood = self.likelihood
        else:
            if self.mpicomm.rank == 0:
                self.log_info('Successfully jit input likelihood.')
        self._likelihood = jitted_likelihood

    def set_pool(self, mpicomm):
        """Set MPI pool."""
        self.pool = MPIPool(comm=mpicomm)
        for name in ['prior_transform', 'compute_prior', 'compute_posterior', 'compute_likelihood']:
            f = getattr(self, name)
            if isinstance(f, FunctionWrapper):
                f = f.function
            setattr(self, name, self.pool.save_function(f, name))

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
            self.varied_params, sample)])

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
                zip(self.varied_params.names(), sample))
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
            sample = dict(zip(self.varied_params.names(), sample))
        log_post, derived = self._likelihood(sample, return_derived=True)
        # [()] to keep value (not derivatives)
        derived = np.concatenate([
            np.asarray(derived[key][()]).flatten() for key in
            self.derived_params])

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

    def array_to_chain(self, samples, derived, **kwargs):
        """Convert NumPy arrays to desilike chains.

        Parameters
        ----------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray of shape (n_samples, n_derived)
            Samples of derived parameters.
        **kwargs : dict, optional
            Extra parameters such as weights.

        """
        params = self.varied_params
        samples = [ParameterArray(samples[..., i], param=param) for i, param in
                   enumerate(params)]
        params = self.derived_params
        derived = np.split(derived, np.cumsum([
            int(np.prod(param.shape)) for param in params])[:-1], axis=-1)
        derived = [derived[i].reshape(derived[i].shape[:-1] + param.shape) for i, param in
                   enumerate(params)]
        derived = [ParameterArray(derived[i], param=param) for i, param in
                   enumerate(params)]

        chain = Chain(samples + derived)
        for key, value in kwargs.items():
            setattr(chain, key, value)

        return chain

    def write(self):
        """Write all results to disk."""
        if self.mpicomm.rank == 0:
            with open(self.directory / 'rng.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)

    def read(self):
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
        results : desilike.samples.Chain
            Sampler results, returned on rank 0.

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

                self.results = self.array_to_chain(
                    samples, derived, logposterior=log_post,
                    aweight=np.exp(log_post - logsumexp(log_post)))
                self.results[self.results._logprior] = log_prior

                self.pool.stop_wait()
            else:
                self.results = None
                self.pool.wait()

        if self.directory is not None:
            self.write()

        # No need to broadcast
        return self.results

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
        results : desilike.samples.Chain
            Sampler results, returned on rank 0.

        """
        if self.pool.comm.rank == 0:
            samples, derived, extras = self.run_sampler(**kwargs)
            results = self.array_to_chain(samples, derived, **extras)
            self.pool.stop_wait()
        else:
            results = None
            self.pool.wait()

        return self.mpicomm.bcast(results, root=0)


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
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.Chain, optional
            If given (to be provided at least on rank 0), continue the chains.
            In that case, we will ignore what was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        ValueError
            If ``burn_in`` is a float and larger than unity.

        """
        self.mpicomm = likelihood.mpicomm
        if not hasattr(self, '_chain'):
            self._chain = None
        input_chains = False
        if self.mpicomm.rank == 0:
            input_chains = chains is not None
            if input_chains:
                if not isinstance(chains, (tuple, list)):
                    chains = [chains]
                n_chains = len(chains)
        input_chains, self.n_chains = self.mpicomm.bcast((input_chains, n_chains), root=0)

        super().__init__(likelihood, rng=rng, directory=directory)
        if input_chains:
            # Move chain to its local process
            for ichain, dest in enumerate(self._pool_mains):
                chain = Chain.sendrecv(chains[ichain] if self.mpicomm.rank == 0 else None,
                                             source=0, dest=dest, mpicomm=self.mpicomm)
                if self.mpicomm.rank == dest:
                    self._chain = chain
        self.checks = []

    def set_rng(self, rng):
        """Set random number generator."""
        if hasattr(self, 'rng') and rng is None:
            pass
        else:
            # Overwrite the RNG that may be read.
            if isinstance(rng, int) or rng is None:
                rng = np.random.default_rng(seed=rng)
            # Draw entropy from existing RNG
            ss = np.random.SeedSequence(rng.integers(0, 2**63, size=4))
            self.rng = [np.random.default_rng(s) for s in ss.spawn(self.n_chains)][self._ichain]

    def set_pool(self, mpicomm):
        """Set MPI pool."""
        if self.n_chains > mpicomm.size:
            raise ValueError(f"n_chains={self.n_chains} cannot exceed MPI size={mpicomm.size}")
        color = mpicomm.rank * self.n_chains // mpicomm.size
        mpicomm = mpicomm.Split(color=color, key=mpicomm.rank)
        super().set_pool(mpicomm=mpicomm)
        mains = self.mpicomm.allgather(self.mpicomm.rank if self.pool.main else None)
        mains = [main for main in mains if main is not None]
        self._pool_mains = mains
        self._ichain = color

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

    def initialize_chains(self, max_init_attempts=100, shape: tuple=None):
        """Initialize the chains.

        Parameters
        ----------
        max_init_attempts : int or None, optional
            Maximum number of attempts per chain. If ``None``, there is no
            limit. Default is 100.
        shape : tuple, optional
            Shape of chain.

        Raises
        ------
        ValueError
            If no finite posterior has been found after ``max_init_attempts``
            attempts.

        """
        if max_init_attempts is None:
            max_init_attempts = sys.maxsize

        if shape is None:
            shape = ()
        shape = tuple(shape)
        size = np.empty(shape).size

        if self.pool.main:
            if self._chain is None:
                samples, log_post, derived = [], [], []

                for _ in range(max_init_attempts):

                    # Draw random samples
                    _samples = np.zeros((shape or (1,)) + (self.n_dim,))
                    for i, param in enumerate(self.varied_params):
                        if param.ref.is_proper():
                            _samples[..., i] = param.ref.sample(size=shape, random_state=self.rng)
                        else:
                            _samples[..., i] = np.full(shape, param.value)

                    results = self.pool.map(self.compute_posterior, _samples.reshape(size, self.n_dim))
                    _log_post = np.array([r[0] for r in results])
                    _derived = np.array([r[1] for r in results])

                    # Accept those with finite posterior
                    is_finite = np.isfinite(_log_post)
                    samples += _samples[is_finite].tolist()
                    log_post += _log_post[is_finite].tolist()
                    derived += _derived[is_finite].tolist()

                    if len(samples) >= size:
                        samples, log_post, derived = np.array(samples[:size]), np.array(log_post[:size]), np.array(derived[:size])
                        if shape:
                            samples, log_post, derived = samples[None, :], log_post[None, :], derived[None, :]
                        self._chain = self.array_to_chain(samples, derived, logposterior=log_post)
                        break

            self.pool.stop_wait()
        else:
            self.pool.wait()

        if any(np.array(self.mpicomm.allgather(self._chain is None))[self._pool_mains]):
            raise ValueError('Could not find finite posterior '
                             f'after {max_init_attempts:d} attempts.')

    @property
    def chains(self):
        """Gather and return all chains on rank 0."""
        chains = []
        for source in self._pool_mains:
            chains.append(Chain.sendrecv(self._chain, source=source, dest=0, mpicomm=self.mpicomm))
        return chains if self.mpicomm.rank == 0 else None

    @property
    def state(self):
        """Return the current state of the (local) chain as NumPy arrays; it is a process-local operation.

        Returns
        -------
        samples : numpy.ndarray of shape  (..., n_dim)
            Current position of the chains, with ... a potential additional dimension (e.g. the number of walkers), if any.
        derived : numpy.ndarray of shape (..., n_derived)
            Current derived parameters.
        log_post : numpy.ndarray of shape (...)
            Current logarithm of the posterior.

        """
        samples = np.concatenate([np.asarray(self._chain[param][-1]).reshape(self._chain.shape[1:] + (-1,)) for param in self.varied_params], axis=-1)
        derived = np.concatenate([np.asarray(self._chain[param][-1]).reshape(self._chain.shape[1:] + (-1,)) for param in self.derived_params], axis=-1)
        log_post = self._chain.logposterior[-1]
        return np.array(samples), np.array(derived), np.array(log_post)

    def extend(self, samples, derived, log_post):
        """Extend the sampler chain; it is a process-local operation.

        Parameters
        ----------
        samples : numpy.ndarray of shape (n_steps, ..., n_dim)
            Positions in parameter space.
        derived : numpy.ndarray of shape (n_steps, ..., ...)
            Blobs returned from the posterior.
        log_post : numpy.ndarray of shape (n_steps, ...)
            Logarithm of the posterior.

        """
        chain = self.array_to_chain(samples, derived, logposterior=log_post)
        self._chain = Chain.concatenate(self._chain, chain)

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
        passed_all = True

        # All on rank 0
        chains = self.chains
        if self.mpicomm.rank == 0:
            chains = [chain.remove_burnin(burn_in) for chain in chains]

            if not quiet:
                self.log_info('Diagnostics:')

            # At least 4 splits
            nsplits = 4 // len(chains)
            if nsplits <= 1: nsplits = None
            gelman_rubin_value = np.max(diagnostics.gelman_rubin(
                chains, method='diag', nsplits=nsplits))
            try:
                geweke_value = np.max(
                    diagnostics.geweke(chains, first=0.1, last=0.5))
            except ValueError:
                geweke_value = float('inf')

            iact = diagnostics.integrated_autocorrelation_time(
                chains, check_valid='ignore')
            ess_value = np.mean([len(chain) for chain in chains]) / iact.max()

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

        return self.mpicomm.bcast(passed_all, root=0)

    def is_converged(self, min_steps=0, max_steps=sys.maxsize,
                     checks_passed=10):
        """Check whether sampling should stop for the local chain.

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
        converged = True
        if self.pool.main:
            converged = (len(self._chain) >= max_steps or
                        (len(self._chain) >= min_steps and
                        len(self.checks) >= checks_passed and
                        all(self.checks[-checks_passed:])))
        return all(self.mpicomm.allgather(converged))

    def run(self, burn_in=0.2, min_steps=0, max_steps=None,
            adaptation_steps=None, check_every=300, checks_passed=2,
            gelman_rubin=1.1, geweke=None, ess=None,
            save_every=300, max_init_attempts=100, concatenate: bool=True):
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
        save_every: int, optional
            After how many steps results are saved. Default is 300.
        max_init_attempts: int, optional
            Maximum number of attempts to initialize each chain. Default is
            100.
        concatenate: bool, optional
            Whether to concatenate individual chains into one chain. Default is
            True.

        Returns
        -------
        desilike.samples.Chain or list of desilike.samples.Chain
            Sampler results, returned on rank 0.

        """
        self.initialize_chains(max_init_attempts=max_init_attempts)

        if self.directory is None:
            save_every = check_every  # Don't stop to save.

        if adaptation_steps is None:
            adaptation_steps = self.default_adaptation_steps
        self.adaptation_steps = adaptation_steps  # only used for MH MCMC

        if adaptation_steps > 0:
            self.adapt_sampler(adaptation_steps)

        # Run the chain until convergence.
        steps = min(self.mpicomm.allgather(len(self._chain) if self.pool.main else sys.maxsize))

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
            self.run_sampler(steps_to_take)
            if steps % check_every == 0:
                self.checks.append(self.check(
                    burn_in=burn_in, gelman_rubin=gelman_rubin,
                    geweke=geweke, ess=ess))

            # Write results.
            if self.directory is not None and steps % save_every == 0:
                self.write()
        # Write results in case it wasn't written in the last iteration.
        if self.directory is not None and steps % save_every != 0:
            self.write()

        if self.pool.main:
            self._chain = self._chain.remove_burnin(burn_in)

        chains = self.chains
        if concatenate and self.mpicomm.rank == 0:
            chains = Chain.concatenate(chains)
        return chains

    def write(self):
        """Write all results to disk."""
        if self.pool.main:
            with open(self.directory / f'rng_{self._ichain + 1}.json', 'w') as fstream:
                json.dump(self.rng.bit_generator.state, fstream)
            self._chain.save(self.directory / f'chain_{self._ichain + 1}.npy')
        if self.mpicomm.rank == 0:
            np.save(self.directory / 'checks.npy', self.checks)

    def read(self):
        """Read internal calculations from disk."""
        if self.pool.main:
            with open(self.directory / f'rng_{self._ichain + 1}.json', 'r') as fstream:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = json.load(fstream)
            self._chain = Chain.load(self.directory / f'chain_{self._ichain + 1}.npy')
        self.checks = list(np.load(self.directory / 'checks.npy'))



class EnsembleSampler(MarkovChainSampler):

    """Base class for ensemble samplers, which run ``nwalkers`` in parallel."""

    def __init__(self, likelihood, n_chains=1, chains=None, rng=None,
                 directory=None, nwalkers=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 1.
        chains : list of desilike.samples.Chain, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        nwalkers : int, str, default=None
            Number of walkers, defaults to :attr:`Chain.shape[1]` of input chains, if any.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        ValueError
            If ``burn_in`` is a float and larger than unity.

        """
        super().__init__(likelihood, rng=rng, directory=directory, n_chains=n_chains, chains=chains)
        if nwalkers is None and self._chain is not None:
            nwalkers = self._chain.shape[1]
        nwalkers = self.mpicomm.allgather(nwalkers)
        for nwalkers in nwalkers:
            if nwalkers is not None: break  # set by input chains
        self.nwalkers = int(nwalkers) if nwalkers is not None else None

    def initialize_chains(self, max_init_attempts=100):
        super().initialize_chains(max_init_attempts=max_init_attempts, shape=(self.nwalkers,))