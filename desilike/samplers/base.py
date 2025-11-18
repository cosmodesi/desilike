"""
Base classes for posterior samplers.

This module defines common functions and classes that are inherited by
specialized classes implementing specific samplers such as `emcee` or
`dynesty`.
"""

# TODO: Properly implement abstract classes and methods.

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

    def __init__(self, likelihood, rng=None, filepath=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        filepath : str, Path, or None, optional
            Save samples to this folder. Default is ``None``.

        """
        self.likelihood = likelihood
        self.n_dim = len(self.likelihood.varied_params)

        if isinstance(rng, int) or rng is None:
            rng = np.random.default_rng(seed=rng)

        self.rng = rng
        if filepath is not None:
            filepath = Path(filepath)
            if filepath.suffix:
                raise ValueError("The filepath cannot have a suffix.")
            filepath.mkdir(parents=True, exist_ok=True)
        self.filepath = filepath
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
        return np.array([self.likelihood.varied_params[i].prior.ppf(x) for
                         i, x in enumerate(sample)])

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
            Whether to save derived parameters internally. Default is False.
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
            return log_l, [derived[key] for key in self.params[self.n_dim:]]
        else:
            return log_l

    def gather_results(self):

        results = self.mpicomm.gather(self.derived)
        if self.mpicomm.rank == 0:
            self.derived = {
                key: list(np.concatenate([r[key] for r in results]))
                for key in self.params.keys()}
        else:
            self.derived = {key: [] for key in self.params.keys()}

    def augment_samples(self, samples, **kwargs):
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
            If not all elements in the chain could be associated with internal
            calculations.

        Returns
        -------
        desilike.samples.Chain
            Samples with added parameters.

        """
        self.gather_results()
        if self.mpicomm.rank == 0:
            samples = Chain(
                [*samples.T] + [*kwargs.values()],
                params=self.params[:self.n_dim] + list(kwargs.keys()))
            results = Chain([*self.derived.values()], params=self.params)

            # Check if there are derived paramters not explicitly passes.
            # Obtain them from the internal results.
            if set(self.params[self.n_dim:]) - set(kwargs.keys()):
                idx_s, idx_r = results.match(
                    samples, params=self.params[:self.n_dim])
                # TODO: Find out why the first index from match is needed.
                if len(idx_s[0]) != len(samples):
                    raise ValueError("Not all derived results could be found.")
                samples = samples[idx_s[0]]
                samples.update(results[idx_r[0]])
        else:
            samples = None

        return self.mpicomm.bcast(samples, root=0)


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
            samples in parameter space to evaluate.
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
        if self.mpicomm.rank == 0:
            log_post = np.array(self.pool.map(self.compute_posterior, samples))
            self.pool.stop_wait()
        else:
            self.pool.wait()

        results = self.augment_samples(
            samples, logposterior=log_post,
            aweight=np.exp(log_post - logsumexp(log_post)))

        return results


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

        return self.augment_samples(samples, **extras)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    # Convergence criteria.
    criteria = {'gelman_rubin_diag_max', 'gelman_rubin_eigen_max',
                'geweke_max'}

    def __init__(self, likelihood, n_chains=4, burn_in=0.2, rng=None,
                 filepath=None):
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
        filepath : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        Raises
        ------
        ValueError
            If ``burn_in`` is a float and larger than unity.

        """
        super().__init__(likelihood, rng=rng, filepath=filepath)
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

        if self.filepath is not None:
            if all((self.filepath / f'chain_{i + 1}.npy').is_file() for i in
                   range(self.n_chains)):
                self.chains = [Chain.load(
                    self.filepath / f'chain_{i + 1}.npy') for i in
                    range(self.n_chains)]
                self.checks = list(np.load(self.filepath / 'checks.npy',
                                           allow_pickle=False))

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

    def check(self, criteria, quiet=False):
        """Check the status of the sampling, including convergence.

        This function will also output the status of the analysis to the log.

        Parameters
        ----------
        criteria : dict
            Criteria for the chains to be considered converged.
        quiet : bool, optional
            If True, do not log results. Default is False.

        Returns
        -------
        bool
            Whether the chains are considered converged.

        """
        for key in criteria.keys():
            if key not in type(self).criteria:
                raise ValueError(
                    f"Unknown convergence criterion '{key}'. Known criteria "
                    f"are {type(self).criteria}.")

        if isinstance(self.burn_in, float):
            burn_in = round(self.burn_in * len(self.chains[0]))
        else:
            burn_in = self.burn_in
        chains = [Chain([*chain[burn_in:].T], params=self.params[:self.n_dim])
                  for chain in self.chains_without_burn_in]

        if not quiet:
            self.log_info('Diagnostics:')

        gelman_rubin_diag = np.amax(diagnostics.gelman_rubin(
            chains, method='diag'))
        gelman_rubin_eigen = np.amax(diagnostics.gelman_rubin(
            chains, method='eigen'))
        try:
            geweke = np.amax(diagnostics.geweke(chains, first=0.1, last=0.5))
        except ValueError:
            geweke = float('inf')

        converged = True

        for name, key, value in zip(
                ["Gelman-Rubin (diagonal)", "Gelman-Rubin (eigen)", "Geweke"],
                ['gelman_rubin_diag', 'gelman_rubin_eigen', 'geweke'],
                [gelman_rubin_diag, gelman_rubin_eigen, geweke]):
            if not quiet:
                self.log_info(f"{name}: {value:.3g}")
            if f'{key}_min' in criteria:
                threshold = criteria[f'{key}_max']
                passed = value > threshold
                converged = converged and passed
                if not quiet:
                    self.log_info(
                        f"{name}: {value:.3g} {'>' if passed else '<'} "
                        f"{threshold:.3g} ({'' if passed else 'not '}passed)")
            if f'{key}_max' in criteria:
                threshold = criteria[f'{key}_max']
                passed = value < threshold
                converged = converged and passed
                if not quiet:
                    self.log_info(
                        f"{name}: {value:.3g} {'<' if passed else '>'} "
                        f"{threshold:.3g} ({'' if passed else 'not '}passed)")

        return converged

    def run(self, n_init=100, burn_in=0.2, min_iterations=0,
            max_iterations=sys.maxsize, convergence_checks_interval=10,
            convergence_checks_passed=10,
            convergence_criteria=dict(gelman_rubin_diag_max=1.1),
            flatten_chains=True, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        n_init : int, optional
            Maximum number of attempts to initialize each chain. Default is
            100.
        burn_in : float or int, optional
            Fraction of samples to remove from each chain. If an integer,
            number of iterations (steps) to remove. Default is 0.2.
        min_iterations : int, optional
            Minimum number of steps to run. Default is 0.
        max_iterations : int, optional
            Maximum number of steps to run. Default is infinity.
        convergence_checks_interval : int, optional
            After how many steps convergence is checked. Default is 10.
        convergence_checks_passed : int, optional
            Threshold for the number of successive succesful convergence
            checks. If fulfilled, the sampling will stop. Default is 10.
        convergence_criteria : dict, optional
            Criteria to define convergence.
        flatten_chains : bool, optional
            Whether to concatenate individual chains into one chain.
        kwargs : dict, optional
            Keyword arguments passed to the run function of the sampler.

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

            while (len(self.chains[0]) < max_iterations) and not (
                    len(self.chains[0]) >= min_iterations and
                    len(self.checks) >= convergence_checks_passed and
                    all(self.checks[-convergence_checks_passed:])):
                n_steps = min(convergence_checks_interval,
                              max_iterations - len(self.chains[0]))
                self.run_sampler(n_steps, **kwargs)
                self.checks.append(self.check(convergence_criteria))
                if self.filepath is not None:
                    for i, chain in enumerate(self.chains):
                        chain.save(self.filepath / f'chain_{i + 1}.npy')
                    np.save(self.filepath / 'checks.npy', self.checks,
                            allow_pickle=False)

            self.pool.stop_wait()
        else:
            self.pool.wait()

        if isinstance(self.burn_in, float):
            burn_in = round(self.burn_in * len(self.chains[0]))
        else:
            burn_in = self.burn_in

        chains = [self.augment_samples(
            chain[burn_in:], logposterior=log_post[burn_in:]) for
            chain, log_post in zip(self.chains, self.log_post)]

        if flatten_chains:
            return Chain.concatenate(chains)
        else:
            return chains
