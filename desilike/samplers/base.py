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

from desilike.samples import Chain, Samples, diagnostics
from desilike.utils import BaseClass
from .pool import MPIPool


PRIOR_TRANSFORM = None
COMPUTE_PRIOR = None
COMPUTE_LIKELIHOOD = None


def set_prior_transform(prior_transform):
    """Set the prior transformation."""
    global PRIOR_TRANSFORM
    PRIOR_TRANSFORM = prior_transform


def prior_transform(x):
    """Set the prior transformation."""
    return PRIOR_TRANSFORM(x)


def set_compute_prior(compute_prior):
    """Set the prior function."""
    global COMPUTE_PRIOR
    COMPUTE_PRIOR = compute_prior


def compute_prior(x):
    """Compute the natural logarithm of the prior."""
    return COMPUTE_PRIOR(x)


def set_compute_likelihood(compute_likelihood):
    """Set the likelihood function."""
    global COMPUTE_LIKELIHOOD
    COMPUTE_LIKELIHOOD = compute_likelihood


def compute_likelihood(x):
    """Compute the natural logarithm of the likelihood."""
    return COMPUTE_LIKELIHOOD(x)


def compute_posterior(x):
    """Compute the natural logarithm of the posterior."""
    return compute_prior(x) + compute_likelihood(x)


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
            Save samples to this location. Default is ``None``.

        """
        self.likelihood = likelihood
        self.n_dim = len(self.likelihood.varied_params)

        if isinstance(rng, int) or rng is None:
            rng = np.random.default_rng(seed=rng)

        self.rng = rng
        if filepath is not None:
            filepath = Path(filepath)
        self.filepath = filepath
        self.mpicomm = likelihood.mpicomm
        self.pool = MPIPool(comm=self.mpicomm)
        self.derived = None

        set_prior_transform(self.prior_transform)
        set_compute_prior(self.compute_prior)
        set_compute_likelihood(self.compute_likelihood)

    def path(self, name, suffix=None):
        """Define the filepath for saving results.

        Parameters
        ----------
        name : str
            Name to be added to the base filename.
        suffix : str, optional
            If given, overwrite the default file extension. Default is None.

        Returns
        -------
        Path
            Filepath.
        """
        if self.filepath is None:
            return None
        else:
            if suffix is None:
                suffix = self.filepath.suffix
            else:
                if not suffix.startswith('.'):
                    suffix = '.' + suffix
            return self.filepath.with_name(
                self.filepath.stem.replace('*', name) + suffix)

    def prior_transform(self, point):
        """Transform from the unit cube to parameter space using the prior.

        Parameters
        ----------
        point : numpy.ndarray of shape (n_dim, )
            Point for which to perform the prior transform.

        Returns
        -------
        numpy.ndarray of shape (n_dim, )
            Prior transformation of the input point.

        """
        return np.array([self.likelihood.varied_params[i].prior.ppf(x) for
                         i, x in enumerate(point)])

    def compute_prior(self, points):
        """
        Compute the natural logarithm of the prior.

        Parameters
        ----------
        points : numpy.ndarray of shape (n_points, n_dim) or (n_dim, )
            Point(s) for which to compute the prior.

        Returns
        -------
        log_prior : numpy.ndarray of shape (n_points, ) or float
            Natural logarithm of the prior.

        """
        if not isinstance(points, dict):
            points = dict(zip(self.likelihood.varied_params.names(), points.T))
        return self.likelihood.all_params.prior(**points)

    def compute_likelihood(self, point):
        """Compute the natural logarithm of the likelihood.

        Note that this function also saves all derived parameters internally.

        Parameters
        ----------
        point : numpy.ndarray of shape (n_dim, )
            Point for which to compute the likelihood.

        Returns
        -------
        log_l : float
            Natural logarithm of the likelihood.

        """
        if not isinstance(point, dict):
            point = dict(zip(self.likelihood.varied_params.names(), point.T))
        derived = self.likelihood(point, return_derived=True)[1]
        derived.update(Samples(point))

        if getattr(self, 'save_derived', True):
            if self.derived is None:
                self.derived = derived
            else:
                self.derived = Samples.concatenate([self.derived, derived])

        return derived['loglikelihood'].value

    def add_derived(self, chain):
        """Add the derived parameters to a chain.

        Parameters
        ----------
        desilike.samples.Chain
            Chain to which derived results should be matched and added.

        Raises
        ------
        ValueError
            If not all elements in the chain could be associated with derived
            parameters.

        Returns
        -------
        desilike.samples.Chain
            Chain with added derived parameters.

        """
        derived = self.mpicomm.gather(self.derived)
        if self.mpicomm.rank == 0:
            self.derived = Samples.concatenate(derived)
        else:
            self.derived = None

        idx_c, idx_d = self.derived.match(
            chain, params=self.likelihood.varied_params)
        # TODO: Find out why the first index from match is needed.
        if len(idx_c[0]) != len(chain):
            raise ValueError("Not all derived results could be found.")
        chain = chain[idx_c[0]]
        chain.update(self.derived[idx_d[0]])
        return chain


class StaticSampler(BaseSampler):
    """Class defining common functions used by static samplers."""

    def get_points(self, **kwargs):
        """Abstract method to get the points to be evaluated.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments.

        Returns
        -------
        numpy.ndarray of shape (n_points, n_dim)
            Points in parameter space to evaluate.
        """
        pass

    def run(self, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to the ``get_points`` method.

        Returns
        -------
        desilike.samples.Chain
            Sampler results.

        """
        points = self.get_points(**kwargs)
        if self.mpicomm.rank == 0:
            log_l = np.array(self.pool.map(compute_likelihood, points))
            log_prior = np.array(self.pool.map(compute_prior, points))
            chain = [points[..., i] for i in range(self.n_dim)]
            chain.append(log_l)
            chain.append(log_prior)
            chain.append(log_l + log_prior)
            chain = Chain(
                chain, params=self.likelihood.varied_params +
                ['loglikelihood', 'logprior', 'logposterior'])
            self.pool.stop_wait()
        else:
            self.pool.wait()
            chain = None

        return self.mpicomm.bcast(chain, root=0)


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
        desilike.samples.Chain
            Sampler results.

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
        desilike.samples.Chain
            Sampler results.

        """
        if self.mpicomm.rank == 0:
            chain = self.run_sampler(**kwargs)
            self.pool.stop_wait()
        else:
            self.pool.wait()
            chain = None

        return self.mpicomm.bcast(chain, root=0)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    # Convergence criteria.
    criteria = {'gelman_rubin_diag_max', 'gelman_rubin_eigen_max',
                'geweke_max'}

    def __init__(self, likelihood, n_chains=10, rng=None, filepath=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 10.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        filepath : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        """
        super().__init__(likelihood, rng=rng, filepath=filepath)
        self.n_chains = n_chains
        self.chains = None
        self.checks = None

        if self.filepath is not None:
            if all(self.path(f'chain_{i + 1}').is_file() for i in
                   range(self.n_chains)):
                self.chains = [Chain.load(
                    self.path(f'chain_{i + 1}')) for i in
                    range(self.n_chains)]
                self.checks = list(np.load(self.path('checks', 'npy'),
                                           allow_pickle=False))

    def run_sampler(self, n_steps, **kwargs):
        """Abstract method to run the sampler from the main MPI process.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        n_steps : int
            How many steps to run.
        kwargs: dict, optional
            Extra keyword arguments passed to sampler's run method.

        """
        pass

    def reset_sampler(self):
        """Abstract method to reset the sampler."""
        pass

    def initialize_chains(self, attempts=100):
        """Initialize the chains.

        Parameters
        ----------
        attempts : int, optional
            Maximum number of attempts for each chain. If None, there is
            no limit. Default is None.

        Raises
        ------
        ValueError
            If no finite posterior has been found after ``attempts`` attempts.

        """
        points = np.zeros((self.n_chains, self.n_dim))
        log_post = np.repeat(-np.inf, self.n_chains)
        n_try = 0

        while n_try < attempts and not np.all(np.isfinite(log_post)):
            use = ~np.isfinite(log_post)
            for i, param in enumerate(self.likelihood.varied_params):
                if param.ref.is_proper():
                    points[use, i] = param.ref.sample(
                        size=np.sum(use), random_state=self.rng)
                else:
                    points[use, i] = np.full(np.sum(use), param.value)

            log_post[use] = self.pool.map(compute_posterior, points[use])
            n_try += 1

        if not np.all(np.isfinite(log_post)):
            raise ValueError('Could not find finite posterior '
                             f'after {attempts:d} attempts.')

        self.chains = [Chain(
            np.append(p, l)[:, np.newaxis],
            params=self.likelihood.varied_params + ['logposterior']) for p, l
            in zip(points, log_post)]

        if self.filepath is not None:
            for i, chain in enumerate(self.chains):
                chain.save(self.path(f'chain_{i + 1}'))

    def check(self, criteria, burn_in=0.2, quiet=False):
        """Check the status of the sampling, including convergence.

        This function will also output the status of the analysis to the log.

        Parameters
        ----------
        criteria : dict
            Criteria for the chains to be considered converged.
        burn_in : float or int, optional
            Fraction of samples to remove from each chain for convergence
            tests. If an integer, number of iterations (steps) to remove.
            Default is 0.2.
        quiet : bool, optional
            If True, do not log results. Default is False.

        Raises
        ------
        ValueError
            If a convergence criterion is not recognized or ``burn_in`` is a
            float and bigger than unity.

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

        if isinstance(burn_in, float):
            burn_in = round(burn_in * len(self.chains[0]))
        chains = [chain[burn_in:] for chain in self.chains]

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

    def run(self, initialization_attempts=100, burn_in=0.2, min_iterations=0,
            max_iterations=sys.maxsize, convergence_checks_interval=10,
            convergence_checks_passed=10,
            convergence_criteria=dict(gelman_rubin_diag_max=1.1),
            flatten_chains=True, **kwargs):
        """Run the sampler.

        Parameters
        ----------
        initialization_attempts : int, optional
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

        Raises
        ------
        ValueError
            If ``burn_in`` is a float and larger than unity.

        Returns
        -------
        desilike.samples.Chain or list of desilike.samples.Chain
            Sampler results.

        """
        if isinstance(burn_in, float):
            if burn_in > 1:
                raise ValueError(
                    f"'burn_in' cannot be a float and bigger than 1. Received "
                    f"{burn_in}.")

        if self.mpicomm.rank == 0:

            # Initialize the chains, if necessary.
            if self.chains is None:
                self.initialize_chains(attempts=initialization_attempts)
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
                        chain.save(self.path(f'chain_{i + 1}'))
                    np.save(self.path('checks', 'npy'), self.checks,
                            allow_pickle=False)

            self.pool.stop_wait()
        else:
            self.pool.wait()

        self.reset_sampler()

        chains = self.mpicomm.bcast(self.chains, root=0)

        if isinstance(burn_in, float):
            burn_in = round(burn_in * len(chains[0]))
        chains = [chain[burn_in:] for chain in chains]

        if flatten_chains:
            return Chain.concatenate(chains)
        else:
            return chains
