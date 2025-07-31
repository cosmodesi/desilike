"""
Base classes for posterior samplers.

This module defines common functions and classes that are inherited by
specialized classes implementing specific samplers such as `emcee` or
`dynesty`.
"""

import warnings

import numpy as np

from desilike.samples import Samples, diagnostics
from desilike.utils import BaseClass
from .mpi_pool import MPIPool


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
    """Class defining common functions used by all samplers."""

    def __init__(self, likelihood, rng=None, save_fn=None, mpicomm=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        save_fn : str, Path, optional
            Save samples to this location. Default is ``None``.
        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        """
        self.likelihood = likelihood
        for param in likelihood.varied_params:
            if not hasattr(param.prior, 'ppf'):
                raise AttributeError(
                    f"Cannot perform prior transform for parameter '{param}'. "
                    "The prior must be proper and have a 'ppf' argument.")
        self.n_dim = len(self.likelihood.varied_params)

        if isinstance(rng, int):
            rng = np.random.default_rng(seed=rng)

        self.rng = rng
        self.save_fn = save_fn
        self.mpicomm = mpicomm if mpicomm is not None else likelihood.mpicomm
        self.pool = MPIPool(comm=self.mpicomm)
        self.blobs = False

    def prior_transform(self, x):
        """Transform from the unit cube to parameter space using the prior.

        Parameters
        ----------
        x : numpy.ndarray
            Point for which to perform the prior transform.

        Returns
        -------
        numpy.ndarray
            Prior transformation of the input point.

        """
        return np.array([self.likelihood.varied_params[i].prior.ppf(x_i) for
                         i, x_i in enumerate(x)])

    def compute_prior(self, values):
        """
        Compute the natural logarithm of the prior.

        Parameters
        ----------
        values : numpy.ndarray of shape (n_points, n_dim)
            Points for which to compute the prior.

        Returns
        -------
        numpy.ndarray of shape (n_points, )
            Natural logarithm of the prior.

        """
        return self.likelihood.all_params.prior(
            **dict(zip(self.likelihood.varied_params.names(), values.T)))

    def compute_likelihood(self, x):
        """Compute the natural logarithm of the likelihood.

        Parameters
        ----------
        x : numpy.ndarray
            Point for which to compute the likelihood.

        Returns
        -------
        log_l : float
            Natural logarithm of the likelihood.
        blob : dict, optional
            Derived results. Only returned if ``self.blobs`` is set to True.

        """
        result = self.likelihood(
            Samples(x, params=self.likelihood.varied_params).to_dict(),
            return_derived=True)[1]
        log_l = result['loglikelihood'].value

        if not self.blobs:
            return log_l

        blob = dict()
        for param in result:
            if param.param.name != 'loglikelihood':
                blob[param.param.name] = param.value

        return log_l, blob


class PopulationSampler(BaseSampler):
    """Class defining common functions used by population samplers."""

    def __init__(self, likelihood, rng=None, save_fn=None, mpicomm=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        save_fn : str, Path, optional
            Save samples to this location. Default is ``None``.
        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        Raises
        ------
        AttributeError
            If the prior does not support prior transforms.

        """
        super().__init__(likelihood, rng=rng, save_fn=save_fn, mpicomm=mpicomm)
        set_prior_transform(self.prior_transform)
        set_compute_likelihood(self.compute_likelihood)

    def run_sampler(self, **kwargs):
        """Abstract method to run the sampler from the main MPI process.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to sampler's run method.

        Returns
        -------
        Chain
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
        Chain
            Sampler results.

        """
        chain = None
        if self.mpicomm.rank == 0:
            chain = self.run_sampler(**kwargs)
        else:
            self.pool.wait()
        self.pool.stop_wait()

        return self.mpicomm.bcast(chain, root=0)


class MarkovChainSampler(BaseSampler):
    """Class defining common functions used by Markov chain samplers."""

    # Convergence criteria.
    criteria = {'gelman_rubin_diag_max', 'gelman_rubin_eigen_max'}

    def __init__(self, likelihood, n_chains, rng=None, save_fn=None,
                 mpicomm=None):
        """Initialize the sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int
            Number of chains.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        save_fn : str, Path, optional
            Save samples to this location. Default is ``None``.
        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        """
        super().__init__(likelihood, rng=rng, save_fn=save_fn, mpicomm=mpicomm)
        self.n_chains = n_chains
        self.chains = None
        set_compute_prior(self.compute_prior)
        set_compute_likelihood(self.compute_likelihood)

    def run_sampler(self, iterations, start=None, **kwargs):
        """Abstract method to run the sampler from the main MPI process.

        This needs to be implemented by the subclass.

        Parameters
        ----------
        iterations : int
            How many iterations to run.
        kwargs: dict, optional
            Extra keyword arguments passed to sampler's run method.

        """
        pass

    def check(self, criteria, burnin=0.5, quiet=False):
        """Check the status of the sampling, including convergence.

        This function will also output the status of the analysis to the log.

        Parameters
        ----------
        criteria : dict
            Criteria for the chains to be considered converged.
        burnin : float or int, optional
            Fraction of samples to remove from each chain for convergence
            tests. If an integer, number of iterations (steps) to remove.
            Default is 0.
        quiet : bool, optional
            If True, do not log results. Default is False.

        Raises
        ------
        ValueError
            If a convergence criterion is not recognized or ``burnin`` is a
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

        if isinstance(burnin, float):
            if burnin > 1:
                raise ValueError(
                    f"'burnin' cannot be a float and bigger than 1. Received "
                    f"{burnin}.")
            burnin = int(burnin * len(self.chains[0]) + 0.5)

        if not quiet:
            self.log_info('Diagnostics:')

        chains = [chain[burnin:] for chain in self.chains]

        gelman_rubin_diag = np.amax(diagnostics.gelman_rubin(
            chains, method='diag'))
        gelman_rubin_eigen = np.amax(diagnostics.gelman_rubin(
            chains, method='eigen'))

        converged = True

        for name, key, value in zip(
                ["Gelman-Rubin (diagonal)", "Gelman-Rubin (eigen)"],
                ['gelman_rubin_diag', 'gelman_rubin_eigen'],
                [gelman_rubin_diag, gelman_rubin_eigen]):
            if not quiet:
                self.log_info(f"{name}: {value:.3g}")
            if f'{key}_max' in criteria:
                threshold = criteria[f'{key}_max']
                passed = value < threshold
                converged = converged and passed
                if not quiet:
                    self.log_info(
                        f"{name}: {value:.3g} {'<' if passed else '>'} "
                        f"{threshold:.3g} ({'' if passed else 'not '}passed)")
            if f'{key}_min' in criteria:
                threshold = criteria[f'{key}_max']
                passed = value > threshold
                converged = converged and passed
                if not quiet:
                    self.log_info(
                        f"{name}: {value:.3g} {'>' if passed else '<'} "
                        f"{threshold:.3g} ({'' if passed else 'not '}passed)")

        return converged

    def run(self, start=None, start_tries=100, check_every=10, checks_passed=1,
            criteria=dict(gelman_rubin_diag_max=1.1), **kwargs):
        """Run the sampler.

        Parameters
        ----------
        max_tries : int or None, optional
            Maximum number of attempts for each chain. If None, there is
            no limit. Default is None.
        kwargs : dict, optional
            Keyword arguments passed to the run function of the sampler.

        Returns
        -------
        Chain
            Sampler results.

        """
        if self.mpicomm.rank == 0:

            # Determine a starting position if not provided.
            if self.chains is None and start is None:
                if start_tries is None:
                    start_tries = float('inf')

                start = np.zeros((self.n_chains, self.n_dim))
                log_p = np.repeat(-np.inf, self.n_chains)
                tries = 0

                while tries < start_tries and not np.all(np.isfinite(log_p)):
                    use = ~np.isfinite(log_p)
                    for i, param in enumerate(self.likelihood.varied_params):
                        if param.ref.is_proper():
                            start[use, i] = param.ref.sample(
                                size=np.sum(use), random_state=self.rng)
                        else:
                            start[use, i] = np.full(np.sum(use), param.value)
                    if self.mpicomm.rank == 0:
                        log_p[use] = list(map(compute_posterior, start[use]))
                    tries += 1

                if not np.all(np.isfinite(log_p)):
                    raise ValueError('Could not find finite posterior '
                                     f'after {start_tries:d} tries.')

            chains = self.run_sampler(start, check_every, **kwargs)
            n_passed = int(self.check(criteria))

            while n_passed < checks_passed:
                chains = self.run_sampler(None, check_every, **kwargs)
                if self.check(criteria):
                    n_passed += 1
                else:
                    n_passed = 0
        else:
            self.pool.wait()
        self.pool.stop_wait()

        return self.mpicomm.bcast(chains, root=0)
