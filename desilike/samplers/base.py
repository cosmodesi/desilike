"""
Base classes for posterior samplers.

This module defines common functions and classes that are inherited by
specialized classes implementing specific samplers such as `emcee` or
`dynesty`.
"""

import logging
import numbers
import sys
import traceback
import warnings

import numpy as np

from desilike import utils, mpi, PipelineError
from desilike.jax import jit
from desilike.samples import Chain, Samples, load_source
from desilike.samples import diagnostics as sample_diagnostics
from desilike.utils import TaskManager, is_path, UserDict, BaseClass


def batch_iterate(func, min_iterations=0, max_iterations=sys.maxsize,
                  check_every=200, **kwargs):
    """
    Iterate over a function until convergence.

    Parameters
    ----------
    func : callable
        Function to be called. Must take ``niterations`` as a keyword argument.
    min_iterations : int, optional
        Minimum number of iterations. The default is 0.
    max_iterations : int, optional
        Maximum number of iterations. The default is ``sys.maxsize``.
    check_every : int, optional
        How often to check for convergence. If the number of iterations is
        at least ``min_iterations`` but below ``max_iterations``, the process
        will stop if convergence is achieved. The default is 200.
    **kwargs
        Additional keyword arguments passed to ``func``.

    Raises
    ------
    ValueError
        If ``max_iterations`` is negative or ``check_every`` is smaller than 1.

    """
    if max_iterations < 0:
        raise ValueError('max_iterations must be positive')
    if check_every < 1:
        raise ValueError(f'check_every must be >= 1, found {check_every:d}.')

    count_iterations = 0
    while count_iterations < max_iterations:
        niter = min(max_iterations - count_iterations, check_every)
        count_iterations += niter
        is_converged = func(niterations=niter, **kwargs)
        if is_converged and count_iterations >= min_iterations:
            break


class BasePosteriorSampler(BaseClass):
    """Base class defining common functions used by all samplers."""

    nwalkers = 1

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000,
                 chains=None, ref_scale=1., save_fn=None, mpicomm=None):
        """
        Initialize the posterior sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        rng : numpy.random.RandomState, optional
            Random state. If ``None``, ``seed`` is used to set random state.
            Default is ``None``.

        seed : int, optional
            Random seed. Default is ``None``.

        max_tries : int, default=1000
            A :class:`ValueError` is raised after this number of likelihood
            (+ prior) calls without finite posterior.

        chains : str, Path, Chain, or None
            Path to or chains to resume from. Default is ``None``.

        ref_scale : float, default=1.0
            If no chains to resume from are provided, initial points are
            sampled from parameters' :attr:`Parameter.ref` reference
            distributions. Rescale parameters' :attr:`Parameter.ref` reference
            distribution by this factor.

        save_fn : str, Path, optional
            If not ``None``, save samples to this location. Default is
            ``None``.

        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        """
        self.likelihood = likelihood
        if mpicomm is None:
            self.mpicomm = likelihood.mpicomm
        else:
            self.mpicomm = mpicomm
        # TODO: Find out what this does.
        self.likelihood.solved_default = '.marg'
        self.varied_params = self.likelihood.varied_params.deepcopy()
        for param in self.varied_params:
            param.update(ref=param.ref.affine_transform(scale=ref_scale))
        if not self.varied_params:
            raise ValueError(
                'No parameters to be varied! Cannot run sampling.')
        if self.mpicomm.rank == 0:
            self.log_info(f'Varied parameters: {self.varied_params.names()}.')
            if chains is None:
                if save_fn is not None and not is_path(save_fn):
                    chains = len(save_fn)
                else:
                    chains = 1
            if isinstance(chains, numbers.Number):
                self.chains = [None] * int(chains)
            else:
                self.chains = load_source(chains)

        nchains = self.mpicomm.bcast(
            len(self.chains) if self.mpicomm.rank == 0 else None, root=0)
        # TODO: Find out why root has the loaded chains and all other are
        # iniated empty. Are they ever used?
        if self.mpicomm.rank != 0:
            self.chains = [None] * nchains
        self.save_fn = save_fn
        if save_fn is not None:
            if is_path(save_fn):
                self.save_fn = [str(save_fn).replace('*', str(i))
                                for i in range(self.nchains)]
            else:
                if len(save_fn) != self.nchains:
                    raise ValueError(f'Need {self.nchains} chain file names. '
                                     f'Received {len(save_fn)}.')
        self.max_tries = int(max_tries)
        self._set_rng(rng=rng, seed=seed)
        self.diagnostics = {}
        self.derived = None

    def logposterior(self, values):
        """
        Compute the logarithm of the posterior.

        Parameters
        ----------
        values : numpy.ndarray of shape (n_points, n_dim)
            Points for which to compute the posterior.

        Raises
        ------
        PipelineError
            If the likelihood raised an error.

        Returns
        -------
        numpy.ndarray of shape (n_points, )
            Natural logarithm of the posterior.

        """
        log_prior = self.logprior(values)
        finite_prior = np.isfinite(log_prior)

        if not np.any(finite_prior):
            return log_prior

        points = Samples(values[finite_prior].T, params=self.varied_params)

        (log_posterior, derived), errors = self._vlikelihood(points.to_dict())

        if self.mpicomm.rank == 0:

            raise_error = None
            update_derived = True

            if errors:
                for i, error in errors.items():
                    if isinstance(error[0], self.likelihood.catch_errors):
                        point_repr = {k: v.flat[i] for k, v in points.items()}
                        self.log_debug(
                            f"Error \"{error[0]}\" raised with parameters "
                            f"{point_repr}. Assigining  -inf loglikelihood. "
                            f"Full stack trace \n{error[1]}:")
                        if not self.logger.isEnabledFor(logging.DEBUG):
                            warnings.warn(
                                f"Error \"{error[0]}\" raised. Set logging "
                                "level to debug (setup_logging (\"debug\")) "
                                "to get full stack trace.")
                        log_posterior[i] = -np.inf
                    else:
                        raise_error = error
                        update_derived = False

            if update_derived:
                if self.derived is None:
                    self.derived = [points, derived]
                else:
                    self.derived = [
                        Samples.concatenate([self.derived[0], points],
                                            intersection=True),
                        Samples.concatenate([self.derived[1], derived],
                                            intersection=True)]

            mask = np.isnan(log_posterior)
            if np.any(mask):
                log_posterior[mask] = -np.inf
                warnings.warn('Posterior is NaN for {}'.format(
                    {k: v[mask].tolist() for k, v in points.items()}))

            log_posterior_finite_prior = log_posterior
            log_posterior = log_prior
            log_posterior[finite_prior] = log_posterior_finite_prior

        else:
            log_posterior, raise_error = None, None

        raise_error = self.mpicomm.bcast(raise_error, root=0)
        if raise_error:
            raise PipelineError(
                f"Error \"{raise_error[0]}\" occured with stack trace:\n"
                f"{raise_error[1]}")

        return self.mpicomm.bcast(log_posterior, root=0)

    @jit(static_argnums=[0])
    def logprior(self, values):
        """
        Compute the logarithm of the prior.

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
            **dict(zip(self.varied_params.names(), values.T)))

    def __getstate__(self):
        """Return a pickleable represenation of the sampler."""
        return {name: getattr(self, name) for name in
                ['max_tries', 'diagnostics']}

    def _set_rng(self, rng=None, seed=None):
        """Set and synchronize the random number generator."""
        if rng is None:
            rng = np.random.RandomState(seed=mpi.bcast_seed(
                seed=seed, mpicomm=self.mpicomm, size=None))
        self.rng = self.mpicomm.bcast(rng, root=0)

    def _set_vlikelihood(self):
        """Vectorize the likelihood."""

        def get_start(size=1):
            toret = {}
            for param in self.varied_params:
                if param.ref.is_proper():
                    value = param.ref.sample(size=size, random_state=self.rng)
                else:
                    # TODO: This may create a problem for emcee. Investigate.
                    value = np.full(size, param.value)
                toret[param.name] = value
            return toret

        self.likelihood()  # initialize before jit
        vlikelihood = self.likelihood
        from desilike import vmap
        try:
            import jax
            _vlikelihood = vmap(vlikelihood, backend='jax',
                                errors='return', return_derived=True)
            _vlikelihood(get_start())
        except:
            if self.mpicomm.rank == 0:
                self.log_info(
                    'Could *not* vmap input likelihood. Set logging level to '
                    'debug (setup_logging("debug")) to get full stack trace.')
                self.log_debug(f'Error was {traceback.format_exc()}.')
            # TODO: This seems suspicious since vmap is applied later, anyway.
            vlikelihood = vmap(vlikelihood, backend=None,
                               errors='return', return_derived=True)
        else:
            if self.mpicomm.rank == 0:
                self.log_info('Successfully vmap input likelihood.')
            vlikelihood = _vlikelihood
        try:
            # TODO: Move import to top.
            import jax
            _vlikelihood = jax.jit(vlikelihood)
            _vlikelihood(get_start())
        except:
            if self.mpicomm.rank == 0:
                self.log_info('Could *not* jit input likelihood.')
        else:
            if self.mpicomm.rank == 0:
                self.log_info('Successfully jit input likelihood.')
            vlikelihood = _vlikelihood
        vlikelihood = vmap(vlikelihood, backend='mpi', errors='return')

        def _vlikelihood(*args, **kwargs):
            return vlikelihood(*args, **kwargs, mpicomm=self.mpicomm)
        self._vlikelihood = _vlikelihood

    def _prepare(self):
        """Prepare the sampler for the first iteration."""
        pass

    @property
    def nchains(self):
        """
        Return the number of chains.

        Returns
        -------
        int
            Number of chains.

        """
        return len(self.chains)

    def _get_start(self, start=None, max_tries=None):
        """
        Get starting values for the chains.

        Parameters
        ----------
        start : numpy.ndarray or None, optional
            Starting values of shape (n_chains, n_walkers, n_dim). If provided,
            this function simply checks they have the right shape. Default is
            ``None``.

        Raises
        ------
        ValueError
            If ``start`` is provided and does not have the correct shape or
            if no finite posterior can be found.

        Returns
        -------
        numpy.ndarray
            Starting values of shape (n_chains, n_walkers, n_dim).

        """
        # Make sure all processes have the same RNG.
        self._set_rng(rng=self.rng)

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                if param.ref.is_proper():
                    value = param.ref.sample(size=size, random_state=self.rng)
                else:
                    # TODO: This may create a problem for emcee. Investigate.
                    value = np.full(size, param.value)
                toret.append(value)
            return np.column_stack(toret)

        if getattr(self, '_vlikelihood', None) is None:
            self._set_vlikelihood()

        shape = (self.nchains, self.nwalkers, len(self.varied_params))
        if start is not None:
            start = np.asarray(start)
            if start.shape != shape:
                raise ValueError(f'Provide start with shape {shape}.')
            return start

        start = np.full(shape, np.nan)
        logposterior = np.full(shape[:2], -np.inf)
        for ichain, chain in enumerate(self.chains):
            if self.mpicomm.bcast(chain is not None and chain.size, root=0):
                start[ichain] = self.mpicomm.bcast(np.array(
                    [chain[param][-1] for param in self.varied_params]).T if
                    self.mpicomm.rank == 0 else None, root=0)
                logposterior[ichain] = self.logposterior(start[ichain])

        start = start.reshape(shape[0] * shape[1], -1)
        logposterior = logposterior.flatten()

        for itry in range(self.max_tries):
            is_finite = np.isfinite(logposterior)
            if is_finite.all():
                break
            values = get_start(size=(~is_finite).sum())
            start[~is_finite] = values
            logposterior[~is_finite] = self.logposterior(values)
        else:
            raise ValueError('Could not find finite log posterior after '
                             f'{self.max_tries} tries.')

        start = start.reshape(shape)

        # TODO: Investigate why the log of the posterior is not returned. This
        # probably leads to double computation.
        return start

    def _set_derived(self, chain):
        chain = Chain(chain, loglikelihood=self.likelihood._param_loglikelihood,
                      logprior=self.likelihood._param_logprior)
        for param in self.likelihood.all_params.select(fixed=True, derived=False):
            chain[param] = np.full(chain.shape, param.value, dtype='f8')
        if self.derived is not None:
            indices_in_chain, indices = self.derived[0].match(
                chain, params=self.varied_params)
            assert indices_in_chain[0].size == chain.size, '{:d} != {:d}, {}, {}'.format(
                indices_in_chain[0].size, chain.size, indices_in_chain, indices)
            for array in self.derived[1]:
                chain.set(array[indices].reshape(
                    chain.shape + array.shape[1:]))
        if chain._logposterior not in chain:
            chain.logposterior = chain[chain._loglikelihood][(
            )] + chain[chain._logprior][()]
        chain.logposterior.param.update(
            derived=True, latex=utils.outputs_to_latex(chain._logposterior))
        return chain

    def run(self, start=None, **kwargs):
        """
        Run chains. Sampling can be interrupted anytime, and resumed by providing
        the path to the saved chains in ``chains`` argument of :meth:`__init__`.

        One will typically run sampling on ``nchains * nprocs_per_chain`` processes,
        with ``nchains >= 1`` the number of chains and ``nprocs_per_chain = max(mpicomm.size // nchains, 1)``
        the number of processes per chain.
        """

        nprocs_per_chain = max(self.mpicomm.size // self.nchains, 1)
        chains = [None] * self.nchains
        ncalls = [None] * self.nchains
        start = self._get_start(start=start)

        mpicomm_bak = self.mpicomm
        self._prepare()

        # Compute results.
        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True,
                         mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                self._set_rng(rng=self.rng)
                # Set _ichain so that individual ranks know where to write
                # results.
                self._ichain = ichain
                chain = self._run_one(start[ichain], **kwargs)
                if self.mpicomm.rank == 0:
                    # TODO: Find out under what conditions self.derived can
                    # be None. This should be an edge case.
                    ncalls[ichain] = (self.derived[1].size if self.derived is
                                      not None else 0)
                    if chain is not None:
                        chains[ichain] = self._set_derived(chain)
                self.derived = None
        self.mpicomm = mpicomm_bak

        # Distribute results accross MPI ranks.
        for i in range(len(chains)):
            computed_chain = ncalls[i] is not None
            computed_chain = np.array(self.mpicomm.allgather(computed_chain))
            if np.sum(computed_chain) != 1:
                raise RuntimeError('Internal error in running the chains. '
                                   'Please report this bug.')
            root = np.argmax(computed_chain)
            ncalls[i] = self.mpicomm.bcast(ncalls[i], root=root)
            if self.mpicomm.bcast(chains[i] is not None, root=root):
                chains[i] = Chain.sendrecv(
                    chain, source=root, dest=0, mpicomm=self.mpicomm)

        self.diagnostics['ncall'] = ncalls
        self.diagnostics['naccepted'] = [
            chain.size if chain is not None else 0 for chain in chains]

        # Concatenate chains and write results, if applicable.
        if self.mpicomm.rank == 0:
            for i, (chain, new_chain) in enumerate(zip(self.chains, chains)):
                if new_chain is not None:
                    if chain is None:
                        self.chains[i] = new_chain.deepcopy()
                    else:
                        self.chains[i] = Chain.concatenate(chain, new_chain)
                    attrs = {name: getattr(self.likelihood, name, None) for
                             name in ['size', 'nvaried', 'ndof',
                                      'hartlap2007_factor',
                                      'percival2014_factor']}
                    self.chains[i].attrs.update(attrs)
            if self.save_fn is not None:
                for save_fn, chain in zip(self.save_fn, self.chains):
                    if chain is not None:
                        chain.save(save_fn)
        return self.chains


class BaseBatchPosteriorSampler(BasePosteriorSampler):

    """Base class for samplers which can run independent chains in parallel."""

    def run(self, min_iterations=0, max_iterations=sys.maxsize, check_every=300, check=None, **kwargs):
        """
        Run chains. Sampling can be interrupted anytime, and resumed by providing
        the path to the saved chains in ``chains`` argument of :meth:`__init__`.

        One will typically run sampling on ``nchains * nprocs_per_chain + 1`` processes,
        with ``nchains >= 1`` the number of chains and ``nprocs_per_chain = max((mpicomm.size - 1) // nchains, 1)``
        the number of processes per chain --- plus 1 root process to distribute the work.

        Parameters
        ----------
        min_iterations : int, default=100
            Minimum number of iterations (MCMC steps) to run (to avoid early stopping
            if convergence criteria below are satisfied by chance at the beginning of the run).

        max_iterations : int, default=sys.maxsize
            Maximum number of iterations (MCMC steps) to run.

        check_every : int, default=300
            Samples are saved and convergence checks are run every ``check_every`` iterations.

        check : bool, dict, default=None
            If ``False``, no convergence checks are run.
            If ``True`` or ``None``, convergence checks are run.
            A dictionary of convergence criteria can be provided, see :meth:`check`.

        **kwargs : dict
            Optional sampler-specific arguments.
        """
        # self.derived = None
        nprocs_per_chain = max(self.mpicomm.size // self.nchains, 1)

        run_check = bool(check) or isinstance(check, dict)
        if run_check and not isinstance(check, dict):
            check = {}

        def _run_batch(niterations):
            chains, ncalls = [[None] * self.nchains for i in range(2)]
            start = self._get_start()
            mpicomm_bak = self.mpicomm

            self._prepare()

            with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:

                self.mpicomm = tm.mpicomm

                for ichain in tm.iterate(range(self.nchains)):
                    self._set_rng(rng=self.rng)
                    self._ichain = ichain
                    chain = self._run_one(
                        start[ichain], niterations=niterations, **kwargs)
                    if self.mpicomm.rank == 0:
                        ncalls[ichain] = self.derived[1][self.likelihood._param_loglikelihood].size if self.derived is not None else 0
                        if chain is not None:
                            chains[ichain] = self._set_derived(chain)
                    self.derived = None
            self.mpicomm = mpicomm_bak

            for ichain, chain in enumerate(chains):
                mpiroot_worker = self.mpicomm.rank if ncalls[ichain] is not None else None
                for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                    if mpiroot_worker is not None:
                        break
                assert mpiroot_worker is not None
                ncalls[ichain] = self.mpicomm.bcast(
                    ncalls[ichain], root=mpiroot_worker)
                if self.mpicomm.bcast(chain is not None, root=mpiroot_worker):
                    chains[ichain] = Chain.sendrecv(
                        chain, source=mpiroot_worker, dest=0, mpicomm=self.mpicomm)

            self.diagnostics['ncall'] = ncalls
            self.diagnostics['naccepted'] = [
                chain.size if chain is not None else 0 for chain in chains]

            if self.mpicomm.rank == 0:
                for ichain, (chain, new_chain) in enumerate(zip(self.chains, chains)):
                    if new_chain is not None:
                        if chain is None:
                            self.chains[ichain] = new_chain.deepcopy()
                        else:
                            self.chains[ichain] = Chain.concatenate(
                                chain, new_chain)
                        attrs = {name: getattr(self.likelihood, name, None) for name in [
                            'size', 'nvaried', 'ndof', 'hartlap2007_factor', 'percival2014_factor']}
                        self.chains[ichain].attrs.update(attrs)
                if self.save_fn is not None:
                    for ichain, chain in enumerate(self.chains):
                        if chain is not None:
                            chain.save(self.save_fn[ichain])

            is_converged = False
            if run_check:
                is_converged = self.check(**check)
            return is_converged

        batch_iterate(_run_batch, min_iterations=min_iterations,
                      max_iterations=max_iterations, check_every=check_every)
        return self.chains

    def check(self, nsplits=4, burnin=0.5, stable_over=2,
              max_eigen_gr=0.03, max_diag_gr=None, max_cl_diag_gr=None, nsigmas_cl_diag_gr=1., max_geweke=None, max_geweke_pvalue=None,
              min_ess=None, reliable_ess=50, max_dact=None,
              min_eigen_gr=None, min_diag_gr=None, min_cl_diag_gr=None, min_geweke=None, min_geweke_pvalue=None,
              max_ess=None, min_dact=None, diagnostics=None, quiet=False, **kwargs):
        """
        Run convergence checks.

        Parameters
        ----------
        nsplits : int, default=4
            Chains are divided into ``nsplits`` to run convergence tests.

        burnin : float, int, default=0.5
            Fraction of samples to remove from each chain for convergence tests.
            If an integer, number of iterations (steps) to remove.

        stable_over : int, default=2
            Each criterion must be fulfilled for ``stable_over`` calls to :meth:`check` for convergence tests to pass.

        max_eigen_gr : float, default=0.03
            Gelman-Rubin criterion (on eigenvalues of the parameter covariance matrix) < ``max_eigen_gr``

        max_diag_gr : float, default=None
            Gelman-Rubin criterion (on diagonal of the parameter covariance matrix) < ``max_eigen_gr``

        max_cl_diag_gr : float, default=None
            Gelman-Rubin criterion on variance of ``nsigmas_cl_diag_gr``-sigma interval limits < ``max_cl_diag_gr``

        nsigmas_cl_diag_gr : int, default=1
            Number of sigmas for the interval of ``max_cl_diag_gr`` test.

        min_ess : int, default=None
            Minimal number of iterations over integrated auto-correlation time (~ # of independent samples). Typically of order ~ 1e3.

        reliable_ess : int, default=50
            After ``reliable_ess`` auto-correlation time estimation is considered reliable.

        diagnostics : dict, default=None
            Dictionary where computed statistics are added.
            Default is :attr:`diagnostics`.

        quiet : bool, default=False
            If ``True``, no logging.

        Note
        ----
        All max_* have a min_* counterpart, and vice-versa.

        Returns
        -------
        convergence : bool
            ``True`` if current chains pass convergence tests.
        """
        toret = None
        has_input_diagnostics = diagnostics is not None
        diagnostics_bak = diagnostics if has_input_diagnostics else self.diagnostics
        diagnostics = Diagnostics(diagnostics_bak)
        assert nsplits > 1

        if self.mpicomm.bcast(any(chain is None for chain in self.chains), root=0):
            toret = False

        elif self.mpicomm.rank == 0:

            if 0 < burnin < 1:
                burnin = int(burnin * self.chains[0].shape[0] + 0.5)

            nsplits = int((nsplits + len(self.chains) - 1) / len(self.chains))
            lensplits = (self.chains[0].shape[0] - burnin) // nsplits

            split_samples = [chain[burnin + islab * lensplits:burnin + (
                islab + 1) * lensplits] for islab in range(nsplits) for chain in self.chains]

            if any(samples.size < 1 for samples in split_samples):
                toret = False
            else:
                kw = dict(stable_over=stable_over, quiet=quiet)
                if not quiet:
                    self.log_info('Diagnostics:')
                toret = True

                try:
                    eigen_gr = sample_diagnostics.gelman_rubin(
                        split_samples, self.varied_params, method='eigen', check_valid='ignore').max() - 1
                except ValueError:
                    eigen_gr = np.nan
                toret &= diagnostics.add_test(
                    'eigen_gr', 'max eigen Gelman-Rubin - 1', eigen_gr, limits=(min_eigen_gr, max_eigen_gr), **kw)

                try:
                    diag_gr = sample_diagnostics.gelman_rubin(
                        split_samples, self.varied_params, method='diag').max() - 1
                except ValueError:
                    diag_gr = np.nan
                toret &= diagnostics.add_test(
                    'diag_gr', 'max diag Gelman-Rubin - 1', diag_gr, limits=(min_diag_gr, max_diag_gr), **kw)

                def cl_lower(samples, params):
                    return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr)[0] for param in params])

                def cl_upper(samples, params):
                    return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr)[1] for param in params])

                try:
                    cl_diag_gr = np.max([sample_diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_lower, method='diag'),
                                         sample_diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_upper, method='diag')]) - 1
                except ValueError:
                    cl_diag_gr = np.nan
                toret &= diagnostics.add_test('cl_diag_gr', 'max diag Gelman-Rubin - 1 at {:.1f} sigmas'.format(
                    nsigmas_cl_diag_gr), cl_diag_gr, limits=(min_cl_diag_gr, max_cl_diag_gr), **kw)

                try:
                    # Source: https://github.com/JohannesBuchner/autoemcee/blob/38feff48ae524280c8ea235def1f29e1649bb1b6/autoemcee.py#L337
                    all_geweke = sample_diagnostics.geweke(
                        split_samples, self.varied_params, first=0.1, last=0.5)
                except ValueError:
                    all_geweke = np.nan
                geweke = np.max(all_geweke)
                toret &= diagnostics.add_test(
                    'geweke', 'max Geweke', geweke, limits=(min_geweke, max_geweke), **kw)

                from scipy import stats
                try:
                    geweke_pvalue = stats.normaltest(
                        all_geweke, axis=None).pvalue
                except ValueError:
                    geweke_pvalue = np.nan
                toret &= diagnostics.add_test(
                    'geweke_pvalue', 'Geweke p-value', geweke_pvalue, limits=(min_geweke_pvalue, max_geweke_pvalue), **kw)

                split_samples = []
                for chain in self.chains:
                    chain = chain[burnin:]
                    chain = chain.reshape(len(chain), -1)
                    for iwalker in range(chain.shape[1]):
                        split_samples.append(chain[:, iwalker])
                try:
                    iact = sample_diagnostics.integrated_autocorrelation_time(
                        split_samples, self.varied_params, check_valid='ignore')
                except ValueError:
                    iact = np.full(len(self.varied_params), np.nan, dtype='f8')
                diagnostics.add('iact', iact)
                niterations = len(split_samples[0])
                iact = iact.max()
                name = 'effective sample size = ({:d} iterations / integrated autocorrelation time)'.format(
                    niterations)
                if reliable_ess * iact < niterations:
                    name = '{} (reliable)'.format(name)
                toret &= diagnostics.add_test(
                    'iterations_over_iact', name, niterations / iact, limits=(min_ess, max_ess), **kw)

                iact = diagnostics['iact']
                if len(iact) >= 2:
                    rel = np.abs(iact[-2] / iact[-1] - 1).max()
                    toret &= diagnostics.add_test(
                        'dact', 'max variation of integrated autocorrelation time', rel, limits=(min_dact, max_dact), **kw)

        toret = self.mpicomm.bcast(toret, root=0)
        diagnostics = self.mpicomm.bcast(diagnostics, root=0)
        toret &= self._add_check(
            diagnostics, quiet=quiet, stable_over=stable_over, **kwargs)

        diagnostics_bak.update(self.mpicomm.bcast(diagnostics, root=0))

        toret = self.mpicomm.bcast(toret, root=0)
        if has_input_diagnostics:
            return toret, diagnostics
        return toret

    def _add_check(self, diagnostics, quiet=False, stable_over=2, **kwargs):
        """Implement sampler-specific checks."""
        return True


class MetaClass(type(BaseClass), type(UserDict)):

    pass


class Diagnostics(UserDict, BaseClass, metaclass=MetaClass):

    def add(self, key, value):
        if key not in self:
            self[key] = [value]
        else:
            self[key].append(value)
        return value

    def is_stable(self, key, stable_over=2):
        if len(self[key]) < stable_over:
            return False
        return all(self[key][-stable_over:])

    def add_test(self, key, name, value, limits=None, stable_over=2, quiet=False):
        item = '- '
        self.add(key, value)
        key = '{}_test'.format(key)
        msg = '{}{} is {:.3g}'.format(item, name, value)
        if limits is None:
            limits = [None] * 2

        def bool_test(value, low=None, up=None):
            test = True
            if low is not None:
                test &= value > low
            if up is not None:
                test &= value < up
            return test

        def log_test(msg, test, low=None, up=None):
            if low is None:
                low = ''
            else:
                low = '{:.3g}'.format(low)
            if up is None:
                up = ''
            else:
                up = '{:.3g}'.format(up)
            isnot = '' if test else 'not '
            if not (low or up):
                msg = '{}.'.format(msg)
            elif (low and up):
                msg = '{}; {}in [{}, {}].'.format(msg, isnot, low, up)
            elif low:
                msg = '{}; {}> {}.'.format(msg, isnot, low)
            elif up:
                msg = '{}; {}< {}.'.format(msg, isnot, up)
            self.log_info(msg)

        if any(li is not None for li in limits):
            test = bool_test(value, *limits)
            if not quiet:
                log_test(msg, test, *limits)
            self.add(key, test)
            return self.is_stable(key, stable_over=stable_over)

        if not quiet:
            self.log_info('{}.'.format(msg))
        return True
