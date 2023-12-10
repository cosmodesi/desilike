import os
import sys

import numpy as np

from desilike.samples import Chain
from .base import BasePosteriorSampler, load_source, Samples, batch_iterate


class FakePool(object):

    def __init__(self, size=1):
        self.size = size

    def map(self, func, values):
        return func(values)


class NautilusSampler(BasePosteriorSampler):

    check = None

    def __init__(self, *args, nlive=2000, n_update=None, enlarge_per_dim=1.1, n_points_min=None,
                 split_threshold=100, n_networks=4, neural_network_kwargs=None, n_like_new_bound=None, **kwargs):
        """
        Initialize nautilus sampler.

        Note
        ----
        nautilus requires state to be saved as hdf5, so requires h5py.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        nlive : int, default=2000
            Number of "live" points. New bounds are constructed so that
            they encompass the live points.

        n_update : int, default=None
            The maximum number of additions to the live set before a new bound
            is created. If ``None``, use ``nlive``.

        enlarge_per_dim : float, default=1.1
            Along each dimension, outer ellipsoidal bounds are enlarged by this
            factor.

        n_points_min : int, default=None
            The minimum number of points each ellipsoid should have.
            Effectively, ellipsoids with less than twice that number will not
            be split further. If ``None``, uses ``npoints_min = ndim + 50``.

        split_threshold: float, default=100
            Threshold used for splitting the multi-ellipsoidal bound used for
            sampling. If the volume of the bound prior enlarging is larger than
            ``split_threshold`` times the target volume, the multi-ellipsiodal
            bound is split further, if possible.

        n_networks : int, default=4
            Number of networks used in the estimator.

        neural_network_kwargs : dict, default=None
            Non-default keyword arguments passed to the constructor of
            MLPRegressor.

        n_like_new_bound : int, default=None
            The maximum number of likelihood calls before a new bounds is
            created. If None, use 10 times ``nlive``.

        rng : np.random.RandomState, default=None
            Random state. If ``None``, ``seed`` is used to set random state.

        seed : int, default=None
            Random seed.

        max_tries : int, default=1000
            A :class:`ValueError` is raised after this number of likelihood (+ prior) calls without finite posterior.

        chains : str, Path, Chain
            Path to or chains to resume from.

        ref_scale : float, default=1.
            Rescale parameters' :attr:`Parameter.ref` reference distribution by this factor.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`.
        """
        self.attrs = dict(n_live=int(nlive), n_update=n_update,
                          enlarge_per_dim=enlarge_per_dim, n_points_min=n_points_min,
                          split_threshold=split_threshold, n_networks=n_networks, neural_network_kwargs=neural_network_kwargs or {},
                          n_like_new_bound=n_like_new_bound)
        super(NautilusSampler, self).__init__(*args, **kwargs)
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save nautilus state')
        self.state_fn = [os.path.splitext(fn)[0] + '.nautilus.state.h5' for fn in self.save_fn]

    def loglikelihood(self, values):
        return self.logposterior(values) - self.logprior(values)

    def prior_transform(self, values):
        values = np.asarray(values)
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values.T, self.varied_params)):
            try:
                toret[..., iparam] = param.prior.ppf(value)
            except AttributeError as exc:
                raise AttributeError('{} has no attribute ppf (maybe infinite prior?). Choose proper prior for nested sampling'.format(param.prior)) from exc
        return toret

    def _prepare(self):
        self.resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)
        self.chains = [None] * len(self.chains)  # avoid chains to be concatenated

    def run(self, *args, **kwargs):
        """
        Run sampling. Sampling can be interrupted anytime, and resumed by providing
        the path to the saved chains in ``chains`` argument of :meth:`__init__`.

        One will typically run sampling on ``nchains * nprocs_per_chain`` processes,
        with ``nchains >= 1`` the number of chains and ``nprocs_per_chain = max(mpicomm.size // nchains, 1)``
        the number of processes per chain.

        Parameters
        ----------
        min_iterations : int, default=100
            Minimum number of iterations to run (to avoid early stopping
            if convergence criteria below are satisfied by chance at the beginning of the run).

        max_iterations : int, default=sys.maxsize
            Maximum number of iterations to run.

        check_every : int, default=300
            Samples are saved and convergence checks are run every ``check_every`` iterations.

        f_live : float, default=0.01
            Maximum fraction of the evidence contained in the live set before
            building the initial shells terminates.

        check : bool, dict, default=None
            If ``False``, no convergence checks are run.
            If ``True`` or ``None``, convergence checks are run.
            A dictionary of convergence criteria can be provided, with:

            - n_shell : Minimum number of points in each shell. The algorithm will sample
            from the shells until this is reached. Default is 100.

            - n_eff : Minimum effective sample size (ESS).
            The algorithm will sample from the shells until this is reached. Default is 10000.

        """
        return super(NautilusSampler, self).run(*args, **kwargs)

    def _run_one(self, start, min_iterations=0, max_iterations=sys.maxsize, check_every=300, check=None, **kwargs):

        import nautilus
        if check is not None and not isinstance(check, bool): kwargs.update(check)
        n_eff = kwargs.get('n_eff', 10000)
        n_shell = kwargs.get('n_shell', 100)

        #from desilike.utils import TaskManager as MPIPool
        #pool = (FakePool(size=self.mpicomm.size), MPIPool(mpicomm=self.mpicomm))  # yields bugs, _check_same_input fails
        pool = (FakePool(size=self.mpicomm.size), None)

        def write_derived(sampler):
            if self.mpicomm.rank == 0:
                samples, logw, logl = sampler.posterior(return_as_dict=True, equal_weight=False, return_blobs=False)
                chain = [samples[..., iparam] for iparam, param in enumerate(self.varied_params)]
                chain.append(logw)
                chain.append(np.exp(logw))
                chain = Chain(chain, params=self.varied_params + ['logweight', 'aweight'])
                if self.resume_derived is not None:
                    if self.derived is not None:
                        self.derived = [Samples.concatenate([resume_derived, derived], intersection=True) for resume_derived, derived in zip(self.resume_derived, self.derived)]
                    else:
                        self.derived = self.resume_derived
                chain = self._set_derived(chain)
                self.resume_chain = chain = self._set_derived(chain)
                self.resume_chain.save(self.save_fn[self._ichain])

        def wrapper(write_bak):

            def write(sampler, *args, **kwargs):
                self.mpicomm.Barrier()
                if self.mpicomm.rank == 0:
                    write_bak(sampler, *args, **kwargs)
                write_derived(sampler)
                self.mpicomm.Barrier()

            return write

        methods_bak = {name: getattr(nautilus.Sampler, name) for name in ['write', 'write_shell_update', 'write_shell_information_update']}
        for name, method in methods_bak.items():
            setattr(nautilus.Sampler, name, wrapper(method))

        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

        self.resume_derived, self.resume_chain = None, None
        seed = self.rng.randint(0, high=0xffffffff)
        if self.resume:
            self.sampler = nautilus.Sampler(self.prior_transform, self.loglikelihood, n_dim=len(self.varied_params), pool=pool, pass_dict=False,
                                            filepath=self.state_fn[self._ichain], seed=seed, **self.attrs)
            source = load_source(self.save_fn[self._ichain])[0]
            self.resume_derived = [source] * 2

        elif not hasattr(self, 'sampler'):
            self.sampler = nautilus.Sampler(self.prior_transform, self.loglikelihood, n_dim=len(self.varied_params), pool=pool, pass_dict=False,
                                            filepath=self.state_fn[self._ichain], resume=False, seed=seed, **self.attrs)

        def _run_one_batch(niterations):
            n_shell_current, n_eff_current = 0, 0
            if self.sampler.shell_n.size:
                n_shell_current = np.min(self.sampler.shell_n)
                n_eff_current = self.sampler.effective_sample_size()
            self.sampler.run(**{**kwargs, 'n_shell': n_shell_current + niterations,
                                          'n_eff': n_eff_current + niterations})
            write_derived(self.sampler)
            self.resume_derived = self.derived
            self.derived = None
            return not (np.any(self.sampler.shell_n < n_shell) or self.sampler.effective_sample_size() < n_eff)

        batch_iterate(_run_one_batch, min_iterations=min_iterations, max_iterations=max_iterations, check_every=check_every)

        for name, method in methods_bak.items():
            setattr(nautilus.Sampler, name, method)
        self.derived = self.resume_derived
        return self.resume_chain

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/johannesulf/nautilus')