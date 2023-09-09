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

    def __init__(self, *args, nlive=2000, nupdate=None, nlike_new_bound=None, enlarge_per_dim=1.1, npoints_min=None,
                 split_threshold=100, nnetworks=4, neural_network_kwargs=None, nbatch=100, **kwargs):
        """
        Initialize nautilus sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        nlive : int, default=2000
            Number of "live" points.

        nupdate : int, default=None
            The maximum number of additions to the live set before a new bound is created.

        nlike_new_bound : int, default=None
            The maximum number of likelihood calls before a new bounds is created.

        enlarge_per_dim : float, default=1.1
            Along each dimension, outer ellipsoidal bounds are enlarged by this factor.

        npoints_min : int, default=None
            The minimum number of points each ellipsoid should have. Effectively,
            ellipsoids with less than twice that number will not be split further.

        split_threshold: float, default=100
            Threshold used for splitting the multi-ellipsoidal bound used for
            sampling. If the volume of the bound prior enlarging is larger than
            `split_threshold` times the target volume, the multi-ellipsiodal
            bound is split further, if possible.

        nnetworks : int, default=4
            Number of networks used in the estimator.

        neural_network_kwargs : dict, default=None
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`.

        nbatch : int, default=100
            Number of likelihood evaluations that are performed at each step.

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
        self.attrs = dict(n_live=int(nlive), n_update=nupdate, nçlike_new_bound=nlike_new_bound,
                          enlarge_per_dim=enlarge_per_dim, n_points_min=npoints_min,
                          split_threshold=split_threshold, n_networks=nnetworks, neural_network_kwargs=neural_network_kwargs or {},
                          n_batch=nbatch)
        super(NautilusSampler, self).__init__(*args, **kwargs)
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save nautilus state')
        self.state_fn = [os.path.splitext(fn)[0] + '.nautilus.state' for fn in self.save_fn]

    def loglikelihood(self, values):
        return self.logposterior(values) - self.logprior(values)

    def prior_transform(self, values):
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values.T, self.varied_params)):
            try:
                toret[..., iparam] = param.prior.ppf(value)
            except AttributeError as exc:
                raise AttributeError('{} has no attribute ppf (maybe infinite prior?). Choose proper prior for nested sampling'.format(param.prior)) from exc
        return toret

    def _prepare(self):
        self.resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)

    def _run_one(self, start, min_iterations=0, max_iterations=sys.maxsize, check_every=300, check=None, **kwargs):
        from dynesty import utils
        if check is not None and not isinstance(check, bool): kwargs.update(check)

        rstate = np.random.Generator(np.random.PCG64(self.rng.randint(0, high=0xffffffff)))

        # Instantiation already runs somes samples
        if not hasattr(self, 'sampler'):
            use_pool = {'prior_transform': True, 'loglikelihood': True, 'propose_point': False, 'update_bound': False}
            pool = FakePool(size=self.mpicomm.size)
            self._set_sampler(rstate, pool, use_pool)

        self.resume_derived, self.resume_chain = None, None
        if self.resume:
            sampler = utils.restore_sampler(self.state_fn[self._ichain])
            del sampler.loglikelihood, sampler.prior_transform, sampler.pool, sampler.M
            if type(sampler) is not type(self.sampler):
                raise ValueError('Previous run used {}, not {}.'.format(type(sampler), type(self.sampler)))
            self.sampler.__dict__.update(sampler.__dict__)
            source = load_source(self.save_fn[self._ichain])[0]
            self.resume_derived = [source] * 2

        self.sampler.rstate = rstate

        def _run_one_batch(niterations):
            it = self.sampler.it
            self._run_nested(niterations, **kwargs)
            is_converged = self.sampler.it - it < niterations
            results = self.sampler.results
            chain = [results['samples'][..., iparam] for iparam, param in enumerate(self.varied_params)]
            #logprior = sum(param.prior(value) for param, value in zip(self.varied_params, chain))
            #chain.append(logprior)
            #chain.append(results['logl'] + logprior)
            chain.append(results['logwt'])
            chain.append(np.exp(results.logwt - results.logz[-1]))
            chain = Chain(chain, params=self.varied_params + ['logweight', 'aweight'])

            if self.mpicomm.rank == 0:
                if self.resume_derived is not None:
                    if self.derived is not None:
                        self.derived = [Samples.concatenate([resume_derived, derived], intersection=True) for resume_derived, derived in zip(self.resume_derived, self.derived)]
                    else:
                        self.derived = self.resume_derived
                chain = self._set_derived(chain)
                self.resume_chain = chain = self._set_derived(chain)
                self.resume_chain.save(self.save_fn[self._ichain])
                utils.save_sampler(self.sampler, self.state_fn[self._ichain])

            self.resume_derived = self.derived
            self.derived = None
            return is_converged

        batch_iterate(_run_one_batch, min_iterations=min_iterations, max_iterations=max_iterations, check_every=check_every)

        self.derived = self.resume_derived
        return self.resume_chain

    def _set_sampler(self, rstate, pool, use_pool):
        import nautilus
        self.sampler = nautilus.Sampler(self.loglikelihood, self.prior_transform, len(self.varied_params), pool=pool, use_pool=use_pool, rstate=rstate, **self.attrs)

    def _run_nested(self, niterations, **kwargs):
        self.sampler.run(maxiter=niterations, **kwargs)

    @classmethod
    def install(cls, config):
        config.pip('nautilus-sampler')