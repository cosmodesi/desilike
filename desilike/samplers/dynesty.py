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


class BaseDynestySampler(BasePosteriorSampler):

    check = None

    def __init__(self, *args, nlive=500, bound='multi', sample='auto', update_interval=None, **kwargs):
        """
        Initialize dynesty sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        nlive : int, default=500
            Number of "live" points. Larger numbers result in a more finely sampled posterior (more accurate evidence),
            but also a larger number of iterations required to converge.

        bound : str, default='multi'
            Method used to approximately bound the prior using the current
            set of live points. Conditions the sampling methods used to
            propose new live points. Choices are no bound ('none'), a single
            bounding ellipsoid ('single'), multiple bounding ellipsoids
            ('multi'), balls centered on each live point ('balls'), and
            cubes centered on each live point ('cubes').

        sample : str, default='auto'
            Method used to sample uniformly within the likelihood constraint,
            conditioned on the provided bounds. Unique methods available are:
            uniform sampling within the bounds('unif'),
            random walks with fixed proposals ('rwalk'),
            multivariate slice sampling along preferred orientations ('slice'),
            "random" slice sampling along all orientations ('rslice'),
            "Hamiltonian" slices along random trajectories ('hslice'), and
            any callable function which follows the pattern of the sample methods
            defined in :mod:`dynesty.sampling`.
            'auto' selects the sampling method based on the dimensionality of the problem (ndim).
            When ndim < 10, this defaults to 'unif'.
            When 10 <= ndim <= 20, this defaults to 'rwalk'.
            When 'ndim > 20', this defaults to 'hslice' if a 'gradient' is provided and 'rslice' otherwise.
            'slice' is provided as alternatives for 'rslice'.

        update_interval : int, float, default=None
            If an integer is passed, only update the proposal distribution
            every update_interval-th likelihood call. If a float is passed,
            update the proposal after every ``round(update_interval * nlive)``-th likelihood call.
            Larger update intervals larger can be more efficient when the likelihood function is quick to evaluate.
            Default behavior is to target a roughly constant change in prior volume, with
            1.5 for 'unif', 0.15 * walks for 'rwalk', 0.9 * ndim * slices for 'slice', 2.0 * slices for 'rslice',
            and 25.0 * slices for 'hslice'.

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
        self.nlive = int(nlive)
        self.attrs = {'bound': bound, 'sample': sample, 'update_interval': update_interval}
        super(BaseDynestySampler, self).__init__(*args, **kwargs)
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save dynesty state')
        self.state_fn = [os.path.splitext(fn)[0] + '.dynesty.state' for fn in self.save_fn]

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

    @classmethod
    def install(cls, config):
        config.pip('dynesty')


class StaticDynestySampler(BaseDynestySampler):
    """
    Wrapper for dynesty (static) nested sampler. Proper priors only are supported.
    Using less "informative" priors will increase the expected number of nested sampling iterations.
    Static nested sampling is designed to estimate the evidence. For posterior estimation, rather use dynamic nested sampling.

    Reference
    ---------
    - https://github.com/joshspeagle/dynesty
    - https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract
    - https://zenodo.org/record/7600689#.Y-GI1RyZNkg
    """
    def run(self, *args, **kwargs):
        """
        Run sampling. Sampling can be interrupted anytime, and resumed by providing
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
            A dictionary of convergence criteria can be provided, with:

            - dlogz : Iteration will stop when the estimated contribution of the
              remaining prior volume to the total evidence falls below
              this threshold. Explicitly, the stopping criterion is
              :math:`\ln(z + z_{est}) - \ln(z) < dlogz`, where :math:`z` is the current
              evidence from all saved samples and :math:`z_{est}` is the estimated
              contribution from the remaining volume.
              The default is 1e-3 * (nlive - 1) + 0.01.

            - n_effective : Minimum number of effective posterior samples.
              If the estimated effective sample size (ESS) exceeds this number,
              sampling will terminate. Default is inf.

        """
        return super(StaticDynestySampler, self).run(*args, **kwargs)

    def _set_sampler(self, rstate, pool, use_pool):
        import dynesty

        self.sampler = dynesty.NestedSampler(self.loglikelihood, self.prior_transform, len(self.varied_params), nlive=self.nlive, pool=pool, use_pool=use_pool, rstate=rstate, **self.attrs)

    def _run_nested(self, niterations, **kwargs):
        self.sampler.run_nested(maxiter=niterations, **kwargs)


class DynamicDynestySampler(BaseDynestySampler):
    """
    Wrapper for dynesty (dynamic) nested sampler, for posterior estimation.

    Reference
    ---------
    - https://github.com/joshspeagle/dynesty
    - https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract
    - https://zenodo.org/record/7600689#.Y-GI1RyZNkg
    """
    def run(self, *args, **kwargs):
        """
        Run sampling. Sampling can be interrupted anytime, and resumed by providing
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
            A dictionary of convergence criteria can be provided, with:

            - dlogz : The baseline run will stop when the estimated contribution of the
              remaining prior volume to the total evidence falls below
              this threshold. Explicitly, the stopping criterion is
              :math:`\ln(z + z_{est}) - \ln(z) < dlogz`, where :math:`z` is the current
              evidence from all saved samples and :math:`z_{est}` is the estimated
              contribution from the remaining volume. The default is 0.01

            - n_effective : Minimum number of effective posterior samples.
              If the estimated effective sample size (ESS) exceeds this number,
              sampling will terminate. Default is ``max(10000, ndim**2)``.

        """
        return super(DynamicDynestySampler, self).run(*args, **kwargs)

    def _set_sampler(self, rstate, pool, use_pool):
        import dynesty

        self.sampler = dynesty.DynamicNestedSampler(self.loglikelihood, self.prior_transform, len(self.varied_params), pool=pool, use_pool=use_pool, rstate=rstate, **self.attrs)

    def _run_nested(self, niterations, **kwargs):
        self.sampler.run_nested(nlive_init=self.nlive, maxiter=niterations + self.sampler.it, **kwargs)
