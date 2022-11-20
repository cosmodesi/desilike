import sys
import numbers
import functools

import numpy as np

from desilike import mpi
from desilike.utils import BaseClass
from desilike.samples import Chain, Samples, load_source
from desilike.samples import diagnostics as sample_diagnostics
from desilike.parameter import ParameterPriorError, ParameterArray
from .utils import TaskManager


class RegisteredSampler(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


def batch_iterate(func, min_iterations=0, max_iterations=sys.maxsize, check_every=200, **kwargs):
    count_iterations = 0
    is_converged = False
    if check_every < 1:
        raise ValueError('check_every must be >= 1, found {:d}'.format(check_every))
    while not is_converged:
        niter = min(max_iterations - count_iterations, check_every)
        count_iterations += niter
        is_converged = func(niterations=niter, **kwargs)
        if count_iterations < min_iterations:
            is_converged = False
        if count_iterations >= max_iterations:
            is_converged = True


def bcast_values(func):

    @functools.wraps(func)
    def wrapper(self, values):
        values = np.asarray(values)
        if self._check_same_input:
            all_values = self.likelihood.mpicomm.allgather(values)
            if not all(np.allclose(values, all_values[0], atol=0., rtol=1e-7, equal_nan=True) for values in all_values if values is not None):
                raise ValueError('Input values different on all ranks: {}'.format(all_values))
        values = self.likelihood.mpicomm.bcast(values, root=0)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        mask = ~np.isnan(values).any(axis=1)
        toret = np.full(values.shape[0], -np.inf)
        values = values[mask]
        if values.size:
            toret[mask] = func(self, values)
        if isscalar and toret.size:
            toret = toret[0]
        return toret

    return wrapper


class BasePosteriorSampler(BaseClass, metaclass=RegisteredSampler):

    name = 'base'
    nwalkers = 1
    _check_same_input = False

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, chains=None, ref_scale=1., save_fn=None, mpicomm=None):
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = likelihood
        self.pipeline = self.likelihood.runtime_info.pipeline
        self.mpicomm = mpicomm
        self.likelihood.solved_default = '.marg'
        self.varied_params = self.likelihood.varied_params.deepcopy()
        for param in self.varied_params: param.ref = param.ref.affine_transform(scale=ref_scale)
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        if self.mpicomm.rank == 0:
            if chains is None: chains = 1
            if isinstance(chains, numbers.Number):
                self.chains = [None] * int(chains)
            else:
                self.chains = load_source(chains)
        nchains = self.mpicomm.bcast(len(self.chains) if self.mpicomm.rank == 0 else None, root=0)
        if self.mpicomm.rank != 0:
            self.chains = [None] * nchains
        self.save_fn = save_fn
        if save_fn is not None:
            if isinstance(save_fn, str):
                self.save_fn = [save_fn.replace('*', '{}').format(i) for i in range(self.nchains)]
            else:
                if len(save_fn) != self.nchains:
                    raise ValueError('Provide {:d} chain file names'.format(self.nchains))
        self.max_tries = int(max_tries)
        self._set_rng(rng=rng, seed=seed)
        self.diagnostics = {}
        self.derived = None

    @bcast_values
    def loglikelihood(self, values):
        points = Samples(values.T, params=self.varied_params)
        self.pipeline.mpicalculate(**points.to_dict())
        toret = None
        if self.pipeline.mpicomm.rank == 0:
            if self.derived is None:
                self.derived = [points, self.pipeline.derived]
            else:
                self.derived = [Samples.concatenate([self.derived[0], points]),
                                Samples.concatenate([self.derived[1], self.pipeline.derived])]
            toret = self.pipeline.derived[self.likelihood._loglikelihood_name] + self.pipeline.derived[self.likelihood._logprior_name]
        else:
            self.derived = None
        toret = self.likelihood.mpicomm.bcast(toret, root=0)
        mask = np.isnan(toret)
        toret[mask] = -np.inf
        if mask.any() and self.mpicomm.rank == 0:
            self.log_warning('loglikelihood is NaN for {}'.format({k: v[mask] for k, v in points.items()}))
        return toret

    @bcast_values
    def logprior(self, values):
        toret = 0.
        for param, value in zip(self.varied_params, values.T):
            toret += param.prior(value)
        return toret

    @bcast_values
    def logposterior(self, values):
        toret = self.logprior(values)
        mask = ~np.isinf(toret)
        toret[mask] = self.loglikelihood(values[mask])
        return toret

    def __getstate__(self):
        state = {}
        for name in ['max_tries', 'diagnostics']:
            state[name] = getattr(self, name)
        return state

    def _set_rng(self, rng=None, seed=None):
        self.rng = self.mpicomm.bcast(rng, root=0)
        if self.rng is None:
            seed = mpi.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None)
            self.rng = np.random.RandomState(seed=seed)

    def _prepare(self):
        pass

    @property
    def nchains(self):
        return len(self.chains)

    def _get_start(self, max_tries=None):
        if max_tries is None:
            max_tries = self.max_tries

        self._set_rng(rng=self.rng)  # to make sure all processes have the same rng

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                try:
                    toret.append(param.ref.sample(size=size, random_state=self.rng))
                except ParameterPriorError as exc:
                    raise ParameterPriorError('Error in ref/prior distribution of parameter {}'.format(param)) from exc
            return np.array(toret).T

        start = np.full((self.nchains, self.nwalkers, len(self.varied_params)), np.nan)
        logposterior = np.full((self.nchains, self.nwalkers), -np.inf)
        for ichain, chain in enumerate(self.chains):
            if self.mpicomm.bcast(chain is not None, root=0):
                start[ichain] = self.mpicomm.bcast(np.array([chain[param][-1] for param in self.varied_params]).T if self.mpicomm.rank == 0 else None, root=0)
                if not np.isnan(start[ichain]).all():
                    logposterior[ichain] = self.logposterior(start[ichain])

        start.shape = (self.nchains * self.nwalkers, -1)
        logposterior.shape = -1

        for itry in range(max_tries):
            mask = np.isfinite(logposterior)
            if mask.all(): break
            mask = ~mask
            values = get_start(size=mask.sum())
            start[mask] = values
            logposterior[mask] = self.logposterior(values)

        if not np.isfinite(logposterior).all():
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(max_tries))

        start.shape = (self.nchains, self.nwalkers, -1)
        return start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = self.pipeline.mpicomm = mpicomm

    def _set_derived(self, chain):
        for param in self.pipeline.params.select(fixed=True, derived=False):
            chain.set(ParameterArray(np.full(chain.shape, param.value, dtype='f8'), param))
        indices_in_chain, indices = self.derived[0].match(chain, params=self.varied_params)
        assert indices_in_chain[0].size == chain.size, '{:d} != {:d}'.format(indices_in_chain[0].size, chain.size)
        for array in self.derived[1]:
            chain.set(array[indices].reshape(chain.shape + array.shape[1:]))
        return chain

    def run(self, **kwargs):
        #self.derived = None
        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)
        chains, ncalls = [[None] * self.nchains for i in range(2)]
        start = self._get_start()
        mpicomm_bak = self.mpicomm
        self._prepare()
        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                self._set_rng(rng=self.rng)
                self.derived = None
                self._ichain = ichain
                chain = self._run_one(start[ichain], **kwargs)
                if self.mpicomm.rank == 0:
                    ncalls[ichain] = self.derived[1]['loglikelihood'].size if self.derived is not None else 0
                    if chain is not None:
                        chains[ichain] = self._set_derived(chain)
        self.mpicomm = mpicomm_bak

        for ichain, chain in enumerate(chains):
            mpiroot_worker = self.mpicomm.rank if ncalls[ichain] is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            ncalls[ichain] = self.mpicomm.bcast(ncalls[ichain], root=mpiroot_worker)
            if self.mpicomm.bcast(chain is not None, root=mpiroot_worker):
                chains[ichain] = Chain.sendrecv(chain, source=mpiroot_worker, dest=0, mpicomm=self.mpicomm)

        self.diagnostics['ncall'] = ncalls
        self.diagnostics['naccepted'] = [chain.size if chain is not None else 0 for chain in chains]
        if self.mpicomm.rank == 0:
            for ichain, (chain, new_chain) in enumerate(zip(self.chains, chains)):
                if new_chain is not None:
                    if chain is None:
                        self.chains[ichain] = new_chain
                    else:
                        self.chains[ichain] = Chain.concatenate(chain, new_chain)
            if self.save_fn is not None:
                for ichain, chain in enumerate(self.chains):
                    if chain is not None: chain.save(self.save_fn[ichain])


class BaseBatchPosteriorSampler(BasePosteriorSampler):

    def run(self, min_iterations=0, max_iterations=sys.maxsize, check_every=300, check=None, **kwargs):
        #self.derived = None
        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)

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
                    self.derived = None
                    self._ichain = ichain
                    chain = self._run_one(start[ichain], niterations=niterations, **kwargs)
                    if self.mpicomm.rank == 0:
                        ncalls[ichain] = self.derived[1]['loglikelihood'].size if self.derived is not None else 0
                        if chain is not None:
                            chains[ichain] = self._set_derived(chain)
            self.mpicomm = mpicomm_bak

            for ichain, chain in enumerate(chains):
                mpiroot_worker = self.mpicomm.rank if ncalls[ichain] is not None else None
                for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                    if mpiroot_worker is not None: break
                assert mpiroot_worker is not None
                ncalls[ichain] = self.mpicomm.bcast(ncalls[ichain], root=mpiroot_worker)
                if self.mpicomm.bcast(chain is not None, root=mpiroot_worker):
                    chains[ichain] = Chain.sendrecv(chain, source=mpiroot_worker, dest=0, mpicomm=self.mpicomm)

            self.diagnostics['ncall'] = ncalls
            self.diagnostics['naccepted'] = [chain.size if chain is not None else 0 for chain in chains]

            if self.mpicomm.rank == 0:
                for ichain, (chain, new_chain) in enumerate(zip(self.chains, chains)):
                    if new_chain is not None:
                        if chain is None:
                            self.chains[ichain] = new_chain.deepcopy()
                        else:
                            self.chains[ichain] = Chain.concatenate(chain, new_chain)
                if self.save_fn is not None:
                    for ichain, chain in enumerate(self.chains):
                        if chain is not None: chain.save(self.save_fn[ichain])

            is_converged = False
            if run_check:
                is_converged = self.check(**check)
            return is_converged

        batch_iterate(_run_batch, min_iterations=min_iterations, max_iterations=max_iterations, check_every=check_every)

    def check(self, nsplits=4, stable_over=2, burnin=0.5,
              max_eigen_gr=0.03, max_diag_gr=None, max_cl_diag_gr=None, nsigmas_cl_diag_gr=1., max_geweke=None, max_geweke_pvalue=None,
              min_iterations_over_iact=None, reliable_iterations_over_iact=50, max_dact=None,
              min_eigen_gr=None, min_diag_gr=None, min_cl_diag_gr=None, min_geweke=None, min_geweke_pvalue=None,
              max_iterations_over_iact=None, min_dact=None, diagnostics=None, quiet=False):

        toret = None
        if diagnostics is None:
            diagnostics = self.diagnostics
        verbose = not quiet
        if self.mpicomm.bcast(any(chain is None for chain in self.chains), root=0):
            return False

        if self.mpicomm.rank == 0:

            def add_diagnostics(name, value):
                if name not in diagnostics:
                    diagnostics[name] = [value]
                else:
                    diagnostics[name].append(value)
                return value

            def is_stable(name):
                if len(diagnostics[name]) < stable_over:
                    return False
                return all(diagnostics[name][-stable_over:])

            def bool_test(value, low=None, up=None):
                test = True
                if low is not None:
                    test &= value > low
                if up is not None:
                    test &= value < up
                return test

            def log_test(msg, test, low=None, up=None):
                if verbose:
                    if low is None: low = ''
                    else: low = '{:.3g}'.format(low)
                    if up is None: up = ''
                    else: up = '{:.3g}'.format(up)
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

            def full_test(key, name, value, low=None, up=None):
                add_diagnostics(key, value)
                key = '{}_test'.format(key)
                msg = '{}{} is {:.3g}'.format(item, name, value)
                if any(lu is not None for lu in (low, up)):
                    test = bool_test(value, low=low, up=up)
                    log_test(msg, test, low=low, up=up)
                    add_diagnostics(key, test)
                    return is_stable(key)
                if verbose:
                    self.log_info('{}.'.format(msg))
                return True

            if 0 < burnin < 1:
                burnin = int(burnin * self.chains[0].shape[0] + 0.5)

            lensplits = (self.chains[0].shape[0] - burnin) // nsplits

            split_samples = [chain[burnin + islab * lensplits:burnin + (islab + 1) * lensplits] for islab in range(nsplits) for chain in self.chains]

            if any(samples.size < 1 for samples in split_samples):
                toret = False
            else:
                if verbose: self.log_info('Diagnostics:')
                item = '- '
                toret = True

                try:
                    eigen_gr = sample_diagnostics.gelman_rubin(split_samples, self.varied_params, method='eigen', check_valid='ignore').max() - 1
                except ValueError:
                    eigen_gr = np.nan
                toret &= full_test('eigen_gr', 'max eigen Gelman-Rubin - 1', eigen_gr, min_eigen_gr, max_eigen_gr)

                try:
                    diag_gr = sample_diagnostics.gelman_rubin(split_samples, self.varied_params, method='diag').max() - 1
                except ValueError:
                    diag_gr = np.nan
                toret &= full_test('diag_gr', 'max diag Gelman-Rubin - 1', diag_gr, min_diag_gr, max_diag_gr)

                def cl_lower(samples, params):
                    return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr)[0] for param in params])

                def cl_upper(samples, params):
                    return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr)[1] for param in params])

                try:
                    cl_diag_gr = np.max([sample_diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_lower, method='diag'),
                                         sample_diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_upper, method='diag')]) - 1
                except ValueError:
                    cl_diag_gr = np.nan
                toret &= full_test('cl_diag_gr', 'max diag Gelman-Rubin - 1 at {:.1f} sigmas'.format(nsigmas_cl_diag_gr), cl_diag_gr, min_cl_diag_gr, max_cl_diag_gr)

                try:
                    # Source: https://github.com/JohannesBuchner/autoemcee/blob/38feff48ae524280c8ea235def1f29e1649bb1b6/autoemcee.py#L337
                    all_geweke = sample_diagnostics.geweke(split_samples, self.varied_params, first=0.1, last=0.5)
                except ValueError:
                    all_geweke = np.nan
                geweke = np.max(all_geweke)
                toret &= full_test('geweke', 'max Geweke', geweke, min_geweke, max_geweke)

                from scipy import stats
                try:
                    geweke_pvalue = stats.normaltest(all_geweke, axis=None).pvalue
                except ValueError:
                    geweke_pvalue = np.nan
                toret &= full_test('geweke_pvalue', 'Geweke p-value', geweke_pvalue, min_geweke_pvalue, max_geweke_pvalue)

                split_samples = []
                for chain in self.chains:
                    chain = chain[burnin:]
                    chain = chain.reshape(len(chain), -1)
                    for iwalker in range(chain.shape[1]):
                        split_samples.append(chain[:, iwalker])
                try:
                    iact = sample_diagnostics.integrated_autocorrelation_time(split_samples, self.varied_params, check_valid='ignore')
                except ValueError:
                    iact = np.full(len(self.varied_params), np.nan, dtype='f8')
                add_diagnostics('iact', iact)
                niterations = len(split_samples[0])
                iact = iact.max()
                name = '({:d} iterations / integrated autocorrelation time)'.format(niterations)
                if reliable_iterations_over_iact * iact < niterations:
                    name = '{} (reliable)'.format(name)
                toret &= full_test('iterations_over_iact', name, niterations / iact, min_iterations_over_iact, max_iterations_over_iact)

                iact = diagnostics['iact']
                if len(iact) >= 2:
                    rel = np.abs(iact[-2] / iact[-1] - 1).max()
                    toret &= full_test('dact', 'max variation of integrated autocorrelation time', rel, min_dact, max_dact)

        diagnostics.update(self.mpicomm.bcast(diagnostics, root=0))

        return self.mpicomm.bcast(toret, root=0)
