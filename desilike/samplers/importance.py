import logging
import warnings

import numpy as np

from desilike.utils import BaseClass, is_path, TaskManager
from desilike.samples import Chain, load_source
from .base import BasePosteriorSampler


class ImportanceSampler(BaseClass):

    name = 'importance'

    def __init__(self, likelihood, chains, save_fn=None, mpicomm=None):
        """
        Importance sample input chains, adding the corresponding weight to :attr:`Chain.aweight`.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        chains : str, Path, Chain
            Path to or chains to importance sample.

        save_fn : str, Path, default=None
            If not ``None``, save chains to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`.
        """
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = likelihood
        #self.pipeline = self.likelihood.runtime_info.pipeline
        self.mpicomm = mpicomm
        self.likelihood.solved_default = '.marg'
        self.varied_params = self.likelihood.varied_params.deepcopy()
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        if self.mpicomm.rank == 0:
            self.input_chains = load_source(chains)
        nchains = self.mpicomm.bcast(len(self.input_chains) if self.mpicomm.rank == 0 else None, root=0)
        if self.mpicomm.rank != 0:
            self.input_chains = [None] * nchains
        self.save_fn = save_fn
        if save_fn is not None:
            if is_path(save_fn):
                self.save_fn = [str(save_fn).replace('*', '{}').format(i) for i in range(self.nchains)]
            else:
                if len(save_fn) != self.nchains:
                    raise ValueError('Provide {:d} chain file names'.format(self.nchains))

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm

    @property
    def nchains(self):
        return len(self.input_chains)

    def run(self, subtract_input=False):
        r"""
        Run importance sampling.

        Parameters
        ----------
        subtract_input : bool, default=False
            If ``True``, :attr:`Chain.aweight` is divided by :math:`e^{\mathcal{L} - \mathcal{L}_\mathrm{max}}`,
            with :math:`\mathcal{L}` the log-posterior.

        """
        if getattr(self, '_vlikelihood', None) is None:
            self._set_vlikelihood()

        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)
        chains = [None] * self.nchains
        self.chains = [None] * self.nchains
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            worker_ranks = getattr(tm, 'self_worker_ranks', [])
            for dest in self.mpicomm.allgather(worker_ranks[0] if worker_ranks else 0):
                for ichain in range(self.nchains):
                    chain = Chain.sendrecv(self.input_chains[ichain], source=0, dest=dest, mpicomm=self.mpicomm)
                    if chain is not None: self.input_chains[ichain] = chain

            self.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                if self.mpicomm.rank == 0:
                    chain = self.input_chains[ichain].deepcopy()
                    if subtract_input:
                        logposterior = chain.logposterior
                        max_logposterior = 0.
                        mask = np.isfinite(logposterior)
                        if mask.any(): max_logposterior = logposterior[mask].max()
                        chain.aweight[...] /= np.exp(logposterior - max_logposterior)
                    points = chain.to_dict(params=self.varied_params)

                results = self._vlikelihood(points if self.mpicomm.rank == 0 else {})

                raise_error = None
                if self.mpicomm.rank == 0:
                    (logposterior, derived), errors = results
                    for param in self.likelihood.all_params.select(fixed=True, derived=False):
                        chain[param] = np.full(chain.shape, param.value, dtype='f8')
                    chain.update(derived)
                    if errors:
                        for ipoint, error in errors.items():
                            if isinstance(error[0], self.likelihood.catch_errors):
                                self.log_debug('Error "{}" raised with parameters {} is caught up with -inf loglikelihood. Full stack trace\n{}:'.format(repr(error[0]), {k: v.flat[ipoint] for k, v in points.items()}, error[1]))
                                for param in [self.likelihood._param_loglikelihood, self.likelihood._param_logprior]:
                                    if param in chain:
                                        chain[param][ipoint, ...] = -np.inf
                            else:
                                raise_error = error
                            if raise_error is None and not self.logger.isEnabledFor(logging.DEBUG):
                                warnings.warn('Error "{}" raised is caught up with -inf loglikelihood. Set logging level to debug (setup_logging("debug")) to get full stack trace.'.format(repr(error[0])))
                    chains[ichain] = chain
        self.mpicomm = mpicomm_bak

        for ichain, chain in enumerate(chains):
            mpiroot_worker = self.mpicomm.rank if chains[ichain] is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            if self.mpicomm.bcast(chain is not None, root=mpiroot_worker):
                chains[ichain] = Chain.sendrecv(chain, source=mpiroot_worker, dest=0, mpicomm=self.mpicomm)

        if self.mpicomm.rank == 0:
            for ichain, chain in enumerate(chains):
                self.chains[ichain] = chain
                if chain is not None:
                    for param in [self.likelihood._param_loglikelihood, self.likelihood._param_logprior]:
                        mask = np.isnan(chain[param])
                        chain[param][mask] = -np.inf
                    logposterior = chain[self.likelihood._param_loglikelihood][()] + chain[self.likelihood._param_logprior][()]
                    max_logposterior = 0.
                    mask = np.isfinite(logposterior)
                    if mask.any(): max_logposterior = logposterior[mask].max()
                    chain.aweight[...] *= np.exp(logposterior - max_logposterior)
                    self.log_info('Importance weight range {} - {}.'.format(chain.aweight.min(), chain.aweight.max()))
                    for name in ['size', 'nvaried', 'ndof']:
                        try:
                            value = getattr(self.likelihood, name)
                        except AttributeError:
                            pass
                        else:
                            chain.attrs[name] = value
            if self.save_fn is not None:
                for ichain, chain in enumerate(self.chains):
                    if chain is not None: chain.save(self.save_fn[ichain])
        return self.chains

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


ImportanceSampler._set_vlikelihood = BasePosteriorSampler._set_vlikelihood