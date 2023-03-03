import numpy as np

from desilike.utils import BaseClass, path_types, TaskManager
from desilike.samples import Chain, load_source


class ImportanceSampler(BaseClass):

    name = 'base'
    nwalkers = 1
    _check_same_input = False

    def __init__(self, likelihood, chains, save_fn=None, mpicomm=None):
        """
        Importance sample input chains, adding the corresponding weight to :attr:`Chain.aweight`.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        input_chains : str, Path, Chain
            Path to or chains to importance sample.

        save_fn : str, Path, default=None
            If not ``None``, save chains to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`.
        """
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = likelihood
        self.pipeline = self.likelihood.runtime_info.pipeline
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
            if isinstance(save_fn, path_types):
                self.save_fn = [str(save_fn).replace('*', '{}').format(i) for i in range(self.nchains)]
            else:
                if len(save_fn) != self.nchains:
                    raise ValueError('Provide {:d} chain file names'.format(self.nchains))

    @property
    def mpicomm(self):
        return self.pipeline.mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self.pipeline.mpicomm = mpicomm

    @property
    def nchains(self):
        return len(self.input_chains)

    def run(self):
        """Run importance sampling."""
        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)
        chains = [None] * self.nchains
        self.chains = [None] * self.nchains
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                chain = self.input_chains[ichain].deepcopy()
                self.pipeline.mpicalculate(**(chain.to_dict(params=self.varied_params) if self.mpicomm.rank == 0 else {}))
                if self.mpicomm.rank == 0:
                    if chain is not None:
                        for param in self.pipeline.params.select(fixed=True, derived=False):
                            chain[param] = np.full(chain.shape, param.value, dtype='f8')
                        chain.update(self.pipeline.derived)
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
                        chain.aweight[...] *= np.exp(chain.logposterior - chain.logposterior.max())
                    for name in ['size', 'nvaried', 'ndof']:
                        try:
                            value = getattr(self.likelihood, name)
                        except AttributeError:
                            pass
                        else:
                            self.chains[ichain].attrs[name] = value
            if self.save_fn is not None:
                for ichain, chain in enumerate(self.chains):
                    if chain is not None: chain.save(self.save_fn[ichain])
        return self.chains

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass