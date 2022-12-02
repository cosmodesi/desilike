import os
import sys
import logging
import itertools

import numpy as np

from desilike.samples import Chain, Samples, load_source
from desilike import mpi, utils
from .base import BasePosteriorSampler


class PolychordSampler(BasePosteriorSampler):

    check = None

    def __init__(self, *args, blocks=None, oversample_power=0.4, nlive='25*ndim', nprior='10*nlive', nfail='1*nlive',
                 nrepeats='2*ndim', nlives=None, do_clustering=True, boost_posterior=0, compression_factor=np.exp(-1),
                 synchronous=True, seed=None, **kwargs):

        super(PolychordSampler, self).__init__(*args, seed=seed, **kwargs)
        logzero = np.nan_to_num(-np.inf)
        di = {'ndim': len(self.varied_params)}
        di['nlive'] = nlive = utils.evaluate(nlive, type=int, locals=di)
        nprior = utils.evaluate(nprior, type=int, locals=di)
        nfail = utils.evaluate(nfail, type=int, locals=di)
        feedback = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                    logging.INFO: 1, logging.DEBUG: 2}[logging.root.level]
        from .mcmc import _format_blocks
        if blocks is None:
            blocks, oversample_factors = self.pipeline.block_params(params=self.varied_params, oversample_power=oversample_power)
        else:
            blocks, oversample_factors = _format_blocks(blocks, self.varied_params)
        if np.any(oversample_factors > 1):
            if self.mpicomm.rank == 0:
                self.log_info('Oversampling with factors:')
                for s, b in zip(oversample_factors, blocks):
                    self.log_info('{:d}: {}'.format(s, b))
        self.varied_params.sort(itertools.chain(*blocks))
        grade_dims = [len(block) for block in blocks]
        grade_frac = [int(o * utils.evaluate(nrepeats, type=int, locals={'ndim': block_size}))
                      for o, block_size in zip(oversample_factors, grade_dims)]

        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save samples in polychord format\
                              alternatively one may update https://github.com/PolyChord/PolyChordLite\
                              to export samples directly as arrays')
        self.base_dirs = [os.path.dirname(fn) for fn in self.save_fn]
        self.file_roots = [os.path.splitext(os.path.basename(fn))[0] + '.polychord' for fn in self.save_fn]
        kwargs = {'nlive': nlive, 'nprior': nprior, 'nfail': nfail,
                  'do_clustering': do_clustering, 'feedback': feedback, 'precision_criterion': 1e-3,
                  'logzero': logzero, 'max_ndead': -1, 'boost_posterior': boost_posterior,
                  'posteriors': True, 'equals': True, 'cluster_posteriors': True,
                  'write_resume': True, 'read_resume': False, 'write_stats': False,
                  'write_live': True, 'write_dead': True, 'write_prior': True,
                  'maximise': False, 'compression_factor': compression_factor, 'synchronous': synchronous,
                  'grade_dims': grade_dims, 'grade_frac': grade_frac, 'nlives': nlives or {}}

        from pypolychord import settings
        self.settings = settings.PolyChordSettings(di['ndim'], 0, seed=(seed if seed is not None else -1), **kwargs)

    def _prepare(self):
        self.settings.read_resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)

    def _run_one(self, start, min_iterations=0, max_iterations=sys.maxsize, check=None, **kwargs):

        import pypolychord

        if check is not None and not isinstance(check, bool): kwargs.update(check)
        for name, value in kwargs.items():
            setattr(self.settings, name, value)
        self.settings.max_ndead = -1 if max_iterations == sys.maxsize else max_iterations

        def dumper(live, dead, logweights, logZ, logZerr):
            # Called only by rank = 0
            # When dumper() is called, save samples
            # BUT: we need derived parameters, which are on rank > 0 (if mpicomm.size > 1)
            # HACK: tell loglikelihood to save samples
            for rank in range(self.mpicomm.size):
                if _req.get(rank, None) is not None: _req[rank].Free()
                _req[rank] = self.mpicomm.isend((self._it_send, live) if rank == loglikelihood_rank else (self._it_send, None), dest=rank, tag=_tag)

        def my_dumper():
            # Called by rank > 0 if mpicomm.size > 1 else rank == 0
            if self.mpicomm.iprobe(source=0, tag=_tag):
                self._it_rec, live = self.mpicomm.recv(source=0, tag=_tag)
                if self.mpicomm.rank != loglikelihood_rank:
                    self.mpicomm.send(self.derived, dest=loglikelihood_rank, tag=_tag + 1)
                else:
                    derived = [self.derived] + [self.mpicomm.recv(source=rank, tag=_tag + 1) for rank in range(2, self.mpicomm.size)]
                    self.derived = [Samples.concatenate([dd[i] for dd in derived if dd is not None]) for i in range(2)]
                    try:
                        samples = list(np.loadtxt(prefix + '.txt', unpack=True))
                    except IOError:
                        pass
                    else:
                        nlive = len(live)
                        aweight, loglikelihood = [np.concatenate([sample, np.full(nlive, value, dtype='f8')]) for sample, value in zip(samples[:2], [0., np.nan])]
                        points = [np.concatenate([sample, live[:, iparam]]) for iparam, sample in enumerate(samples[2: 2 + ndim])]
                        loglikelihood[loglikelihood <= self.settings.logzero] = -np.inf
                        chain = Chain(points + [aweight, loglikelihood], params=self.varied_params + ['aweight', 'loglikelihood'])
                        if self.resume_derived is not None:
                            self.derived = [Samples.concatenate([resume_derived, derived]) for resume_derived, derived in zip(self.resume_derived, self.derived)]
                        chain = self._set_derived(chain)
                        self.resume_chain = chain[:-nlive]
                        self.resume_chain.save(self.save_fn[self._ichain])
                        chain[-nlive:].save(prefix + '.state.npy')
                self.resume_derived = self.derived
                self.derived = None

        def prior_transform(values):
            toret = np.empty_like(values)
            for iparam, (value, param) in enumerate(zip(values, self.varied_params)):
                try:
                    toret[iparam] = param.prior.ppf(value)
                except AttributeError as exc:
                    raise AttributeError('{} has no attribute ppf (maybe infinite prior?). Choose proper prior for nested sampling'.format(param.prior)) from exc
            return toret

        def loglikelihood(values):
            # Called by ranks > 0
            my_dumper()
            return (max(self.loglikelihood(values), self.settings.logzero), [])

        self.pipeline.mpicomm = mpi.COMM_SELF
        loglikelihood_rank = 0 if self.mpicomm.size == 1 else 1
        _tag, _req = 1000, {}
        self._it_send, self._it_rec = 0, 0

        #if self.mpicomm.size > 1:
        #    raise ValueError('Cannot run polychord on multiple processes; one should implement a callback function called by processes in polychord')
        self.settings.base_dir = self.base_dirs[self._ichain]
        self.settings.file_root = self.file_roots[self._ichain]
        prefix = os.path.join(self.settings.base_dir, self.settings.file_root)

        self.resume_derived, self.resume_chain = None, None
        if self.settings.read_resume:
            if self.mpicomm.rank == loglikelihood_rank:
                source = load_source([self.save_fn[self._ichain], prefix + '.state.npy'])
                source = source[0].concatenate(source)
                self.resume_derived = [source.select(name=self.varied_params.names()), source.select(derived=True)]

        ndim = len(self.varied_params)
        kwargs = {}
        if self.mpicomm is not mpi.COMM_WORLD:
            kwargs['comm'] = self.mpicomm
        try:
            pypolychord.run_polychord(loglikelihood, ndim, 0, self.settings,
                                      prior=prior_transform, dumper=dumper,
                                      **kwargs)
        except TypeError as exc:
            raise ImportError('To use polychord in parallel, please use version at https://github.com/adematti/PolyChordLite@mpi4py') from exc

        # Final dump, in case we did not write the last points
        self._it_send = self.mpicomm.bcast(self._it_send, root=0)
        self._it_rec = self.mpicomm.bcast(self._it_rec, root=loglikelihood_rank)
        while self._it_rec != self._it_send:
            my_dumper()
            self._it_rec = self.mpicomm.bcast(self._it_rec, root=loglikelihood_rank)

        self.derived = self.resume_derived
        if self.derived is None: self.derived = [None] * 2
        self.derived = [Samples.sendrecv(derived, source=loglikelihood_rank, dest=0, mpicomm=self.mpicomm) for derived in self.derived]
        chain = Samples.sendrecv(self.resume_chain, source=loglikelihood_rank, dest=0, mpicomm=self.mpicomm)
        return chain

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/adematti/PolyChordLite@mpi4py')
