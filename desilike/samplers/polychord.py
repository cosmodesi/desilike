import os
import sys
import logging
import itertools

import numpy as np

from desilike.samples import Chain, Samples, load_source
from desilike import mpi, utils
from .base import BasePosteriorSampler


class PolychordSampler(BasePosteriorSampler):

    """
    Wrapper for polychord nested sampler.

    Reference
    ---------
    - https://github.com/PolyChord/PolyChordLite
    - https://arxiv.org/abs/1502.01856
    - https://arxiv.org/abs/1506.00171
    """
    check = None

    def __init__(self, *args, blocks=None, oversample_power=0.4, nlive='25*ndim', nprior='10*nlive', nfail='1*nlive',
                 nrepeats='2*ndim', nlives=None, do_clustering=True, boost_posterior=0, compression_factor=np.exp(-1),
                 synchronous=True, seed=None, **kwargs):
        """
        Initialize polychord sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        blocks : list, default=None
            Parameter blocks are groups of parameters which are updated alltogether
            with a frequency proportional to oversample_factor.
            Typically, parameter blocks are chosen such that parameters in a given block
            require the same evaluation time of the likelihood when updated.
            If ``None`` these blocks are defined at runtime, based on (measured) speeds and oversample_power (below),
            but can be specified there in the format:

                - [oversample_factor1, [param1, param2]]
                - [oversample_factor2, [param3, param4]]

        oversample_power : float, default=0.4
            If ``blocks`` is ``None``, i.e. parameter blocks are defined at runtime,
            oversample factors are ``speed**oversample_power``.

        nlive : int, str, default='25 * ndim'
            Number of live points. Increasing nlive increases the accuracy of posteriors and evidences,
            and proportionally increases runtime ~ O(nlive).

        nprior : int, str, default='10*nlive'
            The number of prior samples to draw before starting compression.

        nfail : int, str, default='1*nlive'
            The number of failed spawns before stopping nested sampling.

        nrepeats : int, str, default='2*ndim'
            The number of slice slice-sampling steps to generate a new point.
            Increasing nrepeats increases the reliability of the algorithm.
            Typically:

                - for reliable evidences need nrepeats ~ O(5*ndims)
                - for reliable posteriors need nrepeats ~ O(ndims)

        nlives : dict, default=None
            Variable number of live points option. This dictionary is a mapping
            between loglikelihood contours and ``nlive``.
            You should still set nlive to be a sensible number, as this indicates
            how often to update the clustering, and to define the default value.

        do_clustering : bool, default=True
            Whether or not to explore multi-modality on the posterior.

        boost_posterior : int, default=0
            Increase the number of posterior samples produced. This can be set
            arbitrarily high, but you will not be able to boost by more than nrepeats.
            Warning: in high dimensions PolyChord produces *a lot* of posterior samples.
            You probably do not need to change this.

        synchronous : bool, default=True
            Parallelise with synchronous workers, rather than asynchronous ones.
            This can be set to ``False`` if the likelihood speed is known to be
            approximately constant across the parameter space. Synchronous
            parallelisation is less effective than asynchronous by a factor ~O(1)
            for large parallelisation.

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
                  'write_resume': True, 'read_resume': False, 'write_stats': True,
                  'write_live': True, 'write_dead': True, 'write_prior': True,
                  'maximise': False, 'compression_factor': compression_factor, 'synchronous': synchronous,
                  'grade_dims': grade_dims, 'grade_frac': grade_frac, 'nlives': nlives or {}}

        from pypolychord import settings
        self.settings = settings.PolyChordSettings(di['ndim'], 0, seed=(seed if seed is not None else -1), **kwargs)

    def _prepare(self):
        self.settings.read_resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)

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

        compression_factor : float, default=np.exp(-1)
            How often to update the files and do clustering.

        check : bool, dict, default=None
            If ``False``, no convergence checks are run.
            If ``True`` or ``None``, convergence checks are run.
            A dictionary of convergence criteria can be provided, with:

                - precision_criterion : Nested sampling terminates when the evidence contained in the live points
                  is precision_criterion fraction of the total evidence. Default is 0.001.

        """
        return super(PolychordSampler, self).run(*args, **kwargs)

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
            self._it_send += 1
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
                        #aweight, loglikelihood = [np.concatenate([sample, np.full(nlive, value, dtype='f8')]) for sample, value in zip(samples[:2], [0., np.nan])]
                        aweight = np.concatenate([samples[0], np.zeros(nlive, dtype='f8')])
                        points = [np.concatenate([sample, live[:, iparam]]) for iparam, sample in enumerate(samples[2: 2 + ndim])]
                        #loglikelihood[loglikelihood <= self.settings.logzero] = -np.inf
                        #chain = Chain(points + [aweight, loglikelihood], params=self.varied_params + ['aweight', 'loglikelihood'])
                        chain = Chain(points + [aweight], params=self.varied_params + ['aweight'])
                        if self.resume_derived is not None:
                            self.derived = [Samples.concatenate([resume_derived, derived], intersection=True) for resume_derived, derived in zip(self.resume_derived, self.derived)]
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
            return (max(self.logposterior(values) - self.logprior(values), self.settings.logzero), [])

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
                self.resume_derived = [source] * 2

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
        # if self.mpicomm.bcast(self.derived[0] is not None, root=loglikelihood_rank):
        self.derived = [Samples.sendrecv(derived, source=loglikelihood_rank, dest=0, mpicomm=self.mpicomm) for derived in self.derived]
        chain = Samples.sendrecv(self.resume_chain, source=loglikelihood_rank, dest=0, mpicomm=self.mpicomm)
        return chain

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/adematti/PolyChordLite@mpi4py')
