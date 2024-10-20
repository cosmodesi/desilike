import time
import itertools
import functools

import numpy as np
from numpy import linalg
from numpy.linalg import LinAlgError
from scipy.stats import special_ortho_group

from desilike import utils
from desilike.samples import Chain, load_source
from .base import BaseBatchPosteriorSampler


class State(object):

    _attrs = ['coords', 'log_prob', 'weight']

    def __init__(self, *args, **kwargs):
        attrs = {k: v for k, v in zip(self._attrs, args)}
        attrs.update(kwargs)
        self.__dict__.update(attrs)


class MHSampler(object):

    """Metropolis-Hasting MCMC algorithm, with dragging option, as in cobaya, following emcee interface."""

    def __init__(self, ndim, log_prob_fn, propose, nsteps_drag=None, max_tries=1000, vectorize=1, rng=None):
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.propose = propose
        self.nsteps_drag = nsteps_drag
        if nsteps_drag is not None:
            self.nsteps_drag = int(nsteps_drag)
            if self.nsteps_drag < 2:
                raise ValueError('With dragging nsteps_drag must be >= 2')
            if len(self.propose) != 2:
                raise ValueError('With dragging give list of two propose methods, slow and fast')
        self.max_tries = int(max_tries)
        self.rng = rng or np.random.RandomState()
        self.vectorize = int(vectorize)
        self.states = []

    def sample(self, start, iterations=1, thin_by=1):
        self.state = State(start, self.log_prob_fn(start), 1)
        for iter in range(iterations + 1):  # we skip start
            accept = False
            for itry in range(self.max_tries):
                if self.nsteps_drag:  # dragging
                    current_coords_start = np.repeat(self.state.coords[None, :], self.vectorize, axis=0)
                    current_log_prob_start = np.repeat(self.state.log_prob, self.vectorize, axis=0)
                    current_coords_end = current_coords_start + self.propose[0](self.vectorize)  # slow
                    current_log_prob_end = self.log_prob_fn(current_coords_end)
                    mask_current = current_log_prob_end > -np.inf
                    if not mask_current.any():
                        self.state.weight += mask_current.size
                        continue
                    sum_log_prob_start = current_log_prob_start.copy()
                    sum_log_prob_end = current_log_prob_end.copy()
                    naverage = 1 + self.nsteps_drag
                    for istep in range(1, naverage):
                        proposal_coords_start = current_coords_start + self.propose[1](self.vectorize)  # fast
                        proposal_log_prob_start = self.log_prob_fn(proposal_coords_start)
                        mask_proposal = mask_current & (proposal_log_prob_start > -np.inf)
                        if mask_proposal.any():
                            proposal_coords_end = current_coords_end + self.propose[1](self.vectorize)  # fast
                            proposal_log_prob_end = self.log_prob_fn(proposal_coords_end)
                            mask_proposal &= proposal_log_prob_end > -np.inf
                            if mask_proposal.any():
                                # Create the interpolated probability and perform a Metropolis test
                                frac = istep / naverage
                                proposal_log_prob_interp = (1 - frac) * proposal_log_prob_start + frac * proposal_log_prob_end
                                current_log_prob_interp = (1 - frac) * current_log_prob_start + frac * current_log_prob_end
                                mask_accept = mask_proposal & np.array([self._mh_accept(plogprob, clogprob) for plogprob, clogprob in zip(proposal_log_prob_interp, current_log_prob_interp)])
                                # The dragging step was accepted, do the drag
                                current_coords_start[mask_accept] = proposal_coords_start[mask_accept]
                                current_log_prob_start[mask_accept] = proposal_log_prob_start[mask_accept]
                                current_coords_end[mask_accept] = proposal_coords_end[mask_accept]
                                current_log_prob_end[mask_accept] = proposal_log_prob_end[mask_accept]
                        sum_log_prob_start += current_log_prob_start
                        sum_log_prob_end += current_log_prob_end
                    mh_proposal_log_prob, mh_current_log_prob = sum_log_prob_end / naverage, sum_log_prob_start / naverage
                    proposal_coords, proposal_log_prob = current_coords_end, current_log_prob_end
                else:  # standard MH
                    proposal_coords = self.state.coords + self.propose(size=self.vectorize)
                    mh_current_log_prob = np.full(self.vectorize, self.state.log_prob, dtype='f8')
                    #mh_proposal_log_prob = proposal_log_prob = np.full(self.vectorize, -np.inf, dtype='f8')
                    mh_proposal_log_prob = proposal_log_prob = self.log_prob_fn(proposal_coords)
                for i in range(self.vectorize):
                    accept = self._mh_accept(mh_proposal_log_prob[i], mh_current_log_prob[i])
                    if accept:
                        break
                    else:
                        self.state.weight += 1
                if accept:
                    if iter > 0 and iter % thin_by == 0:
                        self.states.append(self.state)
                    self.state = State(proposal_coords[i], proposal_log_prob[i], 1)
                    #print(proposal_log_prob, accept)
                    #print(self.mpicomm.rank, iter, itry, flush=True)
                    break
            if not accept:
                raise ValueError('Could not find finite log posterior after {:d} tries'.format(self.max_tries))
            yield self.state

    def _mh_accept(self, proposal_log_prob, current_log_prob):
        if proposal_log_prob == -np.inf:
            return False
        if proposal_log_prob > current_log_prob:
            return True
        return self.rng.standard_exponential() > (current_log_prob - proposal_log_prob)

    def get_chain(self):
        return np.array([state.coords for state in self.states])

    def get_weight(self):
        return np.array([state.weight for state in self.states], dtype='i8')

    def get_log_prob(self):
        return np.array([state.log_prob for state in self.states])

    def get_acceptance_rate(self):
        return len(self.states) / self.get_weight().sum()

    def reset(self):
        self.states = []


class IndexCycler(object):

    def __init__(self, ndim, rng):
        self.ndim = ndim
        self.loop_index = -1
        self.rng = rng


class CyclicIndexRandomizer(IndexCycler):

    def __init__(self, ndim, rng):
        if np.ndim(ndim) == 0:
            self.sorted_indices = list(range(ndim))
        else:
            self.sorted_indices = ndim
            ndim = len(ndim)
        super(CyclicIndexRandomizer, self).__init__(ndim, rng)
        if self.ndim <= 2:
            self.indices = list(range(ndim))

    def next(self):
        """Get the next random index, or alternate for two or less."""
        self.loop_index = (self.loop_index + 1) % self.ndim
        if self.loop_index == 0 and self.ndim > 2:
            self.indices = self.rng.permutation(self.sorted_indices)
        return self.indices[self.loop_index]


class SOSampler(IndexCycler):

    def __call__(self):
        return self.sample()

    def sample(self):
        """Propose a random n-dimension vector."""
        if self.ndim == 1:
            return np.array([self.rng.choice([-1, 1]) * self.sample_r()])
        self.loop_index = (self.loop_index + 1) % self.ndim
        if self.loop_index == 0:
            self.rotmat = special_ortho_group.rvs(self.ndim, random_state=self.rng)
        # print(np.sum(self.rotmat[:, self.loop_index]**2)**0.5, self.sample_r())
        return self.rotmat[:, self.loop_index] * self.sample_r()

    def sample_r(self):
        """
        Radial proposal. A mixture of an exponential and 2D Gaussian radial proposal
        (to make wider tails and more mass near zero, so more robust to scale misestimation).
        """
        if self.rng.uniform() < 0.33:
            return self.rng.standard_exponential()
        return np.sqrt(self.rng.chisquare(min(self.ndim, 2)))


def vectorize(func):

    @functools.wraps(func)
    def wrapper(self, size=None, **kwargs):
        if size is None:
            return func(self, **kwargs)
        shape = size
        if np.ndim(size) == 0:
            shape = (size, )
        size = np.prod(size)
        tmp = [func(self, **kwargs) for i in range(size)]
        return np.array(tmp).reshape(shape + tmp[0].shape)

    return wrapper


class BlockProposer(object):

    def __init__(self, blocks, oversample_factors=None,
                 last_slow_block_index=None, proposal_scale=2.4, rng=None):
        """
        Proposal density for fast and slow parameters, where parameters are
        grouped into blocks which are changed at the same time.

        Parameters
        ----------
        blocks : array
            Number of parameters in each block, with blocks sorted by ascending speed.

        oversample_factors : list, default=None
            List of *int* oversampling factors *per parameter*,
            i.e. a factor of n for a block of dimension d would mean n*d jumps for that
            block per full cycle, whereas a factor of 1 for all blocks (default) means
            that all *directions* are treated equally (but the proposals are still
            block-wise).

        last_slow_block_index : int, default=None
            Index of the last block considered slow.
            By default, all blocks are considered slow.

        proposal_scale : float, default=2.4
            Overall scale for the proposal.

        rng : np.random.RandomState, default=None
            Random state.
        """
        self.rng = rng or np.random.RandomState()
        self.proposal_scale = float(proposal_scale)
        self.blocks = np.array(blocks, dtype='i4')
        if np.any(blocks != self.blocks):
            raise ValueError('blocks must be integer! Got {}.'.format(blocks))
        if oversample_factors is None:
            self.oversample_factors = np.ones(len(blocks), dtype='i4')
        else:
            if len(oversample_factors) != len(self.blocks):
                raise ValueError('List of oversample_factors has a different length than list of blocks: {:d} vs {:d}'.format(len(oversample_factors), len(self.blocks)))
            self.oversample_factors = np.array(oversample_factors, dtype='i4')
            if np.any(oversample_factors != self.oversample_factors):
                raise ValueError('oversample_factors must be integer! Got {}.'.format(oversample_factors))
        # Binary fast-slow split
        self.last_slow_block_index = last_slow_block_index
        if self.last_slow_block_index is None:
            self.last_slow_block_index = len(blocks) - 1
        else:
            if self.last_slow_block_index > len(blocks) - 1:
                raise ValueError('The index given for the last slow block, {:d}, is not valid: there are only {:d} blocks'.format(self.last_slow_block_index, len(self.blocks)))
        n_all = sum(self.blocks)
        n_slow = sum(self.blocks[:1 + self.last_slow_block_index])
        self.nsamples_slow = self.nsamples_fast = 0
        # Starting index of each block
        self.block_starts = np.insert(np.cumsum(self.blocks), 0, 0)
        # Prepare indices for the cycler, repeated if there is oversampling
        indices_repeated = np.concatenate([np.repeat(np.arange(b) + s, o) for b, s, o in zip(self.blocks, self.block_starts, self.oversample_factors)])
        self.param_block_indices = np.concatenate([np.full(b, ib, dtype='i4') for ib, b in enumerate(self.blocks)])
        # Creating the blocked proposers
        self.proposer = [SOSampler(b, self.rng) for b in self.blocks]
        # Parameter cyclers, cycling over the j's
        self.param_cycler = CyclicIndexRandomizer(indices_repeated, self.rng)
        # These ones are used by fast dragging only
        self.param_cycler_slow = CyclicIndexRandomizer(n_slow, self.rng)
        self.param_cycler_fast = CyclicIndexRandomizer(n_all - n_slow, self.rng)

    @property
    def ndim(self):
        return len(self.param_block_indices)

    @vectorize
    def __call__(self, params=None):
        current_iblock = self.param_block_indices[self.param_cycler.next()]
        if current_iblock <= self.last_slow_block_index:
            self.nsamples_slow += 1
        else:
            self.nsamples_fast += 1
        return self._get_block_proposal(current_iblock, params=params)

    @vectorize
    def slow(self, params=None):
        current_iblock_slow = self.param_block_indices[self.param_cycler_slow.next()]
        self.nsamples_slow += 1
        return self._get_block_proposal(current_iblock_slow, params=params)

    @vectorize
    def fast(self, params=None):
        current_iblock_fast = self.param_block_indices[self.param_cycler_slow.ndim + self.param_cycler_fast.next()]
        self.nsamples_fast += 1
        return self._get_block_proposal(current_iblock_fast, params=params)

    def _get_block_proposal(self, iblock, params=None):
        if params is None:
            params = np.zeros(self.ndim, dtype='f8')
        else:
            params = np.array(params)
        params[self.block_starts[iblock]:] += self.transform[iblock].dot(self.proposer[iblock]() * self.proposal_scale)
        return params

    def set_covariance(self, matrix):
        """
        Take covariance of sampled parameters (matrix), and construct orthonormal
        parameters where orthonormal parameters are grouped in blocks by speed, so changes
        in the slowest block changes slow and fast parameters, but changes in the fastest
        block only changes fast parameters.
        """
        matrix = np.array(matrix)
        if matrix.shape[0] != self.ndim:
            raise ValueError('The covariance matrix does not have the correct dimension: '
                             'it is {:d}, but it should be {:d}.'.format(matrix.shape[0], self.ndim))
        if not (np.allclose(matrix.T, matrix) and np.all(np.linalg.eigvals(matrix) > 0)):
            raise linalg.LinAlgError('The given covmat is not a positive-definite, symmetric square matrix.')
        L = linalg.cholesky(matrix)
        # Store the basis as transformation matrices
        self.transform = []
        for block_start, bp in zip(self.block_starts, self.proposer):
            block_end = block_start + bp.ndim
            self.transform += [L[block_start:, block_start:block_end]]
        return True


def _format_blocks(blocks, params):
    blocks, oversample_factors = [b[1] for b in blocks], [b[0] for b in blocks]
    blocks = [[params[name] for name in block if name in params] for block in blocks]
    blocks, oversample_factors = [b for b in blocks if b], [s for s, b in zip(oversample_factors, blocks) if b]
    params_in_blocks = set(itertools.chain(*blocks))
    if params_in_blocks != set(params):
        raise ValueError('Missing sampled parameters in provided blocks: {}'.format(set(params) - params_in_blocks))
    argsort = np.argsort(oversample_factors)
    return [blocks[i] for i in argsort], np.array([oversample_factors[i] for i in argsort])


class MCMCSampler(BaseBatchPosteriorSampler):
    """
    Antony Lewis CosmoMC blocked fast-slow Metropolis sampler, wrapped for cobaya by Jesus Torrado.
    FIXME: REMOVE (ADDED BACK FOR URGENCY).

    Reference
    ---------
    - https://github.com/CobayaSampler/cobaya/tree/master/cobaya/samplers/mcmc
    - https://arxiv.org/abs/astro-ph/0205436
    - https://arxiv.org/abs/1304.4473
    - https://arxiv.org/abs/math/0502099
    """
    def __init__(self, *args, blocks=None, oversample_power=0.4, covariance=None, proposal_scale=2.4, learn=True, drag=False, **kwargs):
        """
        Initialize MCMC sampler.

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
            oversample factors are ~ ``speed**oversample_power``.

        covariance : str, dict, Chain, Profiles, ParameterCovariance, default=None
            (Initial) proposal covariance, to draw parameter jumps.
            Can be previous samples e.g. ``({fn: chain.npy, burnin: 0.5})``,
            or profiles (containing parameter covariance matrix), or parameter covariance.
            If variance for a given parameter is not provided, parameter's attr:`Parameter.proposal` squared is used.

        proposal_scale : float, default=2.4
            Scale proposal by this value when drawing jumps.

        learn : bool, default=True
            If ``True``, learn proposal covariance matrix.
            Can be a dictionary, specifying when to update covariance matrix, with same options as ``check``,
            e.g. to check every 40 * dimension steps, and update proposal when Gelman-Rubin is between 0.03 and 0.1: ``{'every': '40 * ndim', 'max_eigen_gr': 0.1, 'min_eigen_gr': 0.03}``.

        drag : bool, default=False
            Use dragging ("integrating out" fast parameters).

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
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`
        """
        super(MCMCSampler, self).__init__(*args, **kwargs)
        if blocks is None:
            blocks, oversample_factors = self.likelihood.runtime_info.pipeline.block_params(params=self.varied_params, nblocks=2 if drag else None, oversample_power=oversample_power)
        else:
            blocks, oversample_factors = _format_blocks(blocks, self.varied_params)
        blocks = [[str(bb) for bb in block] for block in blocks]
        last_slow_block_index = nsteps_drag = None
        if drag:
            if len(blocks) == 1:
                drag = False
                if self.mpicomm.rank == 0:
                    self.log_warning('Dragging disabled: not possible if there is only one block.')
            if max(oversample_factors) / min(oversample_factors) < 2:
                drag = False
                if self.mpicomm.rank == 0:
                    self.log_warning('Dragging disabled: speed ratios between blocks < 2.')
            if drag:
                for first_fast_block_index, speed in enumerate(oversample_factors):
                    if speed != 1: break
                last_slow_block_index = first_fast_block_index - 1
                n_slow = sum(len(b) for b in blocks[:first_fast_block_index])
                n_fast = len(self.varied_params) - n_slow
                nsteps_drag = int(oversample_factors[first_fast_block_index] * n_fast / n_slow + 0.5)
                if self.mpicomm.rank == 0:
                    self.log_info('Dragging:')
                    self.log_info('1 step: {}'.format(blocks[:first_fast_block_index]))
                    self.log_info('{:d} steps: {}'.format(nsteps_drag, blocks[first_fast_block_index:]))
        elif np.any(oversample_factors > 1):
            if self.mpicomm.rank == 0:
                self.log_info('Oversampling with factors:')
                for s, b in zip(oversample_factors, blocks):
                    self.log_info('{:d}: {}'.format(s, b))

        self.varied_params = self.varied_params.sort(itertools.chain(*blocks))
        self.proposer = BlockProposer(blocks=[len(b) for b in blocks], oversample_factors=oversample_factors, last_slow_block_index=last_slow_block_index, proposal_scale=proposal_scale, rng=self.rng)
        self.learn = bool(learn)
        self.learn_check = None
        burnin = 0.5
        if isinstance(learn, dict):
            self.learn = True
            self.learn_check = dict(learn)
            burnin = self.learn_check['burnin'] = self.learn_check.get('burnin', burnin)
        if isinstance(covariance, dict):
            covariance['burnin'] = covariance.get('burnin', burnin)
        else:
            covariance = {'source': covariance, 'burnin': burnin}
        if self.mpicomm.rank == 0:
            covariance = load_source(**covariance, cov=True, params=self.varied_params, return_type='nparray')
            self.log_info('Using provided covariance matrix.')
        covariance = self.mpicomm.bcast(covariance, root=0)
        self.proposer.set_covariance(covariance)
        self.learn_diagnostics = {}
        propose = [self.proposer.slow, self.proposer.fast] if drag else self.proposer
        self.sampler = MHSampler(len(self.varied_params), self.logposterior, propose=propose, nsteps_drag=nsteps_drag, max_tries=self.max_tries, rng=self.rng)
        self._size_every = self.mpicomm.bcast(sum(getattr(chain, 'size', 0) for chain in self.chains) if self.mpicomm.rank == 0 else None, root=0)

    def _set_rng(self, *args, **kwargs):
        super(MCMCSampler, self)._set_rng(*args, **kwargs)
        for name in ['proposer', 'sampler']:
            if hasattr(self, name):
                getattr(self, name).rng = self.rng

    def _prepare(self):
        covariance = None
        if self.learn and self.mpicomm.bcast(all(chain is not None for chain in self.chains), root=0):
            learn = self.learn_check is None
            burnin = 0.5
            if not learn:
                every = self.learn_check.get('every', None)
                if every is not None:
                    every = utils.evaluate(every, type=int, locals={'ndim': len(self.varied_params)})
                    size = self.mpicomm.bcast(sum(chain.size for chain in self.chains) if self.mpicomm.rank == 0 else None, root=0)
                    if size - self._size_every < every: return
                    self._size_every = size
                burnin = self.learn_check['burnin']
                learn = self.check(**self.learn_check, diagnostics=self.learn_diagnostics, quiet=True)[0]
            if learn and self.mpicomm.rank == 0:
                chain = Chain.concatenate([chain.remove_burnin(burnin) for chain in self.chains])
                if chain.size > 1:
                    covariance = chain.covariance(params=self.varied_params)
            covariance = self.mpicomm.bcast(covariance, root=0)
            if covariance is not None:
                try:
                    self.proposer.set_covariance(covariance)
                    # print({param.name: cov**0.5 for param, cov in zip(self.varied_params, np.diag(covariance))})
                except LinAlgError:
                    if self.mpicomm.rank == 0:
                        self.log_info('New proposal covariance is ill-conditioned, skipping update.')
                else:
                    if self.mpicomm.rank == 0:
                        self.log_info('Updating proposal covariance.')
            elif self.mpicomm.rank == 0:
                self.log_info('Skipping update of proposal covariance, as criteria are not met.')

    def run(self, *args, **kwargs):
        """
        Run chains. Sampling can be interrupted anytime, and resumed by providing the path to the saved chains in ``chains`` argument of :meth:`__init__`.

        One will typically run sampling on ``nchains * nprocs_per_chain`` processes,
        with ``nchains >= 1`` the number of chains and ``nprocs_per_chain = max(mpicomm.size // nchains, 1)``
        the number of processes per chain.

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

        thin_by : int, default=1
            Thin samples by this factor.
        """
        return super(MCMCSampler, self).run(*args, **kwargs)

    def _run_one(self, start, niterations=300, thin_by=1):
        self.sampler.reset()
        self.sampler.vectorize = self.mpicomm.size
        self.sampler.mpicomm = self.mpicomm
        if thin_by == 'auto':
            thin_by = int(sum(b * s for b, s in zip(self.proposer.blocks, self.proposer.oversample_factors)) / len(self.varied_params))
        #log_every = max(niterations // 5, 50)
        log_every = 30  # 30 seconds
        t0 = time.time()
        for _ in self.sampler.sample(start=np.ravel(start), iterations=niterations, thin_by=thin_by):
            if self.mpicomm.rank == 0 and time.time() - t0 >= log_every:
                total = int(self.sampler.get_weight().sum())
                self.log_info('{:d} steps, acceptance rate {:.3f}.'.format(total, self.sampler.get_acceptance_rate()))
                t0 = time.time()
        chain = self.sampler.get_chain()
        if chain.size:
            data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_weight(), self.sampler.get_log_prob()]
            return Chain(data=data, params=self.varied_params + ['fweight', 'logposterior'])
        return None

    def _add_check(self, diagnostics, quiet=False, **kwargs):
        """Extend :meth:`BaseBatchPosteriorSampler.check` with acceptance rate."""
        acceptance_rate = self.mpicomm.gather(self.sampler.get_acceptance_rate())
        if self.mpicomm.rank == 0:
            acceptance_rate = np.mean(acceptance_rate)
            diagnostics.add_test('current_acceptance_rate', 'current mean acceptance rate', acceptance_rate, quiet=quiet)
            acceptance_rate = [chain.fweight.size / chain.fweight.sum() for chain in self.chains if chain is not None]
            if acceptance_rate:
                diagnostics.add_test('total_acceptance_rate', 'total mean acceptance rate', np.mean(acceptance_rate), quiet=quiet)
        diagnostics.update(self.mpicomm.bcast(diagnostics))
        return True
