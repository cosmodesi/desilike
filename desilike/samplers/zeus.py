import logging
import random

import numpy as np

from desilike.samples import Chain
from desilike import utils
from .base import BaseBatchPosteriorSampler
from .utils import numpy_to_python_random_state


class ZeusSampler(BaseBatchPosteriorSampler):

    """Wrapper for the zeus sampler (Ensemble Slice Sampling method), see https://github.com/minaskar/zeus."""

    name = 'zeus'

    def __init__(self, *args, nwalkers=None, light_mode=False, **kwargs):
        """
        Initialize zeus sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        nwalkers : int, str, default=None
            Number of walkers, defaults to 2 * max((int(2.5 * ndim) + 1) // 2, 2)
            Can be given in dimension units, e.g. '3 * ndim'.

        light_mode : bool, default=False
            If ``True`` then no expansions are performed after the tuning phase.
            This can significantly reduce the number of likelihood evaluations but works best in target distributions that are approximately Gaussian.

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
        super(ZeusSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': ndim})
        import zeus
        handlers = logging.root.handlers.copy()
        level = logging.root.level
        self.sampler = zeus.EnsembleSampler(self.nwalkers, ndim, self.logposterior, verbose=False, light_mode=bool(light_mode), vectorize=True)
        logging.root.handlers = handlers
        logging.root.level = level

    def run(self, *args, **kwargs):
        """
        Run chains. Sampling can be interrupted anytime, and resumed by providing the path to the saved chains in ``chains`` argument of :meth:`__init__`.

        One will typically run sampling on ``nchains * nprocs_per_chain + 1`` processes, with ``nchains >= 1`` the number of chains
        and ``nprocs_per_chain = max((mpicomm.size - 1) // nchains, 1)`` the number of processes per chain --- plus 1 root process to distribute the work.

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
        return super(ZeusSampler, self).run(*args, **kwargs)

    def _run_one(self, start, niterations=300, thin_by=1, progress=False):
        py_random_state_bak, np_random_state_bak = random.getstate(), np.random.get_state()
        random.setstate(numpy_to_python_random_state(self.rng.get_state()))  # self.rng is same for all ranks
        np.random.set_state(self.rng.get_state())
        for _ in self.sampler.sample(start=start, iterations=niterations, progress=progress, thin_by=thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        random.setstate(py_random_state_bak)
        np.random.set_state(np_random_state_bak)
        return Chain(data=data, params=self.varied_params + ['logposterior'])

    @classmethod
    def install(cls, config):
        config.pip('zeus-mcmc')
