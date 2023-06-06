from desilike.samples import Chain
from desilike import utils
from .base import BaseBatchPosteriorSampler


class EmceeSampler(BaseBatchPosteriorSampler):
    """
    Wrapper for the affine-invariant ensemble sampler for Markov chain Monte Carlo (MCMC) proposed by Goodman & Weare (2010).

    Reference
    ---------
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665
    """
    name = 'emcee'

    def __init__(self, *args, nwalkers=None, **kwargs):
        """
        Initialize emcee sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        nwalkers : int, str, default=None
            Number of walkers, defaults to :attr:`Chain.shape[1]` of input chains, if any,
            else ``2 * max((int(2.5 * ndim) + 1) // 2, 2)``.
            Can be given in dimension units, e.g. ``'3 * ndim'``.

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
        super(EmceeSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            shapes = self.mpicomm.bcast([chain.shape if chain is not None else None for chain in self.chains], root=0)
            if any(shape is not None for shape in shapes):
                try:
                    nwalkers = shapes[0][1]
                    assert all(shape[1] == nwalkers for shape in shapes)
                except (IndexError, AssertionError) as exc:
                    raise ValueError('Impossible to find number of walkers from input chains of shapes {}'.format(shapes)) from exc
            else:
                nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': len(self.varied_params)})
        import emcee
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.logposterior, vectorize=True)

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
        return super(EmceeSampler, self).run(*args, **kwargs)

    def _run_one(self, start, niterations=300, thin_by=1, progress=False):
        self.sampler._random = self.rng
        for _ in self.sampler.sample(initial_state=start, iterations=niterations, progress=progress, store=True, thin_by=thin_by, skip_initial_state_check=False):
            pass
        try:
            chain = self.sampler.get_chain()
        except AttributeError:
            return None
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, params=self.varied_params + ['logposterior'])

    @classmethod
    def install(cls, config):
        config.pip('emcee')
