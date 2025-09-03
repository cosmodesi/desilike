import os

import numpy as np

from desilike.samples import Chain
from desilike import utils
from .base import BaseBatchPosteriorSampler


class Prior(object):

    """Prior distribution for PocoMC."""

    def __init__(self, params, random_state=None):
        self.dists = [param.prior for param in params]
        self.random_state = random_state

    def logpdf(self, x):
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists):
            logp += dist(x[:,i])
        return logp

    def rvs(self, size=1):
        samples = []
        for dist in self.dists:
            samples.append(dist.sample(size=size, random_state=self.random_state))
        return np.transpose(samples)

    @property
    def bounds(self):
        bounds = []
        for dist in self.dists:
            bounds.append(dist.limits)
        return np.array(bounds)

    @property
    def dim(self):
        return len(self.dists)


class PocoMCSampler(BaseBatchPosteriorSampler):
    """
    Wrapper for PocoMC sampler (preconditioned Monte Carlo method).

    Reference
    ---------
    - https://github.com/minaskar/pocomc
    - https://arxiv.org/abs/2207.05652
    - https://arxiv.org/abs/2207.05660
    """
    name = 'pocomc'

    def __init__(self, *args, n_active=250, n_ess=1000, flow='maf6', train_config=None,
                 precondition=True, n_prior=None, sample='tpcn', max_steps=None, patience=None, ess_threshold=None, **kwargs):
        """
        Initialize PocoMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        n_active : int, str, default=250
            The number of active particles (default is ``n_active=250``). It must be smaller than ``n_ess``.
            Defaults to :attr:`Chain.shape[1]` of input chains, if any,
            else ``2 * max((int(2.5 * ndim) + 1) // 2, 2)``.
            Can be given in dimension units, e.g. '3 * ndim'.

        n_ess : int, default=1000
            The effective sample size maintained during the run (default is ``n_ess=1000``).

        flow : ``torch.nn.Module``
            Normalizing flow (default is ``maf6``). The default is a Masked Autoregressive Flow
            (MAF) with 6 blocks of 3x64 layers and residual connections.

        train_config : dict, default=None
            Configuration for training the normalizing flow
            (default is ``train_config=None``). Options include a dictionary with the following
            keys: ``"validation_split"``, ``"epochs"``, ``"batch_size"``, ``"patience"``,
            ``"learning_rate"``, ``"annealing"``, ``"gaussian_scale"``, ``"laplace_scale"``,
            ``"noise"``, ``"shuffle"``, ``"clip_grad_norm"``, ``"verbose"``.

        precondition : bool, default=True
            If True, use preconditioned MCMC (default is ``precondition=True``). If False,
            use standard MCMC without normalizing flow.

        n_prior : int, default=None
            Number of prior samples to draw (default is ``n_prior=2*(n_ess//n_active)*n_active``).

        sample : str, default='tpcn'
            Type of MCMC sampler to use (default is ``sample='tpcn'``). Options are
            ``'tpcn'`` (t-Preconditioned Crank-Nicolson) or ``'rwm'`` (Random-Walk Metropolis).

        max_steps : int, default=None
            Maximum number of MCMC steps (default is ``max_steps=5*n_dim``).

        patience : int, default=None
            Number of steps for early stopping of MCMC (default is ``patience=None``). If ``patience=None``,
            MCMC terminates automatically.

        ess_threshold : int, default=None
            Effective sample size threshold for resampling (default is ``ess_threshold=4*n_dim``).

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
            Save samples to this location. This is mandatory, because as for now PocoMC does not provide proper __getstate__/__setstate__ methods.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`.
        """
        super(PocoMCSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if n_active is None:
            shapes = self.mpicomm.bcast([chain.shape if chain is not None else None for chain in self.chains], root=0)
            if any(shape is not None for shape in shapes):
                try:
                    nwalkers = shapes[0][1]
                    assert all(shape[1] == nwalkers for shape in shapes)
                except (IndexError, AssertionError) as exc:
                    nwalkers = 250  # default
                n_active = nwalkers
        self.nwalkers = utils.evaluate(n_active, type=int, locals={'ndim': ndim})
        bounds = np.array([tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in self.varied_params], dtype='f8')
        import pocomc
        self.prior = Prior(self.varied_params)
        self.sampler = pocomc.Sampler(self.prior, self.loglikelihood, n_dim=ndim, n_effective=n_ess, n_active=self.nwalkers, flow=flow, train_config=train_config, precondition=precondition, n_prior=n_prior, sample=sample, n_max_steps=max_steps, n_steps=patience, vectorize=True, output_dir=None, output_label=None, random_state=self.rng.randint(0, high=0xffffffff))
        if self.save_fn is None:
            raise ValueError('save_fn must be provided, in order to save pocomc state')
        self.state_fn = [os.path.splitext(fn)[0] + '.pocomc.state' for fn in self.save_fn]

    def loglikelihood(self, values):
        return self.logposterior(values) - self.prior.logpdf(values)

    def _prepare(self):
        self.resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)
        #self.chains = [None] * len(self.chains)

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
        return super(PocoMCSampler, self).run(*args, **kwargs)

    def _run_one(self, start, niterations=300, progress=False, **kwargs):
        load_state_bak = type(self.sampler).load_state
        resume_state_path = None

        if self.resume:
            resume_state_path = self.state_fn[self._ichain]

            def load_state(path):
                load_state_bak(self.sampler, path)
                #if self.mpicomm.rank == 0:
                #    self.derived = self.sampler.derived
                #    del self.sampler.derived
                from pocomc.tools import FunctionWrapper
                # Because dill is unable to cope with our loglikelihood and logprior
                self.sampler.log_likelihood = FunctionWrapper(self.loglikelihood, args=None, kwargs=None)
                self.sampler.log_prior = FunctionWrapper(self.logprior, args=None, kwargs=None)
                x = np.asarray(self.sampler.results.get('x'))
                self.sampler.log_likelihood(x.reshape(-1, x.shape[-1]))  # to set derived parameters
                #particles = self.sampler.particles
                #particles.__init__(n_particles=particles.n_particles, n_dim=particles.n_dim, ess_threshold=particles.ess_threshold)  # clear particles
        else:
            def load_state(path):
                load_state_bak(self.sampler, path)

        self.sampler.load_state = load_state

        import torch
        np_random_state_bak, torch_random_state_bak = np.random.get_state(), torch.get_rng_state()
        self.sampler.random_state = self.rng.randint(0, high=0xffffffff)
        self.prior.random_state = self.rng
        np.random.set_state(self.rng.get_state())  # self.rng is same for all ranks
        torch.set_rng_state(self.mpicomm.bcast(torch_random_state_bak, root=0))

        t0 = self.sampler.t
        nparticles = len(self.sampler.particles.get('x'))

        def _not_termination(current_particles):
            return (self.sampler.t - t0) < niterations

        self.sampler._not_termination = _not_termination
        if niterations > 0:
            self.sampler.run(n_total=niterations, progress=progress, resume_state_path=resume_state_path, **kwargs)  # progress=True else bug
        else:
            return None
        np.random.set_state(np_random_state_bak)
        torch.set_rng_state(torch_random_state_bak)
        #self.sampler.load_state = load_state_bak
        particles = self.sampler.particles
        particles.results_dict = None  # to recompute results
        try:
            result = self.sampler.results
        except ValueError:
            return None
        # This is not picklable
        particles.__init__(n_particles=particles.n_particles, n_dim=particles.n_dim)  # clear particles
        particles.update(self.sampler.current_particles)  # for next iteration, only last particles are needed
        try:
            del self.sampler.log_likelihood, self.sampler.log_prior, self.sampler._not_termination, self.sampler.load_state
        except AttributeError:
            pass
        # Clear saved quantities to save space
        for name in self.sampler.__dict__:
            if name.startswith('saved_'): setattr(self.sampler, name, [])
        # Save last parameters, which be reused in the next run
        if self.mpicomm.rank == 0:
            #self.sampler.derived = [d[-1:] for d in self.derived]
            self.sampler.save_state(self.state_fn[self._ichain])  # save ~ all self.sampler.__dict__
        data = [np.asarray(result['x'][..., iparam], dtype='f8')[nparticles:] for iparam, param in enumerate(self.varied_params)] + [np.asarray(result['logl'] + result['logp'], dtype='f8')[nparticles:]]
        return Chain(data=data, params=self.varied_params + ['logposterior']).reshape(-1, self.nwalkers)

    @classmethod
    def install(cls, config):
        config.pip('pocomc')