import numpy as np
from desilike.jax import numpy as jnp
from desilike.jax import jit

from desilike.samples import Chain, load_source
from .base import BaseBatchPosteriorSampler


class MCLMCSampler(BaseBatchPosteriorSampler):
    """
    Wrapper for the Microcanonical Langevin Monte Carlo sampler.

    Reference
    ---------
    - https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    - https://arxiv.org/abs/2212.08549
    """
    name = 'mclmc'

    def __init__(self, *args, adaptation=True, L=1., step_size=0.1, integrator='isokinetic_mclachlan', **kwargs):
        """
        Initialize MCLMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        adaptation : bool, dict, default=True
            Adapt momentum decoherence scale ``L`` and ``step_size``.
            Can be ``{'niterations': 1000, 'frac_tune1': 0.1, 'frac_tune1': 0.1, 'frac_tune2': 0.1, 'frac_tune3': 0.1,
            'desired_energy_var': 5e-4,, 'trust_in_estimate': 1.5, 'num_effective_samples': 150, 'diagonal_preconditioning': True}``

        L : float, default=1.
            Momentum decoherence scale.

        step_size : float, default=0.1
            The value to use for the step size in the integrator.

        integrator : str, default='isokinetic_mclachlan'
            Integrator, from :mod:`blackjax.mcmc.integrators`.

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
        super(MCLMCSampler, self).__init__(*args, **kwargs)
        import blackjax
        if adaptation is True: adaptation = {}
        if isinstance(integrator, str):
            integrator = getattr(blackjax.mcmc.integrators, integrator)
        self.attrs = dict(adaptation=adaptation, L=L, step_size=step_size, integrator=integrator)
        self.hyp = self.mpicomm.bcast(getattr(self.chains[0], 'attrs', {}).get('hyp', None), root=0)
        self.algorithm = None

    def run(self, *args, **kwargs):
        """
        Run chains. Sampling can be interrupted anytime, and resumed by providing the path to the saved chains in ``chains`` argument of :meth:`__init__`.

        One will typically run sampling on ``nchains`` processes,
        with ``nchains >= 1`` the number of chains and 1 process per chain.

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
        return super(MCLMCSampler, self).run(*args, **kwargs)

    @jit(static_argnums=[0])
    def _one_step(self, state, xs):
        _, rng_key = xs
        state, info = self.algorithm.step(rng_key, state)
        return state, (state, info)

    def _run_one(self, start, niterations=300, thin_by=1):
        import jax
        import blackjax
        from desilike import mpi
        key = jax.random.PRNGKey(self.rng.randint(0, high=0xffffffff))
        warmup_key, run_key = jax.random.split(key, 2)
        start = jnp.ravel(start)

        if self.mpicomm.size > 1:
            import warnings
            warnings.warn('MCLMCSampler does not benefit from several processes per chain, please ask for {:d} processes'.format(len(self.chains)))

        self.likelihood.mpicomm = mpi.COMM_SELF

        def _params_forward_transform(values):
            return values * self._params_transform_scale + self._params_transform_loc

        def _params_backward_transform(values):
            return (values - self._params_transform_loc) / self._params_transform_scale

        if self.algorithm is None:

            self._params_transform_loc = np.array([param.value for param in self.varied_params], dtype='f8')
            self._params_transform_scale = np.array([param.proposal for param in self.varied_params], dtype='f8')

            def logdensity_fn(values):
                return self.likelihood(dict(zip(self.varied_params.names(), _params_forward_transform(values))))

            adaptation = self.attrs['adaptation']
            if isinstance(adaptation, dict):
                adaptation = dict(adaptation)
                niterations_adaptation = adaptation.pop('niterations', 1000)
                initial_state = blackjax.mcmc.mclmc.init(position=start, logdensity_fn=logdensity_fn, rng_key=key)
                # build the kernel
                kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(logdensity_fn=logdensity_fn, sqrt_diag_cov=sqrt_diag_cov, integrator=self.attrs['integrator'])
                (initial_state, warmup_params) = blackjax.mclmc_find_L_and_step_size(mclmc_kernel=kernel, num_steps=niterations_adaptation, state=initial_state, rng_key=warmup_key, **adaptation)
                start = initial_state.position
                self.hyp = dict(step_size=warmup_params.step_size, L=warmup_params.L, sqrt_diag_cov=warmup_params.sqrt_diag_cov)
            elif self.hyp is None:
                self.hyp = {name: self.attrs[name] for name in ['step_size', 'L']}
            # use the quick wrapper to build a new kernel with the tuned parameters
            self.log_info('Using hyperparameters: {}.'.format(self.hyp))
            attrs = {name: self.attrs[name] for name in ['integrator']} | self.hyp
            self.algorithm = blackjax.mclmc(logdensity_fn, **attrs)

        initial_state = self.algorithm.init(start, warmup_key)
        keys = jax.random.split(run_key, niterations)
        xs = (jnp.arange(niterations), keys)
        # run the sampler, following https://github.com/blackjax-devs/blackjax/blob/54023350cac935af79fc309006bf37d1603bb945/blackjax/util.py#L143
        final_state, (chain, info_history) = jax.lax.scan(self._one_step, initial_state, xs)

        position = _params_forward_transform(chain.position)
        data = [position[::thin_by, iparam] for iparam, param in enumerate(self.varied_params)] + [chain.logdensity[::thin_by]]
        chain = Chain(data=data, params=self.varied_params + ['logposterior'], attrs={'hyp': self.hyp})
        #self.likelihood.mpicomm = mpicomm
        self.derived = None
        samples = chain.select(name=self.varied_params.names())
        results = self._vlikelihood(samples.to_dict())
        if self.mpicomm.rank == 0:
            results, errors = results
            if results:
                self.derived = [samples, results[1]]
        return chain

    @classmethod
    def install(cls, config):
        config.pip('blackjax')