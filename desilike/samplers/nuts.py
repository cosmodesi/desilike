import numpy as np
from desilike.jax import numpy as jnp

from desilike.samples import Chain, load_source
from .base import BaseBatchPosteriorSampler


class NUTSSampler(BaseBatchPosteriorSampler):
    """
    Wrapper for the No U-Turn sampler.

    Reference
    ---------
    - https://github.com/blackjax-devs/blackjax
    """
    name = 'nuts'

    def __init__(self, *args, adaptation=True, covariance=None, step_size=1e-3, max_num_doublings=10, divergence_threshold=1000, integrator='velocity_verlet', **kwargs):
        """
        Initialize NUTS HMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        adaptation : bool, dict, default=True
            Adapt inverse mass matrix (``covariance``) and ``step_size``.*
            Can be ``{'is_mass_matrix_diagonal': True, 'target_acceptance_rate': 0.8}``

        covariance : str, dict, Chain, Profiles, ParameterCovariance, default=None
            (Initial) proposal covariance, to use as an inverse mass matrix.
            Can be previous samples e.g. ``({fn: chain.npy, burnin: 0.5})``,
            or profiles (containing parameter covariance matrix), or parameter covariance.
            If variance for a given parameter is not provided, parameter's attr:`Parameter.proposal` squared is used.

        step_size : float, default=1e-3
            The value to use for the step size in the integrator.

        max_num_doublings : int, default=10
            The maximum number of times we double the length of the trajectory before
            returning if no U-turn has been obserbed or no divergence has occured.

        divergence_threshold : int, default=1000
            The absolute value of the difference in energy between two states above
            which we say that the transition is divergent. The default value is
            commonly found in other libraries, and yet is arbitrary.

        integrator : str, default='velocity_verlet'
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
        super(NUTSSampler, self).__init__(*args, **kwargs)
        import blackjax
        if adaptation is True: adaptation = {}
        if isinstance(integrator, str):
            integrator = getattr(blackjax.mcmc.integrators, integrator)
        self.attrs = dict(adaptation=adaptation, step_size=step_size, max_num_doublings=max_num_doublings, divergence_threshold=divergence_threshold, integrator=integrator)
        burnin = 0.5
        if isinstance(covariance, dict):
            covariance['burnin'] = covariance.get('burnin', burnin)
        else:
            covariance = {'source': covariance, 'burnin': burnin}
        if self.mpicomm.rank == 0:
            covariance = load_source(**covariance, cov=True, params=self.varied_params, return_type='nparray')
        covariance = self.mpicomm.bcast(covariance, root=0)
        self.attrs['inverse_mass_matrix'] = jnp.array(covariance)
        self.hyp = self.mpicomm.bcast(getattr(self.chains[0], 'attrs', {}).get('hyp', None), root=0)

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
        return super(NUTSSampler, self).run(*args, **kwargs)

    def _run_one(self, start, niterations=300, thin_by=1):
        import jax
        import blackjax
        from desilike import mpi
        key = jax.random.PRNGKey(self.rng.randint(0, high=0xffffffff))
        warmup_key, run_key = jax.random.split(key, 2)
        start = jnp.ravel(start)

        if self.mpicomm.size > 1:
            import warnings
            warnings.warn('NUTSSampler does not benefit from several processes per chain, please ask for {:d} processes'.format(len(self.chains)))

        names = self.varied_params.names()
        mpicomm = self.likelihood.mpicomm
        self.likelihood.mpicomm = mpi.COMM_SELF

        def logdensity_fn(values):
            return self.likelihood(dict(zip(names, values)))

        if self.hyp is None:
            adaptation = self.attrs['adaptation']
            if isinstance(adaptation, dict):
                adaptation = dict(adaptation)
                niterations = adaptation.pop('niterations', 1000)
                adaptation.setdefault('initial_step_size', self.attrs['step_size'])
                warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn=logdensity_fn, **adaptation)
                (initial_state, warmup_params), _ = warmup.run(warmup_key, start, niterations)
                start = initial_state.position
                self.hyp = dict(warmup_params)
            else:
                self.hyp = {name: self.attrs[name] for name in ['step_size', 'inverse_mass_matrix']}
        # use the quick wrapper to build a new kernel with the tuned parameters
        attrs = {name: self.attrs[name] for name in ['max_num_doublings', 'divergence_threshold', 'integrator']}
        sampling_alg = blackjax.nuts(logdensity_fn, **attrs, **self.hyp)

        # run the sampler
        _, chain, _ = blackjax.util.run_inference_algorithm(rng_key=run_key, initial_state_or_position=start, inference_algorithm=sampling_alg, num_steps=niterations, progress_bar=False)
        data = [chain.position[::thin_by, iparam] for iparam, param in enumerate(self.varied_params)] + [chain.logdensity]
        chain = Chain(data=data, params=self.varied_params + ['logposterior'], attrs={'hyp': self.hyp})
        self.likelihood.mpicomm = mpicomm
        #self.derived = [chain.select(name=self.varied_params.names()), Samples()]
        #logprior = sum(param.prior(chain[param]) for param in self.varied_params)
        #self.derived[1][self.likelihood._param_logprior] = logprior
        #self.derived[1][self.likelihood._param_loglikelihood] = chain['logposterior'] - logprior
        return chain

    @classmethod
    def install(cls, config):
        config.pip('blackjax')