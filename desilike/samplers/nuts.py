import numpy as np
from desilike.jax import numpy as jnp
from desilike.jax import jit

from desilike.samples import Chain, Samples, load_source
from .base import BaseBatchPosteriorSampler


def diag_if_diag(mat, rtol=1e-05, atol=1e-08):
    mat = np.asarray(mat)
    diag = np.diag(mat)
    if np.allclose(np.diag(diag), mat, rtol=rtol, atol=atol):
        return diag
    return mat


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
            Adapt inverse mass matrix (``covariance``) and ``step_size``.
            Can be ``{'is_mass_matrix_diagonal': True, 'target_acceptance_rate': 0.8}``.

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
        self.attrs['inverse_mass_matrix'] = diag_if_diag(np.array(covariance))
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

        #mpicomm = self.likelihood.mpicomm
        #self.likelihood.mpicomm = mpi.COMM_SELF

        if self.algorithm is None:

            def logdensity_fn(values):
                return self.likelihood(dict(zip(self.varied_params.names(), values)))

            adaptation = self.attrs['adaptation']
            if isinstance(adaptation, dict):
                adaptation = dict(adaptation)
                niterations_adaptation = adaptation.pop('niterations', 1000)  # better spend time on good adaptation
                adaptation.setdefault('initial_step_size', self.attrs['step_size'])

                #def integrator(logdensity_fn, kinetic_energy_fn):
                #    return jax.jit(self.attrs['integrator'](logdensity_fn, kinetic_energy_fn))

                #adaptation['integrator'] = integrator
                #warmup = window_adaptation(blackjax.nuts, logdensity_fn=logdensity_fn, **adaptation)
                warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn=logdensity_fn, **adaptation)


                (initial_state, warmup_params), _ = warmup.run(warmup_key, start, niterations_adaptation)
                self.hyp = dict(warmup_params)
            elif self.hyp is None:
                self.hyp = {name: self.attrs[name] for name in ['step_size', 'inverse_mass_matrix']}
            self.log_info('Using hyperparameters: {}.'.format(self.hyp))
            # use the quick wrapper to build a new kernel with the tuned parameters
            attrs = {name: self.attrs[name] for name in ['max_num_doublings', 'divergence_threshold', 'integrator']}
            self.algorithm = blackjax.nuts(logdensity_fn, **attrs, **self.hyp)
            #self.step = jax.jit(self.algorithm.step)
            #self.step = self.algorithm.step
            def one_step(state, xs):
                _, rng_key = xs
                state, info = self.algorithm.step(rng_key, state)
                return state, (state, info)

            self.one_step = one_step

        initial_state = self.algorithm.init(start, warmup_key)
        #print('compiling')
        #import time
        #t0 = time.time()
        #self.step(run_key, initial_state)
        #print('done compiling', time.time() - t0)
        keys = jax.random.split(run_key, niterations)
        xs = (jnp.arange(niterations), keys)
        # run the sampler, following https://github.com/blackjax-devs/blackjax/blob/54023350cac935af79fc309006bf37d1603bb945/blackjax/util.py#L143

        final_state, (chain, info_history) = jax.lax.scan(self.one_step, initial_state, xs)
        #final_state, (chain, info_history) = my_scan(one_step, initial_state, xs)
        position, logdensity = chain.position[::thin_by], chain.logdensity[::thin_by]

        data = [position[:, iparam] for iparam, param in enumerate(self.varied_params)] + [logdensity]
        chain = Chain(data=data, params=self.varied_params + ['logposterior'], attrs={'hyp': self.hyp})
        #self.likelihood.mpicomm = mpicomm
        self.derived = None
        #self.derived = [chain.select(name=self.varied_params.names()), Samples()]
        #logprior = sum(param.prior(chain[param]) for param in self.varied_params)
        #self.derived[1][self.likelihood._param_logprior] = logprior
        #self.derived[1][self.likelihood._param_loglikelihood] = chain['logposterior'] - logprior
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


def window_adaptation(algorithm,
    logdensity_fn,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    **extra_parameters):
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily.

    Algorithms in the HMC family on a euclidean manifold depend on the value of
    at least two parameters: the step size, related to the trajectory
    integrator, and the mass matrix, linked to the euclidean metric.

    Good tuning is very important, especially for algorithms like NUTS which can
    be extremely inefficient with the wrong parameter values. This function
    provides a general-purpose algorithm to tune the values of these parameters.
    Originally based on Stan's window adaptation, the algorithm has evolved to
    improve performance and quality.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    progress_bar
        Whether we should display a progress bar.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that runs the adaptation and returns an `AdaptationResult` object.

    """
    import jax
    from blackjax.base import AdaptationAlgorithm
    from blackjax.adaptation.window_adaptation import base, build_schedule
    from blackjax.adaptation.base import AdaptationInfo, AdaptationResults

    mcmc_kernel = algorithm.build_kernel(**{name: extra_parameters.pop(name) for name in ['integrator'] if name in extra_parameters})

    adapt_init, adapt_step, adapt_final = base(
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            info.acceptance_rate,
        )

        return (
            (new_state, new_adaptation_state),
            AdaptationInfo(new_state, info, new_adaptation_state),
        )

    def run(rng_key, position, num_steps: int = 1000):
        init_state = algorithm.init(position, logdensity_fn)
        init_adaptation_state = adapt_init(position, initial_step_size)

        keys = jax.random.split(rng_key, num_steps)
        schedule = build_schedule(num_steps)

        #last_state, info = jax.lax.scan(
        last_state, info = my_scan(
            one_step,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys, schedule),
        )
        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)


def my_scan(f, init, xs):
    import jax.tree_util as jtu
    carry = init
    outs = []
    for xx in zip(*xs):
        carry, out = f(carry, xx)
        outs.append(out)
    return carry, jtu.tree_map(lambda *v: jnp.stack(v), *outs)