import numpy as np

from desilike.samples.profiles import Profiles, Samples, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


def create_learning_rate_fn(base_learning_rate, num_epochs, steps_per_epoch=1):
    """
    Create learning rate schedule.
    Taken from https://flax.readthedocs.io/en/latest/guides/training_techniques/lr_schedule.html.
    """
    import optax
    warmup_epochs = int(0.1 * num_epochs + 0.5)
    warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def _make_tuple(obj, length=None):
    # Return tuple from ``obj``.
    if np.ndim(obj) == 0:
        obj = (obj,)
        if length is not None:
            obj *= length
    return tuple(obj)


class OptaxProfiler(BaseProfiler):

    """
    Wrapper for the collection of optax's profilers.

    Reference
    ---------
    https://github.com/google-deepmind/optax
    """

    def __init__(self, *args, method='adam', **kwargs):
        """
        Initialize profiler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        method : str, default='adam'
            Type of solver. For example ['adam', 'adamw', 'bfgs'].
            See https://github.com/google-deepmind/optax/blob/main/optax/__init__.py.

        rng : np.random.RandomState, default=None
            Random state. If ``None``, ``seed`` is used to set random state.

        seed : int, default=None
            Random seed.

        max_tries : int, default=1000
            A :class:`ValueError` is raised after this number of likelihood (+ prior) calls without finite posterior.

        profiles : str, Path, Profiles
            Path to or profiles, to which new profiling results will be added.

        ref_scale : float, default=1.
            Rescale parameters' :attr:`Parameter.ref` reference distribution by this factor

        rescale : bool, default=False
            If ``True``, internally rescale parameters such their variation range is ~ unity.
            Provide ``covariance`` to take parameter variations from;
            else parameters' :attr:`Parameter.proposal` will be used.

        covariance : str, Path, ParameterCovariance, Chain, default=None
            If ``rescale``, path to or covariance or chain, which is used for rescaling parameters.
            If ``None``, parameters' :attr:`Parameter.proposal` will be used instead.

        save_fn : str, Path, default=None
            If not ``None``, save profiles to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`
        """
        super(OptaxProfiler, self).__init__(*args, **kwargs)
        self.method = method
        self.with_gradient = True

    def maximize(self, *args, **kwargs):
        r"""
        Maximize :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.start`
        - :attr:`Profiles.bestfit`
        - :attr:`Profiles.error`  # parabolic errors at best fit (if made available by the solver)
        - :attr:`Profiles.covariance`  # parameter covariance at best fit (if made available by the solver).

        One will typically run several independent likelihood maximizations in parallel,
        on number of MPI processes - 1 ranks (1 if single process), to make sure the global maximum is found.

        Parameters
        ----------
        niterations : int, default=None
            Number of iterations, i.e. of runs of the profiler from independent starting points.
            If ``None``, defaults to :attr:`mpicomm.size - 1` (if > 0, else 1).

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.

        learning_rate : float, default=1e-2
            (Base) learning rate.

        learning_rate_scheduling : bool, callable, default=False
            If ``True``, use learning rate scheduling (cosine scheduler).
            If ``callable``, provide a function with same signature as :func:`create_learning_rate_fn`.
            See https://flax.readthedocs.io/en/latest/guides/training_techniques/lr_schedule.html.

        patience : int, tuple, list, default=100
            Wait for this number of epochs without loss improvement before stopping the optimization.

        tol : float, default=None
            Tolerance for termination. When ``tol`` is specified, the selected minimization algorithm sets some relevant solver-specific tolerance(s)
            equal to ``tol``. For detailed control, use solver-specific options.

        kwargs : dict
            Solver-specific options.
        """
        return super(OptaxProfiler, self).maximize(*args, **kwargs)

    def _maximize_one(self, start, chi2, varied_params, max_iterations=10000, learning_rate=1e-2, learning_rate_scheduling=True, patience=100, gradient=None, **kwargs):
        import jax
        import optax

        # Create the optimizer
        learning_rate_fn = learning_rate

        if isinstance(learning_rate_scheduling, bool) and learning_rate_scheduling:
            learning_rate_scheduling = create_learning_rate_fn
        if learning_rate_scheduling:
            learning_rate_fn = learning_rate_scheduling(learning_rate, max_iterations)
        tx = getattr(optax, self.method)(learning_rate_fn)

        @jax.jit
        def train_step(state, params):
            gradient_fn = jax.value_and_grad(chi2)
            loss, grads = gradient_fn(params)
            updates, state = tx.update(grads, state)
            params = optax.apply_updates(params, updates)
            return state, params, loss

        params = start
        state = tx.init(params)
        loss = np.infty
        best_params, best_loss = params, loss
        early_stopping_counter = 0

        # loop over epochs
        for epoch in range(max_iterations):
            # loop over batches
            #train_batch_metrics = []
            state, params, loss = train_step(state, params)
            # early stopping condition
            if loss < best_loss:
                best_state, best_params, best_loss = state, params, loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= patience:
                break

        profiles = Profiles()
        attrs = {}
        profiles.set(bestfit=ParameterBestFit([np.atleast_1d(xx) for xx in best_params] + [- 0.5 * chi2(best_params)], params=varied_params + ['logposterior']), attrs=attrs)
        return profiles

    def profile(self, *args, **kwargs):
        """
        Compute 1D profiles for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.profile`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to compute 1D profiles.

        grid : array, list, default=None
            Parameter values on which to compute the profile, for each parameter. If grid is set, size and bound are ignored.

        size : int, list, default=30
            Number of scanning points. Ignored if grid is set. Can be specified for each parameter.

        cl : int, list, default=2
            If bound is a number, it specifies an interval of N sigmas symmetrically around the minimum.
            Ignored if grid is set. Can be specified for each parameter.

        niterations : int, default=1
            Number of iterations, i.e. of runs of the profiler from independent starting points.

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.
        """
        return super(OptaxProfiler, self).profile(*args, **kwargs)

    def grid(self, *args, **kwargs):
        """
        Compute best fits on grid for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.grid`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to compute 1D profiles.

        grid : array, list, dict, default=None
            Parameter values on which to compute the profile, for each parameter. If grid is set, size and bound are ignored.

        size : int, list, dict, default=1
            Number of scanning points. Ignored if grid is set. Can be specified for each parameter.

        cl : int, list, dict, default=2
            If bound is a number, it specifies an interval of N sigmas symmetrically around the minimum.
            Ignored if grid is set. Can be specified for each parameter.

        niterations : int, default=1
            Number of iterations, i.e. of runs of the profiler from independent starting points.

        max_iterations : int, default=int(1e5)
            Maximum number of likelihood evaluations.
        """
        return super(OptaxProfiler, self).grid(*args, **kwargs)

    @classmethod
    def install(cls, config):
        config.pip('optax')