import functools
import logging
import warnings

import numpy as np

from desilike import utils, mpi, PipelineError
from desilike.utils import BaseClass, expand_dict, TaskManager
from desilike.samples import load_source
from desilike.samples.profiles import Profiles, ParameterBestFit, ParameterGrid, ParameterProfiles
from desilike.parameter import ParameterPriorError, Samples, ParameterCollection, is_parameter_sequence


class RegisteredProfiler(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


def bcast_values(func):

    @functools.wraps(func)
    def wrapper(self, values):
        values = np.asarray(values)
        if self._check_same_input:
            all_values = self.likelihood.mpicomm.allgather(values)
            if not all(np.allclose(values, all_values[0], atol=0., rtol=1e-7, equal_nan=True) for values in all_values if values is not None):
                raise ValueError('Input values different on all ranks: {}'.format(all_values))
        values = self.likelihood.mpicomm.bcast(values, root=0)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        mask = ~np.isnan(values).any(axis=1)
        toret = np.full(values.shape[0], -np.inf)
        values = values[mask]
        if values.size:
            toret[mask] = func(self, values)
        if isscalar and toret.size:
            toret = toret[0]
        return toret

    return wrapper


def _get_grid(self, param, grid=None, size=30, cl=2):
    if grid is not None:
        return np.array(grid, dtype='f8')
    if np.ndim(cl) == 0:
        if self.profiles is not None:
            argmax = self.profiles.bestfit.logposterior.argmax()
            cl = cl * self.profiles.error[param][argmax]
            center = self.profiles.bestfit[param][argmax]
            cl = (center - cl, center + cl)
        else:
            center = param.value
            limits = np.array(param.ref.limits)
            if param.ref.is_limited() and not hasattr(param.ref, 'scale'):
                cl = cl * (limits - center) + center
            elif param.proposal:
                cl = cl * np.array([-param.proposal, param.proposal]) + center
            else:
                raise ParameterPriorError('Provide proper parameter reference distribution or proposal for {}'.format(param))
    grid = np.linspace(*cl, num=size)
    return grid


def _profiles_transform(self, profiles):
    toret = profiles.deepcopy()

    def transform_array(array, scale_only=False):
        try:
            iparam = self.varied_params.index(array.param)
        except KeyError:
            return array
        array.param = self.varied_params[iparam]
        array = array * self._params_transform_scale[iparam]
        if not scale_only: array += self._params_transform_loc[iparam]
        return array

    for name, item in toret.items():
        if name == 'covariance':
            iparams = [self.varied_params.index(param) for param in item._params]
            item._params = self.varied_params.sort(key=iparams)
            item._value = item._value * (self._params_transform_scale[iparams, None] * self._params_transform_scale[iparams])
        elif name == 'contour':
            item.data = [tuple(transform_array(array) for array in arrays) for arrays in item.data]
        else:  # 'start', 'bestfit', 'error', 'interval', 'profile'
            item.data = [transform_array(array, scale_only=(name in ['error', 'interval'])) for array in item.data]
        toret.set(name=item)
    return toret


def _iterate_over_params(self, params, method, **kwargs):
    nparams = len(params)
    nprocs_per_param = max(self.mpicomm.size // nparams, 1)
    if self.profiles is None:
        start = self._get_start()
    else:
        argmax = self.profiles.bestfit.logposterior.argmax()
        start = self._params_backward_transform([self.profiles.bestfit[param][argmax] for param in self.varied_params])
    list_profiles = [None] * nparams
    mpicomm_bak = self.mpicomm
    with TaskManager(nprocs_per_task=nprocs_per_param, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
        self.mpicomm = tm.mpicomm
        for iparam, param in tm.iterate(list(enumerate(params))):
            self.derived = None
            profiles = method(start, self.chi2, self.transformed_params, param, **kwargs)
            list_profiles[iparam] = _profiles_transform(self, profiles) if self.mpicomm.rank == 0 else None
    self.mpicomm = mpicomm_bak
    profiles = Profiles()
    for iprofile, profile in enumerate(list_profiles):
        mpiroot_worker = self.mpicomm.rank if profile is not None else None
        for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
            if mpiroot_worker is not None: break
        assert mpiroot_worker is not None
        profiles.update(Profiles.bcast(profile, mpicomm=self.mpicomm, mpiroot=mpiroot_worker))

    if self.profiles is None:
        self.profiles = profiles
    else:
        self.profiles.update(profiles)

    if self.mpicomm.rank == 0 and self.save_fn is not None:
        self.profiles.save(self.save_fn)
    return self.profiles


class BaseProfiler(BaseClass, metaclass=RegisteredProfiler):

    name = 'base'
    _check_same_input = False

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, profiles=None, ref_scale=1., rescale=False, covariance=None, save_fn=None, mpicomm=None):
        """
        Initialize profiler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

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
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = likelihood
        self.pipeline = self.likelihood.runtime_info.pipeline
        self.mpicomm = mpicomm
        self.likelihood.solved_default = '.best'
        self.varied_params = self.likelihood.varied_params.deepcopy()
        for param in self.varied_params: param.update(ref=param.ref.affine_transform(scale=ref_scale))
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        self.max_tries = int(max_tries)
        self.profiles = profiles
        if profiles is not None and not isinstance(profiles, Profiles):
            self.profiles = Profiles.load(profiles)
        self._set_rng(rng=rng, seed=seed)
        if self.mpicomm.rank == 0:
            covariance = load_source(covariance, cov=True, params=self.varied_params, return_type='nparray')
        covariance = self.mpicomm.bcast(covariance, root=0)
        self.varied_params = self.mpicomm.bcast(self.varied_params, root=0)
        self.transformed_params = self.varied_params.deepcopy()

        if rescale:

            self._params_transform_loc = np.array([param.value for param in self.varied_params], dtype='f8')
            self._params_transform_scale = np.diag(covariance)**0.5

            def _params_forward_transform(values):
                return values * self._params_transform_scale + self._params_transform_loc

            def _params_backward_transform(values):
                return (values - self._params_transform_loc) / self._params_transform_scale

            self.transformed_params = ParameterCollection()
            for param, loc, scale in zip(self.varied_params, self._params_transform_loc, self._params_transform_scale):
                loc, scale = - loc, 1. / scale
                param = param.clone(prior=param.prior.affine_transform(loc=loc, scale=scale),
                                    ref=param.ref.affine_transform(loc=loc, scale=scale),
                                    proposal=param.proposal * scale)
                self.transformed_params.set(param)

        else:

            self._params_transform_loc = np.zeros(len(self.transformed_params), dtype='f8')
            self._params_transform_scale = np.ones(len(self.transformed_params), dtype='f8')

            def _params_forward_transform(values):
                return values

            def _params_backward_transform(values):
                return values

            self.transformed_params = self.transformed_params.deepcopy()

        self._params_forward_transform = _params_forward_transform
        self._params_backward_transform = _params_backward_transform
        self.derived = None
        self.save_fn = save_fn

    @bcast_values
    def logposterior(self, values):
        logprior = self.logprior(values)
        mask_finite_prior = ~np.isinf(logprior)
        if not mask_finite_prior.any():
            return logprior
        points = Samples(self._params_forward_transform(values[mask_finite_prior]).T, params=self.varied_params)
        self.pipeline.mpicalculate(**points.to_dict())
        logposterior, raise_error = None, None
        if self.pipeline.mpicomm.rank == 0:
            update_derived = True
            di = {}
            try:
                di = {'loglikelihood': self.pipeline.derived[self.likelihood._param_loglikelihood],
                      'logprior': self.pipeline.derived[self.likelihood._param_logprior]}
            except KeyError:
                di['loglikelihood'] = di['logprior'] = np.full(points.shape, -np.inf)
                update_derived = False
            if self.pipeline.errors:
                for ipoint, error in self.pipeline.errors.items():
                    if isinstance(error[0], self.likelihood.catch_errors):
                        self.log_debug('Error "{}" raised with parameters {} is caught up with -inf loglikelihood. Full stack trace\n{}:'.format(error[0],
                                       {k: v.flat[ipoint] for k, v in points.items()}, error[1]))
                        for values in di.values():
                            values[ipoint, ...] = -np.inf  # should be useless, as no step with -inf loglikelihood should be kept
                    else:
                        raise_error = error
                        update_derived = False
                    if raise_error is None and not self.logger.isEnabledFor(logging.DEBUG):
                        warnings.warn('Error "{}" raised is caught up with -inf loglikelihood. Set logging level to debug to get full stack trace.'.format(error[0]))
            if update_derived:
                if self.derived is None:
                    self.derived = [points, self.pipeline.derived]
                else:
                    self.derived = [Samples.concatenate([self.derived[0], points]),
                                    Samples.concatenate([self.derived[1], self.pipeline.derived])]
            logposterior = logprior.copy()
            logposterior[mask_finite_prior] = 0.
            for name, values in di.items():
                values = values[()]
                mask = np.isnan(values)
                values[mask] = -np.inf
                logposterior[mask_finite_prior] += values
                if mask.any() and self.mpicomm.rank == 0:
                    warnings.warn('{} is NaN for {}'.format(name, {k: v[mask] for k, v in points.items()}))
        else:
            self.derived = None
        raise_error = self.likelihood.mpicomm.bcast(raise_error, root=0)
        if raise_error:
            raise PipelineError('Error "{}" occured with stack trace:\n{}'.format(*raise_error))
        return self.likelihood.mpicomm.bcast(logposterior, root=0)

    @bcast_values
    def logprior(self, values):
        toret = 0.
        values = self._params_forward_transform(values)
        for param, value in zip(self.varied_params, values.T):
            toret += param.prior(value)
        return toret

    def chi2(self, values):
        return -2. * self.logposterior(values)

    def __getstate__(self):
        state = {}
        for name in ['max_tries']:
            state[name] = getattr(self, name)
        return state

    def _set_rng(self, rng=None, seed=None):
        self.rng = self.mpicomm.bcast(rng, root=0)
        if self.rng is None:
            seed = mpi.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None)
            self.rng = np.random.RandomState(seed=seed)

    def _set_profiler(self):
        raise NotImplementedError

    def _get_start(self, start=None, niterations=1, max_tries=None):
        if max_tries is None:
            max_tries = self.max_tries

        self._set_rng(rng=self.rng)  # to make sure all processes have the same rng

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                if param.ref.is_proper():
                    toret.append(param.ref.sample(size=size, random_state=self.rng))
                else:
                    toret.append([param.value] * size)
            return np.array(toret).T

        shape = (niterations, len(self.varied_params))
        if start is not None:
            start = np.asarray(start)
            if start.shape != shape:
                raise ValueError('Provide start with shape {}'.format(shape))
            return start

        start = np.full(shape, np.nan)
        logposterior = np.full(shape[:1], -np.inf)

        for itry in range(max_tries):
            mask = np.isfinite(logposterior)
            if mask.all(): break
            mask = ~mask
            values = get_start(size=mask.sum())
            values = self._params_backward_transform(values)
            start[mask] = values
            logposterior[mask] = self.logposterior(values)

        if not np.isfinite(logposterior).all():
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(max_tries))

        return start, logposterior

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = self.pipeline.mpicomm = mpicomm

    def maximize(self, niterations=None, start=None, **kwargs):
        """
        Maximize :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.start`
        - :attr:`Profiles.bestfit`
        - :attr:`Profiles.error`  # parabolic errors at best fit (if made available by the profiler)
        - :attr:`Profiles.covariance`  # parameter covariance at best fit (if made available by the profiler).

        One will typically run several independent likelihood maximizations in parallel,
        on number of MPI processes - 1 ranks (1 if single process), to make sure the global maximum is found.

        Parameters
        ----------
        niterations : int, default=None
            Number of iterations, i.e. of runs of the profiler from independent starting points.
            If ``None``, defaults to :attr:`mpicomm.size - 1` (if > 0, else 1).

        **kwargs : dict
            Optional profiler-specific arguments.
        """
        if start is not None:
            if not utils.is_sequence(start):
                start = [start]
            if niterations is None:
                niterations = len(start)
        if niterations is None: niterations = max(self.mpicomm.size, 1)
        niterations = int(niterations)
        start, logposterior = self._get_start(start=start, niterations=niterations)
        nprocs_per_iteration = max(self.mpicomm.size // niterations, 1)
        list_profiles = [None] * niterations
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                p = self._maximize_one(start[ii], self.chi2, self.transformed_params, **kwargs)
                if self.mpicomm.rank == 0:
                    profiles = Profiles(start=Samples(start[ii], params=self.varied_params),
                                        bestfit=ParameterBestFit(list(start[ii]), params=self.varied_params, loglikelihood=self.likelihood._param_loglikelihood, logprior=self.likelihood._param_logprior))
                    profiles.bestfit.logposterior = logposterior[ii]
                    profiles.update(p)
                    profiles = _profiles_transform(self, profiles)
                    for param in self.likelihood.params.select(fixed=True, derived=False):
                        profiles.bestfit[param] = np.array(param.value, dtype='f8')
                    index_in_profile, index = self.derived[0].match(profiles.bestfit, params=profiles.start.params())
                    assert index_in_profile[0].size == 1
                    #logposterior = -(self.derived[1][self.likelihood._param_loglikelihood][index] + self.derived[1][self.likelihood._param_logprior][index])
                    #covariance = []
                    #if logposterior.derivs:
                    #    from desilike.parameter import ParameterPrecision
                    #    solved_params = ParameterCollection([self.likelihood.all_params[param] for deriv in logposterior.derivs for param in deriv.keys()])
                    #    covariance = ParameterPrecision(logposterior[0], params=solved_params).to_covariance()
                    for array in self.derived[1]:
                        profiles.bestfit.set(array[index])
                        #if array.param in covariance:
                        #    profiles.error[array.param] = covariance.std([array.param])
                    if profiles.bestfit._logposterior not in profiles.bestfit:
                        profiles.bestfit.logposterior = profiles.bestfit[profiles.bestfit._loglikelihood] + profiles.bestfit[profiles.bestfit._logprior]
                    profiles.bestfit.logposterior.param.update(derived=True, latex=utils.outputs_to_latex(profiles.bestfit._logposterior))
                else:
                    profiles = None
                list_profiles[ii] = profiles
        self.mpicomm = mpicomm_bak
        for iprofile, profile in enumerate(list_profiles):
            mpiroot_worker = self.mpicomm.rank if profile is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            list_profiles[iprofile] = Profiles.bcast(profile, mpicomm=self.mpicomm, mpiroot=mpiroot_worker)

        profiles = Profiles.concatenate(list_profiles)

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles = Profiles.concatenate(self.profiles, profiles)

        attrs = {name: getattr(self.likelihood, name, None) for name in ['size', 'nvaried', 'ndof']}
        for name, value in attrs.items():
            for value in self.mpicomm.allgather(value):
                if value is not None:
                    attrs[name] = value
                    break
        self.profiles.bestfit.attrs.update(attrs)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def interval(self, params=None, **kwargs):
        """
        Compute confidence intervals for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.interval`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to estimate confidence intervals.

        **kwargs : dict
            Optional arguments for specific profiler.
        """
        if params is None:
            params = self.varied_params
        else:
            if not is_parameter_sequence(params): params = [params]
            params = ParameterCollection([self.varied_params[param] for param in params])
        return _iterate_over_params(self, params, self._interval_one, **kwargs)

    def contour(self, params=None, **kwargs):
        """
        Compute 2D contours for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.contour`

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            List of tuples of parameters for which to compute 2D contours.
            If a list of parameters is provided instead, contours are computed for unique tuples of parameters.

        **kwargs : dict
            Optional arguments for specific profiler.
        """
        if params is None:
            params = self.varied_params
        params = list(params)
        if not is_parameter_sequence(params[0]):
            params = [(param1, param2) for iparam1, param1 in enumerate(params) for param2 in params[iparam1 + 1:]]
        params = [(self.varied_params[param1], self.varied_params[param2]) for param1, param2 in params]
        return _iterate_over_params(self, params, self._contour_one, **kwargs)

    def grid(self, params=None, grid=None, size=1, cl=2, niterations=1, **kwargs):
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

        **kwargs : dict
            Optional arguments for specific profiler.
        """
        if params is not None and not is_parameter_sequence(params):
            params = [params]
        if isinstance(grid, Samples):
            pass
        elif params is not None and getattr(grid, 'ndim', None) == len(params) + 1:
            params = [self.varied_params[param] for param in params]
            grid = Samples([np.asarray(g) for g in grid], params=params)
        else:
            if params is None:
                for name in ['grid', 'size', 'cl']:
                    try:
                        params = list(locals()[name].keys())
                    except AttributeError:
                        pass
                    else:
                        break
            params = self.varied_params.select(name=[str(param) for param in params])
            grid = expand_dict(grid, params.names())
            size = expand_dict(size, params.names())
            cl = expand_dict(cl, params.names())
            for param in params:
                grid[param.name] = _get_grid(self, param, grid=grid[param.name], size=size[param.name], cl=cl[param.name])
            grid = list(grid.values())
            grid = Samples(np.meshgrid(*grid, indexing='ij'), params=params)

        grid = ParameterGrid(grid)
        grid_params = grid.params()
        nsamples = grid.size
        nprocs_per_param = max(self.mpicomm.size // nsamples, 1)
        start = self._get_start(niterations=niterations)[0]

        flat_grid = grid.ravel()
        varied_params = self.transformed_params - grid_params
        varied_indices = [self.varied_params.index(param) for param in varied_params]  # varied_params ordered as self.varied_params
        grid_indices = [self.varied_params.index(param) for param in grid_params]
        insert_indices = grid_indices - np.arange(len(grid_indices))
        start = [start[varied_indices] for start in start]

        states = {}
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_param, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ipoint in tm.iterate(range(nsamples)):
                self.derived = None
                point = flat_grid.choice(index=ipoint, params=grid_params, return_type='nparray')
                point = (point - self._params_transform_loc[grid_indices]) / self._params_transform_scale[grid_indices]

                def chi2(values):
                    values = np.insert(values, insert_indices, point, axis=-1)
                    return self.chi2(values)

                if varied_params:
                    profile = Profiles.concatenate([self._maximize_one(start, chi2, varied_params, **kwargs) for start in start])
                    try:
                        logposterior = profile.bestfit.logposterior.max()
                    except AttributeError:
                        logposterior = -np.inf
                else:
                    logposterior = -0.5 * chi2([])
                states[ipoint] = logposterior

        self.mpicomm = mpicomm_bak
        states = self.mpicomm.gather(states, root=0)
        logposterior = None
        if self.mpicomm.rank == 0:
            logposterior = {}
            for state in states: logposterior.update(state)
            logposterior = np.array([logposterior[i] for i in range(nsamples)])
        grid.logposterior = self.mpicomm.bcast(logposterior, root=0)
        if self.profiles is None:
            self.profiles = Profiles()
        self.profiles.set(grid=grid)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def profile(self, params=None, grid=None, size=30, cl=2, **kwargs):
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

        **kwargs : dict
            Optional arguments for specific profiler.
        """
        if params is None:
            params = self.varied_params
        else:
            if not is_parameter_sequence(params):
                params = [params]
            params = ParameterCollection([self.varied_params[param] for param in params])

        if grid is None or np.ndim(grid[0]) == 0: grid = [grid] * len(params)
        if not utils.is_sequence(size): size = [size] * len(params)
        if not utils.is_sequence(cl): cl = [cl] * len(params)

        if grid is not None and len(grid) != len(params):
            raise ValueError('Provide a list of grids, one for each of {}'.format(params))

        nparams = len(params)
        nprocs_per_param = max(self.mpicomm.size // nparams, 1)
        list_profiles = [None] * nparams
        profiles_bak, save_fn_bak, mpicomm_bak = self.profiles, self.save_fn, self.mpicomm
        self.save_fn = None
        with TaskManager(nprocs_per_task=nprocs_per_param, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for iparam, param in tm.iterate(list(enumerate(params))):
                self.profiles, self.derived = profiles_bak.copy() if profiles_bak is not None else None, None
                profiles = self.grid(params=param, grid=grid[iparam], size=size[iparam], cl=cl[iparam], **kwargs)
                list_profiles[iparam] = profiles
        self.profiles, self.save_fn, self.mpicomm = profiles_bak, save_fn_bak, mpicomm_bak
        profiles = Profiles()
        for iprofile, profile in enumerate(list_profiles):
            mpiroot_worker = self.mpicomm.rank if profile is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            profiles.update(Profiles.bcast(profile, mpicomm=self.mpicomm, mpiroot=mpiroot_worker))
        profile = ParameterProfiles([np.column_stack([profiles.grid[param], profiles.grid['logposterior']]) for param in params], params=params)
        profiles.set(profile=profile)
        del profiles.grid

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles.update(profiles)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles