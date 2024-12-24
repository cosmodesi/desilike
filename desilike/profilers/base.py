import functools
import logging
import warnings
import traceback

import numpy as np

from desilike import utils, mpi, PipelineError
from desilike.utils import BaseClass, expand_dict, TaskManager
from desilike.samples import load_source
from desilike.samples.profiles import Profiles, ParameterBestFit, ParameterCovariance, ParameterGrid, ParameterProfiles, ParameterArray, ParameterContours
from desilike.parameter import ParameterPriorError, Samples, ParameterCollection, is_parameter_sequence
from desilike.jax import jit, cond, numpy_jax, exception
from desilike.jax import numpy as jnp



class RegisteredProfiler(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


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


def _grid_transform_backward(self, grid):
    toret = grid.copy()

    def transform_array(array, scale_only=False):
        try:
            iparam = self.varied_params.index(array.param)
        except KeyError:
            return array
        array.param = self.varied_params[iparam]
        array = (array - self._params_transform_loc[iparam]) / self._params_transform_scale[iparam]
        return array

    toret.data = [transform_array(array) for array in toret.data]
    return toret


def _profiles_transform_forward(self, profiles):
    toret = profiles.deepcopy()

    def transform_array(array, scale_only=False, index=None):
        try:
            iparam = self.varied_params.index(array.param)
        except KeyError:
            return array
        array.param = self.varied_params[iparam]
        array[index] = array[index] * self._params_transform_scale[iparam]
        if not scale_only: array[index] += self._params_transform_loc[iparam]
        return array

    for name, item in toret.items():
        if name == 'covariance':
            iparams = [self.varied_params.index(param) for param in item._params]
            item._params = self.varied_params.sort(key=iparams)
            item._value = item._value * (self._params_transform_scale[iparams, None] * self._params_transform_scale[iparams])
        elif name == 'profile':
            item.data = [transform_array(array, index=(..., 0)) for array in item.data]
        elif name == 'grid':
            item.data = [transform_array(array) if array.param.name != item._logposterior else array for array in item.data]
        elif name == 'contour':
            for contour in item.values():
                contour.data = [tuple(transform_array(array) for array in arrays) for arrays in contour.data]
        elif name in ['error', 'interval']:  # 'start', 'bestfit', 'error', 'interval', 'profile', 'grid'
            item.data = [transform_array(array, scale_only=True) for array in item.data]
        else:  # 'start', 'bestfit'
            item.data = [transform_array(array) for array in item.data]
        toret.set(name=item)
    return toret


class ProfilerState(object):

    def __init__(self, chi2, varied_params, start, gradient=None, **kwargs):
        self.chi2 = chi2
        self.varied_params = varied_params
        self.start = start
        self.gradient = gradient
        self.__dict__.update(kwargs)

    def __copy__(self, *args, **kwargs):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, *args, **kwargs):
        return self.__copy__(*args, **kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def clone(self, **kwargs):
        new = self.copy()
        new.update(**kwargs)
        return new


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

        self.save_fn = save_fn

    def chi2(self, values):
        jnp = numpy_jax(values[0])
        values = jnp.asarray(values)
        values = self._params_forward_transform(values)

        def compute_logprior(values):
            return jnp.asarray(self.likelihood.all_params.prior(**dict(zip(self.varied_params.names(), values))))

        def warning_nan(logposterior, points):
            if np.isnan(logposterior):
                warnings.warn('logposterior is NaN for {}'.format(points))

        def compute_logposterior(values):
            points = {param.name: value for param, value in zip(self.varied_params, values)}
            raise_error = None
            logposterior = -np.inf
            try:
                logposterior = self.likelihood(points)
            except Exception as exc:
                import traceback
                error = (exc, traceback.format_exc())
                if isinstance(error[0], self.likelihood.catch_errors):
                    self.log_debug('Error "{}" raised with parameters {} is caught up with -inf loglikelihood. Full stack trace\n{}:'.format(repr(error[0]), points, error[1]))
                else:
                    raise_error = error
                if raise_error is None and not self.logger.isEnabledFor(logging.DEBUG):
                    warnings.warn('Error "{}" raised with parameters {} is caught up with -inf loglikelihood. Set logging level to debug (setup_logging("debug")) to get full stack trace.'.format(repr(error[0]), points))
                if raise_error:
                    raise PipelineError('Error "{}" occured at {} with stack trace:\n{}'.format(repr(error[0]), points, error[1]))

            exception(warning_nan, logposterior, points)
            return jnp.where(jnp.isnan(logposterior), -np.inf, logposterior)

        return -2. * cond(compute_logprior(values) > -np.inf, compute_logposterior, lambda values: -np.inf, values)

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

    def _get_vchi2(self, chi2=None, start=None, aux=None):
        """Vectorize the :math:`\chi^{2}`."""
        #self.likelihood.mpicomm = mpi.COMM_SELF
        if start is None:
            start = self._get_start(niterations=3, max_tries=None)
        aux = aux or {}

        if chi2 is None:
            if getattr(self, '_vchi2', None) is not None:
                return self._vchi2, self._gchi2
            chi2 = self.chi2

        chi2(start[0], **{name: value[0] for name, value in aux.items()})
        vchi2 = chi2
        try:
            import jax
            _vchi2 = jax.jit(chi2)
            _vchi2(start[1], **{name: value[1] for name, value in aux.items()})
        except:
            if self.mpicomm.rank == 0:
                self.log_info('Could *not* jit input likelihood.')
                self.log_info('Could *not* vmap input likelihood. Set logging level to debug (setup_logging("debug")) to get full stack trace.')
                self.log_debug('Error was {}.'.format(traceback.format_exc()))
            vchi2(start[0], **{name: value[0] for name, value in aux.items()})
        else:
            if self.mpicomm.rank == 0:
                self.log_info('Successfully jit input likelihood.')
            vchi2 = _vchi2
        gchi2 = None
        if getattr(self, 'with_gradient', False):
            try:
                import jax
                _gchi2 = jax.grad(chi2)
            except:
                if self.mpicomm.rank == 0:
                    self.log_info('Could *not* take gradient.')
            else:
                if self.mpicomm.rank == 0:
                    self.log_info('Successfully took gradient.')
                gchi2 = _gchi2
                try:
                    _gchi2 = jax.jit(gchi2)
                    _gchi2(start[2], **{name: value[2] for name, value in aux.items()})
                except:
                    if self.mpicomm.rank == 0:
                        self.log_info('Could *not* jit input gradient.')
                    gchi2(start[0], **{name: value[0] for name, value in aux.items()})
                else:
                    if self.mpicomm.rank == 0:
                        self.log_info('Successfully jit input gradient.')
                    gchi2 = _gchi2
        if chi2 is self.chi2:
            self._vchi2, self._gchi2 = vchi2, gchi2
        return vchi2, gchi2

    def _get_start(self, start=None, niterations=1, max_tries=None):
        if max_tries is None:
            max_tries = self.max_tries

        self._set_rng(rng=self.rng)  # to make sure all processes have the same rng

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                if param.ref.is_proper():
                    value = param.ref.sample(size=size, random_state=self.rng)
                else:
                    value = np.full(size, param.value)
                toret.append(value)
            return np.column_stack(toret)

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
            logposterior[mask] = [-0.5 * self.chi2(value) for value in values]

        if not np.isfinite(logposterior).all():
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(max_tries))

        return start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm

    def maximize(self, niterations=None, start=None, **kwargs):
        """
        Maximize :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.start`
        - :attr:`Profiles.bestfit`
        - :attr:`Profiles.error`  # parabolic errors at best fit (if made available by the profiler)
        - :attr:`Profiles.covariance`  # parameter covariance at best fit (if made available by the profiler).

        One will typically run several independent likelihood maximizations in parallel,
        on number of MPI processes ranks (1 if single process), to make sure the global maximum is found.

        Parameters
        ----------
        niterations : int, default=None
            Number of iterations, i.e. of runs of the profiler from independent starting points.
            If ``None``, defaults to :attr:`mpicomm.size` (if > 0, else 1).

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
        start = self._get_start(start=start, niterations=niterations)
        chi2, gradient = self._get_vchi2()

        from desilike import vmap
        vlikelihood = vmap(self.likelihood, backend=None, errors='return', return_derived=True)

        with TaskManager(nprocs_per_task=(self.mpicomm.size + niterations - 1) // niterations, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            list_profiles = []
            for ii in tm.iterate(range(niterations)):
                logposterior = -0.5 * self.chi2(start[ii])
                profiles = Profiles(start=Samples(start[ii][..., None], params=self.varied_params),
                                    bestfit=ParameterBestFit(start[ii][..., None], params=self.varied_params,
                                                             loglikelihood=self.likelihood._param_loglikelihood, logprior=self.likelihood._param_logprior))
                profiles.bestfit.logposterior[...] = logposterior
                state = ProfilerState(chi2=chi2, start=start[ii], varied_params=self.transformed_params, gradient=gradient, fast=False)
                profiles.update(self._maximize_one(state, **kwargs))
                profiles = _profiles_transform_forward(self, profiles)
                for param in self.likelihood.all_params.select(fixed=True, derived=False):
                    profiles.bestfit[param] = np.array([param.value], dtype='f8')

                ret, exc = vlikelihood(profiles.bestfit.to_dict(params=profiles.bestfit.params(input=True)))
                for ipoint, error in exc.items():
                    self.log_warning('Could not get derived parameters, error {}. Full stack trace\n{}:'.format(repr(error[0]), error[1]))
                derived = []
                if ret is not None:
                    derived = ret[1]
                for array in derived:
                    profiles.bestfit.set(array)
                if profiles.bestfit._logposterior not in profiles.bestfit:
                    profiles.bestfit.logposterior = profiles.bestfit[profiles.bestfit._loglikelihood] + profiles.bestfit[profiles.bestfit._logprior]
                profiles.bestfit.logposterior.param.update(derived=True, latex=utils.outputs_to_latex(profiles.bestfit._logposterior))
                list_profiles.append(profiles)
            list_profiles = tm.allreduce(list_profiles)
            self.mpicomm = tm.basecomm

        profiles = Profiles.concatenate(list_profiles)

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles = Profiles.concatenate(self.profiles, profiles)

        attrs = {name: self.mpicomm.bcast(getattr(self.likelihood, name, None), root=0) for name in ['size', 'nvaried', 'ndof', 'hartlap2007_factor', 'percival2014_factor']}
        self.profiles.bestfit.attrs.update(attrs)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def covariance(self, **kwargs):
        """
        Maximize :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.error`  # parabolic errors at best fit (if made available by the profiler)
        - :attr:`Profiles.covariance`  # parameter covariance at best fit (if made available by the profiler).

        """
        if self.profiles is None or 'bestfit' not in self.profiles:
            self.maximize()

        from desilike import Fisher
        fisher = Fisher(self.likelihood, mpicomm=self.mpicomm, **kwargs)
        fisher = fisher(self.profiles.bestfit.choice(index='argmax', input=True, return_type='dict'))

        try:
            covariance = fisher.covariance(params=self.varied_params)
        except np.linalg.LinAlgError:
            covariance = np.full((len(self.varied_params),) * 2, np.nan)
            if self.mpicomm.rank == 0:
                self.log_warning('covariance failed')
        finally:
            covariance = ParameterCovariance(covariance, params=self.varied_params)
            error = Samples([np.full(self.profiles.bestfit.shape, covariance.std(param)) for param in self.varied_params], params=self.varied_params)

            if 'error' not in self.profiles:
                self.profiles.set(error=error)
            self.profiles.set(covariance=covariance)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def interval(self, params=None, cl=1, niterations=1, **kwargs):
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
            all_params = self.varied_params
        else:
            if not is_parameter_sequence(params): params = [params]
            all_params = ParameterCollection([self.varied_params[param] for param in params])

        if self.profiles is None or 'bestfit' not in self.profiles:
            self.maximize(niterations=niterations)
        if 'covariance' not in self.profiles:
            self.covariance()

        chi2, gradient = self._get_vchi2()
        start = self._get_start(niterations=niterations)
        argmax = self.profiles.bestfit.logposterior.argmax()
        center = self._params_backward_transform(self.profiles.bestfit.choice(index=argmax, params=self.transformed_params, return_type='nparray'))
        covariance = self.profiles.covariance.view(params=self.transformed_params) / (self._params_transform_scale[:, None] * self._params_transform_scale)
        state = ProfilerState(chi2=chi2, start=start, varied_params=self.transformed_params, gradient=gradient, fast=True,
                              center=center, covariance=covariance, center_logposterior=self.profiles.bestfit.logposterior[argmax])

        with TaskManager(nprocs_per_task=(self.mpicomm.size + len(all_params) - 1) // len(all_params), use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            list_profiles = []
            for param in tm.iterate(all_params):
                list_profiles.append(self._interval_one(param, state, cl=cl, **kwargs))
            profiles = Profiles()
            for profile in tm.allreduce(list_profiles): profiles.update(profile)
            profiles = _profiles_transform_forward(self, profiles)
            self.mpicomm = tm.basecomm

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles.update(profiles)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def _interval_one(self, param, state, cl=1, xtol=1e-3, **kwargs):
        from scipy.optimize import root_scalar

        grid_params = ParameterCollection([state.varied_params[param]])

        start, center_logposterior, center, covariance = state.start, state.center_logposterior, state.center, state.covariance

        varied_params = state.varied_params - grid_params
        varied_indices = [state.varied_params.index(param) for param in varied_params]  # varied_params ordered as self.varied_params
        grid_indices = [state.varied_params.index(param) for param in grid_params]
        insert_indices = grid_indices - np.arange(len(grid_indices))
        start, center, covariance = start[..., varied_indices], center[grid_indices[0]], covariance[grid_indices[0], grid_indices[0]]
        s = covariance**0.5
        limits = param.prior.limits

        def get_point(z):
            x = z * s + center
            x = np.clip(x, *limits)
            return x

        def chi2(values, point):
            jnp = numpy_jax(values[0])
            values = jnp.asarray(values)
            values = jnp.insert(values, insert_indices, point, axis=-1)
            return self.chi2(values)

        n = 3
        chi2, gradient = self._get_vchi2(chi2=chi2,
                                         start=self._get_start(niterations=n)[..., varied_indices],
                                         aux=dict(point=[get_point(z) for z in self.rng.uniform(0., 1., n)]))

        with TaskManager(nprocs_per_task=(self.mpicomm.size + 2 - 1) // 2, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            interval = []

            def scan(z):
                point = get_point(z)
                ipoint_start = start
                if self._last_profile is not None:
                    best = self._last_profile.bestfit.choice(index='argmax', params=varied_params, return_type='nparray')
                    ipoint_start = (start - np.mean(start, axis=0)) / 10. + best  # center around the previous best fit, with reduced dispersion
                state = ProfilerState(chi2=lambda x: chi2(x, point), start=None, varied_params=varied_params, fast=True)
                if gradient is not None:
                    state.update(gradient=lambda x: gradient(x, point))

                if varied_params:
                    # run with multiple starting points
                    with TaskManager(nprocs_per_task=(self.mpicomm.size + len(ipoint_start) - 1) // len(ipoint_start), use_all_nprocs=True, mpicomm=self.mpicomm) as tm_iter:
                        self.mpicomm = tm_iter.mpicomm
                        list_profiles = []
                        for ss in tm_iter.iterate(ipoint_start):
                            list_profiles.append(self._maximize_one(state.clone(start=ss), **kwargs))
                        profile = Profiles.concatenate(tm_iter.allreduce(list_profiles))
                        self.mpicomm = tm_iter.basecomm
                    try:
                        logposterior = profile.bestfit.logposterior.max()
                        self._last_profile = profile
                    except AttributeError:
                        logposterior = -np.inf
                else:
                    logposterior = -0.5 * chi2(np.array([], dtype='f8'), point)

                return -2. * (logposterior - center_logposterior) - cl

            for sign in tm.iterate([-1, 1]):

                self._last_profile = None

                # find bracket
                a = 0.5
                while scan(sign * a) > 0 and a > 1e-7:
                    a *= 0.5

                if a < 1e-7:
                    interval.append(np.nan)
                    continue

                b = 1.2
                while scan(sign * b) < 0 and b < 8:
                    b *= 1.1

                if b > 8:
                    interval.append(get_point(sign * b))
                    continue

                # low xtol was found to be sufficient in experimental trials
                r = root_scalar(scan, bracket=(sign * a, sign * b), xtol=xtol)
                interval.append(get_point(r.root) if r.converged else np.nan)

                del self._last_profile

            interval = np.array(tm.allreduce(interval)) - center
            interval = Samples([ParameterArray(interval, param=param)])
            profiles = Profiles(interval=interval)
            self.mpicomm = tm.basecomm

        return profiles

    def contour(self, params=None, cl=1, niterations=1, **kwargs):
        """
        Compute 2D contours for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.contour`

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            List of tuples of parameters for which to compute 2D contours.
            If a list of parameters is provided instead, contours are computed for unique tuples of parameters.

        cl : float, int, default=1
            Confidence level for the confidence contour.
            If not set or None, a standard 68.3 % confidence contour is produced.
            If 0 < cl < 1, the value is interpreted as the confidence level (a probability).
            If cl >= 1, it is interpreted as number of standard deviations. For example, cl = 3 produces a 3 sigma contour.

        **kwargs : dict
            Optional arguments for specific profiler.
        """
        if params is None:
            params = self.varied_params
        params = list(params)
        if cl < 1.:
            from scipy import stats
            cl = - stats.norm.ppf(cl / 2.)
        if not is_parameter_sequence(params[0]):
            params = [(param1, param2) for iparam1, param1 in enumerate(params) for param2 in params[iparam1 + 1:]]
        all_params = [(self.varied_params[param1], self.varied_params[param2]) for param1, param2 in params]

        if self.profiles is None or 'bestfit' not in self.profiles:
            self.maximize(niterations=niterations)
        if 'covariance' not in self.profiles:
            self.covariance()

        chi2, gradient = self._get_vchi2()
        start = self._get_start(niterations=niterations)
        argmax = self.profiles.bestfit.logposterior.argmax()
        center = self._params_backward_transform(self.profiles.bestfit.choice(index=argmax, params=self.transformed_params, return_type='nparray'))
        covariance = self.profiles.covariance.view(params=self.transformed_params) / (self._params_transform_scale[:, None] * self._params_transform_scale)
        state = ProfilerState(chi2=chi2, start=start, varied_params=self.transformed_params, gradient=gradient, fast=True,
                              center=center, covariance=covariance, center_logposterior=self.profiles.bestfit.logposterior[argmax])

        with TaskManager(nprocs_per_task=(self.mpicomm.size + len(all_params) - 1) // len(all_params), use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            list_profiles = []
            for params in tm.iterate(all_params):
                list_profiles.append(self._contour_one(params, state, cl=cl, **kwargs))
            profiles = Profiles()
            for profile in tm.allreduce(list_profiles): profiles.update(profile)
            profiles = _profiles_transform_forward(self, profiles)
            self.mpicomm = tm.basecomm

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles.update(profiles)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def _contour_one(self, params, state, size=50, cl=1, xtol=1e-3, **kwargs):
        from scipy.optimize import root_scalar

        grid_params = ParameterCollection([state.varied_params[str(param)] for param in params])
        start, center_logposterior, center, covariance = state.start, state.center_logposterior, state.center, state.covariance

        varied_params = state.varied_params - grid_params
        varied_indices = [state.varied_params.index(param) for param in varied_params]  # varied_params ordered as self.varied_params
        grid_indices = [state.varied_params.index(param) for param in grid_params]
        insert_indices = grid_indices - np.arange(len(grid_indices))
        start, center, covariance = start[..., varied_indices], center[grid_indices], covariance[np.ix_(grid_indices, grid_indices)]
        limits = np.array([param.prior.limits for param in grid_params]).T

        t, u = np.linalg.eig(covariance)
        s = (t * cl) ** 0.5

        def get_point(phi, z):
            r = u @ (z * s[0] * np.cos(phi), z * s[1] * np.sin(phi))
            x1x2 = np.clip(r + center, *limits)
            return x1x2

        def chi2(values, point):
            jnp = numpy_jax(point[0])
            values = jnp.asarray(values)
            values = jnp.insert(values, insert_indices, point, axis=-1)
            return self.chi2(values)

        n = 3
        chi2, gradient = self._get_vchi2(chi2=chi2,
                                         start=self._get_start(niterations=n)[..., varied_indices],
                                         aux=dict(point=[get_point(phi, z) for phi, z in zip(self.rng.uniform(0., 1., n), self.rng.uniform(0., 1., n))]))

        with TaskManager(nprocs_per_task=(self.mpicomm.size + size - 1) // size, use_all_nprocs=True, mpicomm=self.mpicomm) as tm_grid:
            self.mpicomm = tm_grid.mpicomm
            ce = []
            self._last_profile = None

            for phi in tm_grid.iterate(np.linspace(-np.pi, np.pi, size, endpoint=False)):

                def scan(z):
                    point = get_point(phi, z)
                    ipoint_start = start
                    if self._last_profile is not None:
                        transformed_best = self._last_profile.bestfit.choice(index='argmax', params=varied_params, return_type='nparray')
                        ipoint_start = (start - np.mean(start, axis=0)) / 10. + transformed_best  # center around the previous best fit, with reduced dispersion

                    state = ProfilerState(chi2=lambda x: chi2(x, point), start=None, varied_params=varied_params, fast=True)
                    if gradient is not None:
                        state.update(gradient=lambda x: gradient(x, point))
                    if varied_params:
                        # run with multiple starting points
                        with TaskManager(nprocs_per_task=(self.mpicomm.size + len(ipoint_start) - 1) // len(ipoint_start), use_all_nprocs=True, mpicomm=self.mpicomm) as tm_iter:
                            self.mpicomm = tm_iter.mpicomm
                            list_profiles = []
                            for ss in tm_iter.iterate(ipoint_start):
                                list_profiles.append(self._maximize_one(state.clone(start=ss), **kwargs))
                            profile = Profiles.concatenate(tm_iter.allreduce(list_profiles))
                            self.mpicomm = tm_iter.basecomm
                        try:
                            logposterior = profile.bestfit.logposterior.max()
                            self._last_profile = profile
                        except AttributeError:
                            logposterior = -np.inf
                    else:
                        logposterior = -0.5 * chi2(np.array([], dtype='f8'), point)

                    return -2. * (logposterior - center_logposterior) - cl

                # find bracket
                a = 0.5
                while scan(a) > 0 and a > 1e-7:
                    a *= 0.5

                if a < 1e-7:
                    ce.append((np.nan, np.nan))
                    continue

                b = 1.2
                while scan(b) < 0 and b < 8:
                    b *= 1.1

                if b > 8:
                    ce.append(get_point(phi, b))
                    continue

                # low xtol was found to be sufficient in experimental trials
                r = root_scalar(scan, bracket=(a, b), xtol=xtol)
                ce.append(get_point(phi, r.root) if r.converged else (np.nan, np.nan))

            ce = tm_grid.allreduce(ce)
            ce = np.append(ce, ce[:1], axis=0).T
            self.mpicomm = tm_grid.basecomm

            del self._last_profile

        contour = ParameterContours({cl: [tuple(ParameterArray(xx, param, copy=True) for param, xx in zip(grid_params, ce))]})
        profiles = Profiles(contour=contour)
        return profiles

    def grid(self, params=None, grid=None, size=1, cl=2, niterations=1, **kwargs):
        """
        Compute best fits on grid for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.grid`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for the grid.

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
        chi2, gradient = self._get_vchi2()
        start = self._get_start(niterations=niterations)
        profiles = self._grid(_grid_transform_backward(self, grid), ProfilerState(chi2=chi2, start=start, varied_params=self.transformed_params, gradient=gradient, fast=True), **kwargs)
        profiles = _profiles_transform_forward(self, profiles)

        if self.profiles is None:
            self.profiles = Profiles()
        self.profiles.update(profiles)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def _grid(self, grid, state, **kwargs):
        # Internal routine
        grid_params = grid.params()
        nsamples = grid.size
        if nsamples <= 0: raise ValueError('> 0 grid size requested')

        start = state.start
        flat_grid = grid.ravel()
        varied_params = state.varied_params - grid_params
        varied_indices = [state.varied_params.index(param) for param in varied_params]  # varied_params ordered as self.varied_params
        grid_indices = [state.varied_params.index(param) for param in grid_params]
        insert_indices = grid_indices - np.arange(len(grid_indices))
        start = start[..., varied_indices]

        def get_point(ipoint):
            return flat_grid.choice(index=ipoint, params=grid_params, return_type='nparray')

        def chi2(values, point):
            jnp = numpy_jax(point[0])
            values = jnp.asarray(values)
            values = jnp.insert(values, insert_indices, point, axis=-1)
            return self.chi2(values)

        n = 3
        chi2, gradient = self._get_vchi2(chi2=chi2, start=self._get_start(niterations=n)[..., varied_indices],
                                         aux=dict(point=[get_point(ipoint) for ipoint in range(min(flat_grid.size, 3))] * n))

        with TaskManager(nprocs_per_task=(self.mpicomm.size + nsamples - 1) // nsamples, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            last_profile = None
            logposteriors = []
            for ipoint in tm.iterate(range(nsamples)):
                point = get_point(ipoint)
                state = ProfilerState(chi2=lambda x: chi2(x, point), start=None, varied_params=varied_params, fast=True)
                if gradient is not None:
                    state.update(gradient=lambda x: gradient(x, point))
                ipoint_start = start
                if last_profile is not None:
                    transformed_best = last_profile.bestfit.choice(index='argmax', params=varied_params, return_type='nparray')
                    ipoint_start = (start - np.mean(start, axis=0)) / 10. + transformed_best  # center around the previous best fit, with reduced dispersion
                if varied_params:
                     # run with multiple starting points
                    with TaskManager(nprocs_per_task=(self.mpicomm.size + len(ipoint_start) - 1) // len(ipoint_start), use_all_nprocs=True, mpicomm=self.mpicomm) as tm_iter:
                        self.mpicomm = tm_iter.mpicomm
                        list_profiles = []
                        for ss in tm_iter.iterate(ipoint_start):
                            list_profiles.append(self._maximize_one(state.clone(start=ss), **kwargs))
                        profile = Profiles.concatenate(tm_iter.allreduce(list_profiles))
                        self.mpicomm = tm_iter.basecomm
                    try:
                        logposterior = profile.bestfit.logposterior.max()
                        last_profile = profile
                    except AttributeError:
                        logposterior = -np.inf
                else:
                    logposterior = -0.5 * chi2(np.array([], dtype='f8'), point)
                logposteriors.append(logposterior)
            grid.logposterior = np.array(tm.allreduce(logposteriors))
            self.mpicomm = tm.basecomm

        profiles = Profiles(grid=grid)
        return profiles

    def profile(self, params=None, grid=None, size=30, cl=2, niterations=1, **kwargs):
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
        if nparams <= 0: raise ValueError('> 0 params requested')

        grids = []
        for iparam, param in enumerate(params):
            if grid[iparam] is None and np.ndim(cl[iparam]) == 0:
                if self.profiles is None or 'bestfit' not in self.profiles:
                    self.maximize(niterations=niterations)
                if 'covariance' not in self.profiles:
                    self.covariance()
            grids.append(ParameterGrid([_get_grid(self, param, grid=grid[iparam], size=size[iparam], cl=cl[iparam])], params=[param]))

        chi2, gradient = self._get_vchi2()
        start = self._get_start(niterations=niterations)

        with TaskManager(nprocs_per_task=(self.mpicomm.size + nparams - 1) // nparams, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            list_profiles = []
            for iparam, param in tm.iterate(list(enumerate(params))):
                list_profiles.append(self._grid(_grid_transform_backward(self, grids[iparam]), ProfilerState(chi2=chi2, start=start, varied_params=self.transformed_params, gradient=gradient, fast=True)))
            list_profiles = tm.allreduce(list_profiles)
            self.mpicomm = tm.basecomm

        profile = ParameterProfiles([np.column_stack([profiles.grid[param], profiles.grid.logposterior]) for profiles, param in zip(list_profiles, params)], params=params)
        profiles = _profiles_transform_forward(self, Profiles(profile=profile))

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles.update(profiles)

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles