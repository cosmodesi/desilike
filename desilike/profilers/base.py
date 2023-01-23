import functools

import numpy as np

from desilike import mpi, utils
from desilike.utils import BaseClass
from desilike.samples import load_source
from desilike.samples.profiles import Profiles, Samples, ParameterBestFit
from desilike.parameter import ParameterCollection, is_parameter_sequence
from desilike.samplers.utils import TaskManager


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
        self._original_params = self.mpicomm.bcast(self.varied_params, root=0)

        if rescale:

            self._params_transform_loc = np.array([param.value for param in self.varied_params], dtype='f8')
            self._params_transform_scale = np.diag(covariance)**0.5

            def _params_forward_transform(values):
                return values * self._params_transform_scale + self._params_transform_loc

            def _params_backward_transform(values):
                return (values - self._params_transform_loc) / self._params_transform_scale

            self.varied_params = ParameterCollection()
            for param, loc, scale in zip(self._original_params, self._params_transform_loc, self._params_transform_scale):
                loc, scale = - loc, 1. / scale
                param = param.clone(prior=param.prior.affine_transform(loc=loc, scale=scale),
                                    ref=param.ref.affine_transform(loc=loc, scale=scale),
                                    proposal=param.proposal * scale)
                self.varied_params.set(param)

        else:

            self._params_transform_loc = np.zeros(len(self.varied_params), dtype='f8')
            self._params_transform_scale = np.ones(len(self.varied_params), dtype='f8')

            def _params_forward_transform(values):
                return values

            def _params_backward_transform(values):
                return values

            self.varied_params = self.varied_params.deepcopy()

        self._params_forward_transform = _params_forward_transform
        self._params_backward_transform = _params_backward_transform

        self.save_fn = save_fn

    @bcast_values
    def loglikelihood(self, values):
        values = self._params_forward_transform(values)
        points = Samples(values.T, params=self.varied_params)
        self.pipeline.mpicalculate(**points.to_dict())
        toret = None
        if self.pipeline.mpicomm.rank == 0:
            if self.derived is None:
                self.derived = [points, self.pipeline.derived]
            else:
                self.derived = [Samples.concatenate([self.derived[0], points]),
                                Samples.concatenate([self.derived[1], self.pipeline.derived])]
            toret = self.pipeline.derived[self.likelihood._param_loglikelihood] + self.pipeline.derived[self.likelihood._param_logprior]
        else:
            self.derived = None
        toret = self.pipeline.mpicomm.bcast(toret, root=0)
        mask = np.isnan(toret)
        toret[mask] = -np.inf
        if mask.any() and self.mpicomm.rank == 0:
            self.log_warning('loglikelihood is NaN for {}'.format({k: v[mask] for k, v in points.items()}))
        return toret

    @bcast_values
    def logprior(self, values):
        toret = 0.
        values = self._params_forward_transform(values)
        for param, value in zip(self.varied_params, values.T):
            toret += param.prior(value)
        return toret

    @bcast_values
    def logposterior(self, values):
        toret = self.logprior(values)
        mask = ~np.isinf(toret)
        toret[mask] = self.loglikelihood(values[mask])
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

    def _get_start(self, max_tries=1000):
        if max_tries is None:
            max_tries = self.max_tries

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                if param.ref.is_proper():
                    toret.append(param.ref.sample(size=size, random_state=self.rng))
                else:
                    toret.append([param.value] * size)
            return np.array(toret).T

        logposterior = -np.inf
        for itry in range(max_tries):
            if np.isfinite(logposterior): break
            self.derived = None
            start = np.ravel(get_start(size=1))
            logposterior = self.logposterior(start)

        if not np.isfinite(logposterior):
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(itry))
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

    def _profiles_transform(self, profiles):
        toret = profiles.deepcopy()

        def transform_array(array, scale_only=False):
            try:
                iparam = self._original_params.index(array.param)
            except KeyError:
                return array
            array.param = self._original_params[iparam]
            array = array * self._params_transform_scale[iparam]
            if not scale_only: array += self._params_transform_loc[iparam]
            return array

        for name, item in toret.items():
            if name == 'covariance':
                iparams = [self._original_params.index(param) for param in item._params]
                item._params = self._original_params.sort(key=iparams)
                item._value = item._value * (self._params_transform_scale[iparams, None] * self._params_transform_scale[iparams])
            elif name == 'contour':
                item.data = [tuple(transform_array(array) for array in arrays) for arrays in item.data]
            else: # 'start', 'bestfit', 'error', 'interval', 'profile'
                item.data = [transform_array(array, scale_only=(name == 'error')) for array in item.data]
            toret.set(name=item)
        return toret

    def maximize(self, niterations=None, **kwargs):
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
        if niterations is None: niterations = max(self.mpicomm.size - 1, 1)
        niterations = int(niterations)
        nprocs_per_iteration = max((self.mpicomm.size - 1) // niterations, 1)
        list_profiles = [None] * niterations
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                start, logposterior = self._get_start()
                p = self._maximize_one(start, **kwargs)
                if self.mpicomm.rank == 0:
                    profiles = Profiles(start=Samples(start, params=self.varied_params),
                                        bestfit=ParameterBestFit(list(start) + [logposterior], params=self.varied_params + ['logposterior']))
                    profiles.update(p)
                    profiles = self._profiles_transform(profiles)
                    for param in self.likelihood.params.select(fixed=True, derived=False):
                        profiles.bestfit[param] = np.array(param.value, dtype='f8')
                    index_in_profile, index = self.derived[0].match(profiles.bestfit, params=profiles.start.params())
                    assert index_in_profile[0].size == 1
                    for array in self.derived[1]:
                        profiles.bestfit.set(array[index])
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

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)
        return self.profiles

    def _iterate_over_params(self, params, method, **kwargs):
        nparams = len(params)
        nprocs_per_param = max((self.mpicomm.size - 1) // nparams, 1)
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
                profiles = method(start, param, **kwargs)
                list_profiles[iparam] = self._profiles_transform(profiles) if self.mpicomm.rank == 0 else None
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
        return self._iterate_over_params(params, self._interval_one, **kwargs)

    def profile(self, params=None, **kwargs):
        """
        Compute 1D profiles for :attr:`likelihood`.
        The following attributes are added to :attr:`profiles`:

        - :attr:`Profiles.profile`

        Parameters
        ----------
        params : str, Parameter, list, ParameterCollection, default=None
            Parameters for which to compute 1D profiles.

        **kwargs : dict
            Optional arguments for specific profiler.
        """
        if params is None:
            params = self.varied_params
        else:
            if not is_parameter_sequence(params): params = [params]
            params = ParameterCollection([self.varied_params[param] for param in params])
        return self._iterate_over_params(params, self._profile_one, **kwargs)

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
        return self._iterate_over_params(params, self._contour_one, **kwargs)
