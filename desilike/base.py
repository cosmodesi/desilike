import os
import re
import sys
import copy
import warnings
import traceback

import numpy as np

from . import mpi, jax
from .jax import numpy as jnp
from .utils import BaseClass, UserDict, Monitor, deep_eq, is_sequence
from .io import BaseConfig
from .parameter import Parameter, ParameterCollection, ParameterConfig, ParameterCollectionConfig, ParameterArray, Samples


namespace_delimiter = '.'


class PipelineError(Exception):

    """Exception raised when issue with pipeline."""


class Info(BaseConfig):

    """Namespace/dictionary holding calculator static attributes."""


class InitConfig(BaseConfig):

    """Structure, used internally in the code, holding configuration (passed to :meth:`BaseCalculator.initialize`) and parameters at initialization."""

    _attrs = ['_args', '_params', '_updated_args', '_updated_params', '_args_func_params']  # will be copied

    def __init__(self, *arg, args=None, params=None, **kwargs):
        """
        Initialize :class:`InitConfig`.

        Parameters
        ----------
        params : list, ParameterCollection
            Parameters at initialization.
        """
        self._args = args or tuple()
        self._params = params or ParameterCollection()
        self._updated_args = self._updated_params = True
        self._func_params, self._args_func_params = None, tuple()
        super(InitConfig, self).__init__(*arg, **kwargs)

    def setdefault(self, name, value, if_none=False):
        if if_none:
            if self.get(name, None) is None:
                self[name] = value
        else:
            super().setdefault(name, value)

    @property
    def updated(self):
        """Whether the configuration parameters have been updated (which requires reinitialization of the calculator)."""
        return self._updated_args or (self._updated_params or self.params.updated)

    def _clear(self):
        try:
            calculator = self.runtime_info.calculator
            if not getattr(self.runtime_info, '_initialization', False):
                calculator.__clear__()
        except AttributeError:
            pass

    @updated.setter
    def updated(self, updated):
        """Set the 'updated' status."""
        self._updated_args = self._updated_params = self.params.updated = bool(updated)
        if self._updated_args: self._clear()

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        """Positional arguments to be passed to calculator."""
        self._args = tuple(args)
        self._updated_args = True
        self._clear()

    @property
    def params(self):
        """Parameters."""
        return self._params

    @params.setter
    def params(self, params):
        """Set parameters."""
        self._params = ParameterCollection(params)
        self._updated_params = True

    def __getstate__(self):
        """Return state."""
        return {name: getattr(self, name) for name in ['data'] + self._attrs}

    def _call_func_params(self):
        if self._func_params is not None:
            self._params = self._func_params(self._cls_params.deepcopy(), **{key: self.data[key] for key in self._args_func_params if key in self.data})

    def __setitem__(self, key, item):
        super(InitConfig, self).__setitem__(key, item)
        if key in self._args_func_params: self._call_func_params()
        self._updated_args = True
        self._clear()

    def __delitem__(self, key):
        super(InitConfig, self).__delitem__(key)
        if key in self._args_func_params: self._call_func_params()
        self._updated_args = True
        self._clear()

    def __getitem__(self, key):  # in case mutable
        self._updated_args = True
        return super(InitConfig, self).__getitem__(key)


def _params_args_or_kwargs(args, kwargs):
    if not args: params = kwargs
    elif len(args) == 1 and not kwargs: params = args[0]
    else: raise PipelineError('could not interpret input args = {}, kwargs = {}'.format(args, kwargs))
    return params


def _check_params(params, with_jax=False):
    cshapes = []
    params = dict(params)
    names = list(params.keys())
    for name in names:
        array = params[name]
        if array is not None:
            if with_jax:
                arr = jnp.asarray(array)
                cshapes.append(arr.shape)
                params[name] = arr
            else:
                arr = jax.to_nparray(array)
                if arr is None:
                    raise ValueError('is input array {}: {} a JAX array? if so, use jax.vmap instead'.format(name, array))
                if not arr.shape:
                    raise ValueError('input array {}: {} must be of rank >= 1 for lmap'.format(name, array))
                cshapes.append(arr.shape)
                params[name] = arr.ravel()

    def _all_eq(cshapes):
        if cshapes:
            return all(cshape == cshapes[0] for cshape in cshapes)
        return True

    if not _all_eq(cshapes):
        raise ValueError('input shapes are different: {}'.format(dict(zip(names, cshapes))))

    return params, cshapes[0]


def _concatenate_results(results, shape, add_dims=True):

    def concatenate(results):
        if isinstance(results[0], Samples):
            return Samples.concatenate([result[None, ...] if add_dims else result for result in results]).reshape(shape)
        if results[0] is None:
            return None
        if jax.to_nparray(results[0]) is not None:
            if add_dims:
                results = np.asarray(results)
                results.shape = shape + results[0].shape
            else:
                results = np.concatenate(results)
        else:
            from jax import numpy as jnp
            if add_dims:
                results = jnp.asarray(results)
                results = results.reshape(shape + results[0].shape)
            else:
                results = jnp.concatenate(results)
        return results

    if results:
        if isinstance(results[0], (tuple, list)):
            results = type(results[0])(concatenate([res[i] for res in results]) for i in range(len(results[0])))
        else:
            results = concatenate(results)

    return results


def _check_states(states):
    ref, results, errors = None, [], {}
    for state in states:
        if state[1] is None:  # no error
            ref = state
            break
    for istate, state in enumerate(states):
        if state[1] is None:  # no error
            results.append(state[0])
        else:
            errors[istate] = state[1]
            if ref is not None:
                results.append(ref[0])
            else:
                results.append(None)

    return results, errors


import functools
# Set map routines

def vmap(calculate, backend=None, errors='raise', mpicomm=None, mpi_max_chunk_size=100, **kwargs):

    __wrapped__vmap__ = getattr(calculate, '__wrapped__vmap__', None)
    __wrapped__errors__ = getattr(calculate, '__wrapped__errors__', None)
    errors = str(errors)

    def _calculate_map(params, **kw):
        for value in params.values():
            size = len(value)
            break
        states = []
        for ivalue in range(size):
            state = [None, None]
            try:
                state[0] = calculate({name: value[ivalue] for name, value in params.items()}, **kw)
            except Exception as exc:
                if errors == 'raise':
                    raise exc
                state[1] = (exc, traceback.format_exc())
            finally:
                states.append(state)
        return states

    if backend is None:

        if __wrapped__vmap__ is not None: calculate =  __wrapped__vmap__
        # Standard map, no MPI nor jax
        @functools.wraps(calculate)
        def wrapper(params, **kw):
            kw = {**kwargs, **kw}
            params, shape = _check_params(params)
            states = _calculate_map(params, **kw)
            results, errs = _check_states(states)
            results = _concatenate_results(results, shape, add_dims=True)
            if errors == 'return':
                return results, errs
            return results

    if backend == 'jax':

        if __wrapped__vmap__ is not None: calculate = __wrapped__vmap__

        import jax
        jwrapper = jax.vmap(functools.partial(calculate, **kwargs))

        def wrapper(params, **kw):
            _jwrapper = jwrapper
            if kw: _jwrapper = jax.vmap(functools.partial(calculate, **{**kwargs, **kw}))
            params, shape = _check_params(params, with_jax=True)
            toret = _jwrapper(params)
            if errors == 'return':
                return toret, {}
            return toret

    if backend == 'mpi':

        has_input_mpicomm = mpicomm is not None

        @functools.wraps(calculate)
        def wrapper(params, **kw):
            kw = {**kwargs, **kw}
            if not has_input_mpicomm:
                if __wrapped__vmap__ is None: mpicomm = calculate._mpicomm
                else: mpicomm = __wrapped__vmap__._mpicomm
            if mpicomm.rank == 0:
                params, shape = _check_params(params)
            params, shape = mpicomm.bcast((params, shape) if mpicomm.rank == 0 else None, root=0)
            params = {name: array if mpicomm.rank == 0 else None for name, array in params.items()}
            shape = mpicomm.bcast(shape if mpicomm.rank == 0 else None, root=0)
            all_size = np.prod(shape, dtype='i')

            nchunks = (all_size // mpi_max_chunk_size) + 1
            all_states = []
            for ichunk in range(nchunks):  # divide in chunks to save memory for MPI comm
                chunk_offset = all_size * ichunk // nchunks
                chunk_params = {}
                for name in params:
                    chunk_params[name] = mpi.scatter(params[name][chunk_offset:all_size * (ichunk + 1) // nchunks] if mpicomm.rank == 0 else None, mpicomm=mpicomm, mpiroot=0)
                if not has_input_mpicomm:
                    calculate.mpicomm = mpi.COMM_SELF
                states = []
                error = None
                if __wrapped__vmap__ is None:
                    states = _calculate_map(chunk_params, **kw)
                else:  # calculate already vmap
                    states = calculate(chunk_params, **kw)
                    local_chunk_size = len(chunk_params[name])
                    if local_chunk_size:
                        if __wrapped__errors__ != 'return':
                            states = (states, {})  # adding empty error
                        states = (states[0], {chunk_offset + ierror: error for ierror, error in states[1].items()}, local_chunk_size)
                        for error in states[1].values():
                            break
                        states = [states]
                    else:
                        states = []
                tmp_states = mpicomm.reduce(states, root=0)
                if mpicomm.rank == 0:
                    all_states += tmp_states
                if not has_input_mpicomm:
                    calculate.mpicomm = mpicomm
                if errors == 'raise' and error:
                    raise PipelineError('found error: {}'.format(error))

            if __wrapped__vmap__ is None:
                results, errs = _check_states(all_states)
                results = _concatenate_results(results, shape, add_dims=True)
            else:
                errs = {}
                for states in all_states:
                    errs.update(states[1])
                if errs:  # fill in with results with placeholder
                    ref = None
                    for istates, states in enumerate(all_states):
                        if len(states[1]) != states[2]:
                            if isinstance(states[0], (tuple, list)):
                                ref = type(states[0])(s[:1] if s is not None else None for s in states[0])
                            else:
                                ref = states[0][:1] if states[0] is not None else None
                            break
                    for istates, states in enumerate(all_states):
                        if len(states[1]) == states[2]:  # all errors
                            all_states[istates] = (_concatenate_results([ref] * states[2], (states[2],), add_dims=False),)
                results = _concatenate_results([states[0] for states in all_states], shape, add_dims=False)
            # For MPI ranks != 0, let's put the correct structure for unpacking
            none_result = None
            if mpicomm.rank == 0:
                if isinstance(results, (tuple, list)):
                    none_result = type(results)([None] * len(results))
            none_result = mpicomm.bcast(none_result, root=0)
            if mpicomm.rank != 0:
                results = none_result

            if errors == 'return':
                return results, errs
            return results

    wrapper.__wrapped__vmap__ = calculate if __wrapped__vmap__ is None else __wrapped__vmap__
    wrapper.__wrapped__errors__ = errors

    return wrapper



class BasePipeline(BaseClass):
    """
    Pipeline, used internally in the code, connecting all caclulators up to the calculator that it is attached to
    (:attr:`calculator.runtime_info.pipeline`).
    """
    def __init__(self, calculator):
        """
        Initialize pipeline for input ``calculator``.
        Calculators ``calculator`` depends upon are initialized.
        """
        self.calculators = []
        self.more_derived, self.more_calculate, self.more_initialize = None, None, None

        def callback(calculator):
            self.calculators.append(calculator)
            for require in calculator.runtime_info.requires:
                require.runtime_info.initialize()
                require.runtime_info._initialized_for_pipeline.append(id(self))
                if require in self.calculators:
                    del self.calculators[self.calculators.index(require)]  # we want first dependencies at the end
                callback(require)

        if not getattr(calculator.runtime_info, '_calculation', False):
            calculator.runtime_info.initialized = False  # may depend on the whole pipeline
        callback(calculator.runtime_info.initialize())
        # To avoid loops created by one calculator, which when updated, requests reinitialization of the calculators which depend on it
        for calculator in self.calculators:
            calculator.runtime_info.initialized = True
            #print(calculator, id(self), calculator.runtime_info._initialized_for_pipeline)
        self.calculators = self.calculators[::-1]
        self.mpicomm = calculator._mpicomm
        for calculator in self.calculators:
            calculator.runtime_info.tocalculate = True
            more_initialize = getattr(calculator, 'more_initialize', None)
            if more_initialize is not None: self.more_initialize = more_initialize
        #self._params = ParameterCollection()
        self._set_params()

    def _set_params(self, params=None):
        # Internal method to reset parameters, based on calculator's :class:`BaseCalculator.runtime_info.params`
        params_from_calculator = {}
        params = ParameterCollectionConfig(params, identifier='name')
        new_params = ParameterCollection()
        for calculator in self.calculators:
            calculator_params = ParameterCollection(ParameterCollectionConfig(calculator.runtime_info.params, identifier='name').clone(params))
            for iparam, param in enumerate(calculator.runtime_info.params):
                param = calculator_params[param]
                if param in new_params:
                    if param.derived and param.fixed:
                        msg = 'Derived parameter {} of {} is already derived in {}.'.format(param, calculator, params_from_calculator[param.name])
                        if self.mpicomm.rank == 0: warnings.warn(msg)
                    else:
                        diff = param.__diff__(new_params[param])
                        if diff: # and list(diff) != ['value']:
                            msg = 'Parameter {} of {} is different from that of {}: {}.'.format(param, calculator, params_from_calculator[param.name], diff)
                            if self.mpicomm.rank == 0: warnings.warn(msg)
                params_from_calculator[param.name] = calculator
                #new_calculator_params.set(param)
                new_params.set(param)
            #calculator.runtime_info.params = new_calculator_params
            #for param in new_calculator_params:
            #    if param.basename in calculator.runtime_info.init._params:
            #        calculator.runtime_info.init._params[param.basename] = param.clone(namespace=None)
            #    calculator.runtime_info.init.updated = False
        for param in ParameterCollection(params):
            if any(param.name in p.depends.values() for p in new_params):
                new_params.set(param)
            if param not in new_params:
                raise PipelineError('Cannot attribute parameter {} to any calculator'.format(param))
        self._params = getattr(self, '_params', None) or ParameterCollection()
        for param in self._params:
            if param not in new_params:
                # Add in previous parameters to be dropped
                if any(param.name in p.depends.values() for p in new_params):
                    new_params.set(param)
        self._params = new_params.deepcopy()
        self._params.updated = False
        self._varied_params = self._params.select(varied=True, derived=False)
        self.input_values = {param.name: param.value for param in self._params if param.input or param.depends or param.drop}  # param.drop for depends
        self.derived = Samples()
        self._initialized = False

    @property
    def params(self):
        """Get pipeline parameters."""
        _params = getattr(self, '_params', None)
        if _params is None or _params.updated:
            self._set_params(_params)
        return self._params

    @property
    def varied_params(self):
        """Pipeline parameters that are varied (and not derived)."""
        self.params
        return self._varied_params

    @params.setter
    def params(self, params):
        """Set pipeline parameters."""
        self._set_params(params)

    @property
    def mpicomm(self):
        """MPI communicator."""
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        """Set MPI communicator."""
        self._mpicomm = mpicomm
        for calculator in self.calculators:
            calculator._mpicomm = mpicomm

    def calculate(self, *args, force=None, return_derived=False, **kwargs):
        """
        Calculate, i.e. call calculators' :meth:`BaseCalculator.calculate` if their parameters are updated,
        or if they depend on previous calculation that has been updated.
        Derived parameter values are stored in :attr:`derived`.
        """
        params = _params_args_or_kwargs(args, kwargs)
        if not self._initialized:
            if self.more_initialize is not None: self.more_initialize()
            self._initialized = True
            for calculator in self.calculators: calculator.runtime_info.tocalculate = True
        names = list(params.keys())
        self_params = self.params
        for name in names:
            if name not in self_params:
                raise PipelineError('input parameter {} is not one of parameters: {}'.format(name, self_params))

        bak_input_values = dict(self.input_values)
        self.input_values.update(params)
        params = self_params.eval(**self.input_values)
        # Here we updated self.input_values as we need to access it (in e.g. BaseLikelihood._solve)
        self.input_values.update(params)  # to update parameters with depends
        result, self.derived = None, (Samples() if return_derived else None)
        if self.derived is not None:
            for param in self._params:
                if param.depends:
                    self.derived.set(ParameterArray(params[param.name], param=param))

        for calculator in self.calculators:  # start by first calculator
            runtime_info = calculator.runtime_info
            derived = Samples()
            try:
                result = runtime_info.calculate(params, force=force)
                if self.derived is not None:
                    derived = runtime_info.derived
            except Exception:  # we want to keep track of the Exception class, so do not raise PipelineError
                self.log_debug('error in method calculate of {} with calculator parameters {} and pipeline parameters {}'.format(calculator, runtime_info.input_values, self.input_values))
                raise
            if self.derived is not None:
                self.derived.update(derived)
        if self.more_calculate:
            toret = self.more_calculate()
            if toret is not None: result = toret
        if self.more_derived and self.derived is not None:
            tmp = self.more_derived()
            if tmp is not None: self.derived.update(tmp)
        # Now we update self.input_values only with non-traced arrays
        for name, value in self.input_values.items():
            value = jax.to_nparray(value)
            if value is not None: bak_input_values[name] = value
        self.input_values = bak_input_values
        if return_derived:
            return result, self.derived
        return result

    def get_cosmo_requires(self):
        """Return a dictionary mapping section to method's name and arguments,
        e.g. 'background': {'comoving_radial_distance': {'z': z}}."""
        from .cosmo import BaseExternalEngine
        return BaseExternalEngine.get_requires(*[getattr(calculator, 'cosmo_requires', {}) for calculator in self.calculators])

    def set_cosmo_requires(self, cosmo):
        """Set input :class:`cosmoprimo.Cosmology` instance for bindings."""
        for calculator in self.calculators:
            cosmo_requires = getattr(calculator, 'cosmo_requires', {})
            if cosmo_requires:
                conversions = {'m_ncdm': 'm_ncdm_tot'}  # 'logA': 'ln10^10A_s'
                cosmo_params = cosmo_requires.get('params', {})
                if cosmo_params:
                    for basename, name in calculator.runtime_info.base_names.items():
                        if basename in cosmo_params:
                            #print(name, cosmo[conversions.get(basename, basename)])
                            value = cosmo[conversions.get(basename, basename)]
                            if basename in calculator.runtime_info.input_values:
                                calculator.runtime_info.input_values[basename] = value
                            if basename in self.input_values:
                                self.input_values[name] = value
                #if set(cosmo_requires.keys()) != {'params'}:  # requires a :class:`cosmoprimo.Cosmology` instance as ``cosmo`` attribute
                calculator.cosmo = cosmo
                calculator.runtime_info.tocalculate = True

    def _classify_derived(self, calculators=None, niterations=3, seed=42):
        """
        Internal method to classify calculators' derived parameters as
        "fixed" (they do not vary when parameters are changed) or "varied" (they vary when parameters are changed)

        Parameters
        ----------
        calculators : list, default=None
            List of calculators for which to classify derived parameters,
            as well as quantities returned by their :meth:`BaseCalculator.__getstate__` method.

        niterations : int, default=3
            To test whether derived parameters are fixed or vary, the pipeline is run ``niterations`` times,
            with varied parameters randomly varied (within their :attr:`Parameter.ref` reference distribution).

        seed : int, default=42
            Random seed, used to sample varied parameters within their reference distribution.

        Returns
        -------
        calculators : list
            List of calculators.

        fixed : list
            List of dictionaries (one for each calculator) mapping names of derived quantities with their (constant) values.

        varied : list
            List of list (one for each calculator) of derived quantities which vary.
        """
        if niterations < 1:
            raise ValueError('Need at least 1 iteration to classify between fixed and varied parameters')
        if calculators is None:
            calculators = self.calculators

        states = [{} for i in range(len(calculators))]
        rng = np.random.RandomState(seed=seed)
        input_values = {param.name: self.input_values[param.name] for param in self.varied_params}
        if calculators:
            for params in [{str(param): param.ref.sample(random_state=rng) for param in self.varied_params} for ii in range(niterations)] + [input_values]:
                self.calculate(params)
                for calculator, state in zip(calculators, states):
                    calcstate = calculator.__getstate__()
                    for name, value in calcstate.items():
                        state[name] = state.get(name, []) + [value]
                    for param in calculator.runtime_info.derived_params:
                        name = param.basename
                        if name not in calcstate:
                            state[name] = state.get(name, []) + [getattr(calculator, name)]

        fixed, varied = [], []
        for calculator, state in zip(calculators, states):
            fixed.append({})
            varied.append([])
            for name, values in state.items():
                try:
                    eq = all(deep_eq(value, values[0]) for value in values)
                except Exception as exc:
                    raise ValueError('Unable to check equality of {} (type: {})'.format(name, type(values[0]))) from exc
                if eq:
                    fixed[-1][name] = values[0]
                else:
                    varied[-1].append(name)
                    dtype = np.asarray(values[0]).dtype
                    if not np.issubdtype(dtype, np.inexact):
                        raise ValueError('Attribute {} is of type {}, which is not supported (only float and complex supported)'.format(name, dtype))
        return calculators, fixed, varied

    def _set_derived(self, calculators, params):
        """
        Internal method to set derived parameters.

        Parameters
        ----------
        calculators : list
            List of calculators for which to set derived parameters.

        params : list
            List of :class:`ParameterCollection` to set as derived parameters, for each input calculator.
        """
        for calculator, params in zip(calculators, params):
            for param in params:
                if param not in self.varied_params:
                    #if hasattr(param, 'setdefault'):
                    #    param = param.copy()
                    #    param.setdefault('namespace', calculator.runtime_info.namespace)
                    param = Parameter(param).clone(derived=True)
                    for dparam in calculator.runtime_info.derived_params.names(basename=param.basename):
                        param = calculator.runtime_info.params[dparam].clone(namespace=param.namespace)
                        break
                    calculator.runtime_info.params.set(param)
            calculator.runtime_info.params = calculator.runtime_info.params
        self._set_params(params=self.params)

    def _set_speed(self, niterations=10, override=False, seed=42):
        """
        Internal method to compute and set calculators' speed (i.e. inverse of run time for one :meth:`BaseCalculator.calculate` call).

        Parameters
        ----------
        niterations : int, default=10
            To compute (average) execution time, the pipeline is run ``niterations`` times,
            with varied parameters randomly varied (within their :attr:`Parameter.ref` reference distribution).

        override : bool, default=False
            If ``False``, and :attr:`BaseCalculator.runtime_info.speed` is not ``None``, it is left untouched.
            Else, :attr:`BaseCalculator.runtime_info.speed` is set to measured speed (as 1 / (average execution time)).

        seed : int, default=42
            Random seed, used to sample varied parameters within their reference distribution.
        """
        seed = mpi.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=10000)[self.mpicomm.rank]  # to get different seeds on each rank
        rng = np.random.RandomState(seed=seed)
        self.calculate()  # to set _derived
        for calculator in self.calculators:
            calculator.runtime_info.monitor.reset()
        for ii in range(niterations):
            params = {str(param): param.ref.sample(random_state=rng) for param in self.params.select(varied=True, solved=False)}
            self.calculate(**params)
        if self.mpicomm.rank == 0:
            self.log_info('Found speeds:')
        total = 0.
        for calculator in self.calculators:
            if calculator.runtime_info.speed is None or override:
                total_time = self.mpicomm.allreduce(calculator.runtime_info.monitor.get('time', average=False))
                counter = self.mpicomm.allreduce(calculator.runtime_info.monitor.counter)
                if counter == 0:
                    calculator.runtime_info.speed = 1e6
                else:
                    calculator.runtime_info.speed = counter / total_time
                total += 1. / calculator.runtime_info.speed
                if self.mpicomm.rank == 0:
                    self.log_info('- {}: {:.2f} iterations / second - {:.3f} s / iteration'.format(calculator, calculator.runtime_info.speed, 1. / calculator.runtime_info.speed))
        if self.mpicomm.rank == 0:
            self.log_info('- total speed: {:.2f} iterations / second - {:.4f} s / iteration'.format(1. / total, total))


class RuntimeInfo(BaseClass):
    """
    Store information about calculator name, requirements, parameters values at a given step, etc.

    Attributes
    ----------
    calculator : BaseCalulator
        Calculator this is attached to, as :attr:`BaseCalculator.runtime_info`.

    speed : float
        Inverse of number of iterations per second.
    """
    installer = None

    def __init__(self, calculator, init=None):
        """
        initialize :class:`RuntimeInfo`.

        Parameters
        ----------
        calculator : BaseCalculator
            The calculator this :class:`RuntimeInfo` instance is attached to.

        init : InitConfig, default=None
            Configuration at initialization.
        """
        self.calculator = calculator
        self.namespace = None
        self.speed = None
        self.monitor = Monitor()
        if init is None: init = InitConfig()
        self.init = init
        if not isinstance(init, InitConfig):
            self.init = InitConfig(init)
        self._initialized = False
        self._initialized_for_pipeline = []
        self._tocalculate = True
        self.calculated = False
        self.name = self.calculator.__class__.__name__
        self._initialize_with_namespace = False
        self._calculate_with_namespace = False
        self.params = ParameterCollection(init.params)
        self.init.runtime_info = self

    def install(self):
        """Install calculator, called by :class:`install.Installer`."""
        if self.installer is not None:
            try:
                func = self.calculator.install
            except AttributeError:
                return
            func(self.installer)
            self.installer.setenv()

    @property
    def params(self):
        """Return parameters specific to this calculator."""
        if self._params.updated: self.params = self._params
        return self._params

    @params.setter
    def params(self, params):
        """Set parameters specific to this calculator."""
        self._params = ParameterCollection(params)
        self._params.updated = False
        self.base_names = {(param.name if self._calculate_with_namespace else param.basename): param.name for param in self.params}
        self.input_names = {param.name: (param.name if self._calculate_with_namespace else param.basename) for param in self.params.select(input=True)}
        self.input_values = {(param.name if self._calculate_with_namespace else param.basename): param.value for param in self.params.select(input=True)}
        self.derived_params = self.params.select(derived=True)
        self._tocalculate = True

    @property
    def derived(self):
        """Return derived parameter values."""
        if getattr(self, '_derived', None) is None:
            self._derived = Samples()
            if self.derived_params:
                state = self.calculator.__getstate__()
                for param in self.derived_params:
                    name = param.basename
                    if name in state: value = state[name]
                    else: value = getattr(self.calculator, name)
                    array = ParameterArray(value, param=param)
                    array.param._shape = array.shape  # a bit hacky, but no need to update parameters for this...
                    self._derived.set(array)
        return self._derived

    @property
    def pipeline(self):
        """Return pipeline for this calculator."""
        if getattr(self, '_pipeline', None) is None or not self.initialized:
            self._pipeline = BasePipeline(self.calculator)
        else:
            for calculator in self._pipeline.calculators[:-1]:
                if (not calculator.runtime_info.initialized) or (id(self._pipeline) not in calculator.runtime_info._initialized_for_pipeline):
                    #print(calculator.runtime_info.initialized, calculator, id(self._pipeline), calculator.runtime_info._initialized_for_pipeline)
                    self._pipeline = BasePipeline(self.calculator)
                    break
        return self._pipeline

    @property
    def requires(self):
        """
        Return set of calculators this calculator directly depends upon.
        If not set, defaults to the :class:`BaseCalculator` instances in this calculator's ``__dict__``.
        """
        if getattr(self, '_requires', None) is None:
            if getattr(self, '_initialization', False): return []
            self.initialized = False
            self.initialize()
        return self._requires

    @requires.setter
    def requires(self, requires):
        """Set list of calculators this calculator depends upon."""
        self._requires = list(requires)
        self.initialized = False

    @property
    def initialized(self):
        """Has this calculator been initialized?"""
        if self.init.updated:
            self._initialized = False
            self._initialized_for_pipeline.clear()
        return self._initialized

    @initialized.setter
    def initialized(self, initialized):
        if initialized:
            self.init.updated = False
        self._initialized = initialized

    def initialize(self):
        """Initialize calculator (if not already initialized), calling :meth:`BaseCalculator.initialize` with :attr:`init` configuration."""
        if not self.initialized:
            self.clear()
            self._initialization = True   # to avoid infinite loops
            self.calculator.__clear__()
            self.install()
            bak = self.init.params
            params_with_namespace = ParameterCollection(self.init.params).deepcopy()
            self._initialize_with_namespace = getattr(self.calculator, '_initialize_with_namespace', False)
            self._calculate_with_namespace = getattr(self.calculator, '_calculate_with_namespace', False)
            if not self._initialize_with_namespace:
                params_basenames = params_with_namespace.basenames()
                # Pass parameters without namespace
                self.params = self.init.params = params_with_namespace.clone(namespace=None)
            else:
                self.params = self.init.params = params_with_namespace
            try:
                self.calculator.initialize(*self.init.args, **self.init)
            except Exception as exc:
                raise PipelineError('Error in method initialize of {}'.format(self.calculator)) from exc
            if not self._initialize_with_namespace:
                for param in self.init.params:
                    if param.basename in params_basenames:  # update namespace
                        param.update(namespace=params_with_namespace[params_basenames.index(param.basename)].namespace)
            self.params = self.init.params
            self.init.params = bak
            self.initialized = True
            self._initialized_for_pipeline = []
            self._initialization = False
            if getattr(self, '_requires', None) is None:
                self._requires = []
                for name, value in self.calculator.__dict__.items():
                    # never use set() when order may matter for MPI'ed code...
                    if isinstance(value, BaseCalculator) and value not in self._requires:
                        self._requires.append(value)
        return self.calculator

    @property
    def tocalculate(self):
        """Should calculator's :class:`BaseCalculator.calculate` be called?"""
        return self._tocalculate or any(require.runtime_info.calculated for require in self.requires) or not hasattr(self, '_get')

    @tocalculate.setter
    def tocalculate(self, tocalculate):
        self._tocalculate = tocalculate

    def calculate(self, params, force=None):
        """
        If calculator's :class:`BaseCalculator.calculate` has not be called with input parameter values, call it,
        keeping track of running time with :attr:`monitor`.
        """
        self.params
        #print('calculate', force, type(self.calculator), self.tocalculate, self._tocalculate, any(require.runtime_info.calculated for require in self.requires))
        for name, value in params.items():
            name = str(name)
            if name in self.input_names:
                invalue = jax.to_nparray(value)
                basename = self.input_names[name]
                if force is not None:
                    self._tocalculate = force
                elif invalue is None:
                    if value is not self.input_values[basename]:  # jax
                        self._tocalculate = True
                else:
                    if type(invalue) != type(self.input_values[basename]) or invalue != self.input_values[basename]:
                        self._tocalculate = True
                        #print(self.calculator, invalue, self.input_values[basename], type(invalue), type(self.input_values[basename]), invalue == self.input_values[basename])
                if invalue is not None:
                    value = invalue
                self.input_values[basename] = value
        if self.tocalculate:
            #print(self.calculator, self.input_values)
            self._calculation = True
            self.monitor.start()
            self.calculator.calculate(**self.input_values)
            self._derived = None
            self.calculated = True
            self._get = self.calculator.get()
            self.monitor.stop()
            self._calculation = False
        else:
            self.calculated = False
        self._tocalculate = False
        return self._get

    def __getstate__(self):
        """Return this class state dictionary."""
        return self.__dict__.copy()

    def clear(self, **kwargs):
        calculator, init = self.calculator, self.init
        self.__dict__.clear()
        self.__init__(calculator, init=init)
        self.update(**kwargs)

    def update(self, *args, **kwargs):
        """Update with provided :class:`RuntimeInfo` instance of dict."""
        state = self.__getstate__()
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update(args[0].__getstate__())
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        state.update(kwargs)
        for name, value in state.items():
            setattr(self, name, value)  # this is to properly update properties with setters

    def clone(self, *args, **kwargs):
        """Clone, i.e. copy and update."""
        new = self.copy()
        new.update(*args, **kwargs)
        return new


class BaseCalculator(BaseClass):
    """
    Base calculator class, to be extended by any calculator, which will typically redefine:

    - :meth:`initialize`: set meta parameters and other calculators it depends on
    - :meth:`calculate`: takes in parameter values, and do some calculation
    - :meth:`get`: returns the quantity of interest

    """
    def __new__(cls, *args, **kwargs):
        cls_info = Info(getattr(cls, '_info', {}))
        cls_init = InitConfig(data=getattr(cls, '_init', {}))
        func_params, args_func_params = None, tuple()
        params = getattr(cls, '_params', None)
        if callable(params):
            import inspect
            func_params = params
            sig = inspect.signature(params)
            args_func_params = tuple(param.name for param in sig.parameters.values() if param.name not in ['self', 'params'])
            cls_params = ParameterCollection()
        else:
            cls_params = ParameterCollection(params)
        if getattr(cls, 'config_fn', None):
            dirname = os.path.dirname(sys.modules[cls.__module__].__file__)
            config = BaseConfig(os.path.join(dirname, cls.config_fn), index={'class': cls.__name__})
            cls_info = Info({**config.get('info', {}), **cls_info})
            params = ParameterCollectionConfig(config.get('params', {})).init()
            params.update(cls_params)
            init = InitConfig(config.get('init', {}))
            init.update(cls_init)
            init.update(kwargs)
        else:
            init = cls_init.deepcopy()
            params = cls_params.deepcopy()
        new = super(BaseCalculator, cls).__new__(cls)
        new.info = cls_info
        init._cls_params = params.deepcopy()
        init._args_func_params = args_func_params
        init._func_params = func_params
        init._params = params
        new.runtime_info = RuntimeInfo(new, init=init)
        new._mpicomm = mpi.COMM_WORLD
        init._call_func_params()
        return new

    def __init__(self, *args, **kwargs):
        self.init.args = args
        self.init.update(**kwargs)

    @property
    def mpicomm(self):
        return self._mpicomm
        #if not self.runtime_info.initialized:
        #    return self._mpicomm
        #return self.runtime_info.pipeline.mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        if not self.runtime_info.initialized:
            self._mpicomm = mpicomm
        self.runtime_info.pipeline.mpicomm = mpicomm

    @property
    def init(self):
        """Return configuration at initialization."""
        return self.runtime_info.init

    def __call__(self, *args, **kwargs):
        """Take all parameters as input, calculate, and return the result of :attr:`get`"""
        return self.runtime_info.pipeline.calculate(*args, **kwargs)

    def initialize(self, **kwargs):
        # Define this method, with takes meta parameters as input. Parameters can be accessed through self.params.
        pass

    def calculate(self, **params):
        # Define this method, which takes parameter values as input.
        pass

    def get(self):
        """Return quantity of main interest, e.g. loglikelihood + logprior if ``self`` is a likelihood."""
        return None

    def __getattr__(self, name):
        if not getattr(self.runtime_info, '_initialization', False):
            self.runtime_info.initialize()
        try:
            return object.__getattribute__(self, name)
        except AttributeError as exc:
            raise AttributeError('calculator {} has no attribute {}; '
                                 'have you run any calculation already by calling this calculator or calculators '
                                 'that depend on it (typically, a likelihood?)'.format(self.__class__.__name__, name)) from exc

    def __getstate__(self):
        """
        Return this class' state dictionary.
        To be able to emulate this calculator, it should return all the quantities that can then be used by any other calculator.
        """
        return {}

    #def __repr__(self):
    #    """Return string representation, i.e. calculator's name."""
    #    return self.runtime_info.name

    def __copy__(self):
        """
        Copy this calculator only (not the calculators it may depend on).

        >>> calculator2 = calculator1.copy()
        # calculator2 will call calculator1 dependencies
        """
        new = object.__new__(self.__class__)
        new._mpicomm = self._mpicomm
        #new.__dict__.update(self.__dict__)  # this is problematic, we should remove every but non-standard attributes
        for name in ['info', 'runtime_info']:
            setattr(new, name, getattr(self, name).copy())
        new.runtime_info.calculator = new
        if new.runtime_info.initialized:
            new.runtime_info.clear(params=self.runtime_info.params.deepcopy(),
                                   _requires=self.runtime_info.requires.copy(),
                                   _initialized=True)
            if getattr(self.runtime_info, '_pipeline', None) is not None:
                new.runtime_info.pipeline._set_params(self.runtime_info.pipeline.params.deepcopy())  # to preserve depends
                new(**self.runtime_info.pipeline.input_values)
        else:
            new.runtime_info.clear()
        return new

    def __deepcopy__(self, memo):
        new = object.__new__(self.__class__)
        new._mpicomm = self._mpicomm
        #new.__dict__.update(self.__dict__)  # this is problematic, we should remove every but non-standard attributes
        for name in ['info', 'runtime_info']:
            setattr(new, name, getattr(self, name).copy())
        memo[id(self)] = new
        new.info = copy.deepcopy(self.info)
        new.runtime_info = self.runtime_info.copy()
        new.runtime_info.calculator = new
        new.runtime_info.init = copy.deepcopy(self.runtime_info.init)
        new.runtime_info.clear()
        if self.runtime_info.initialized:
            # Let's reinitialize, other we'd need to replace references to calculator dependencies in each calculator
            new.runtime_info.initialize()
            new.runtime_info.params = self.runtime_info.params.deepcopy()
            if getattr(self.runtime_info, '_pipeline', None) is not None:  # no need if self.runtime_info.pipeline isn't created
                params = ParameterCollection([param.copy() for param in self.runtime_info.pipeline.params if param.derived is not True])
                new.runtime_info.pipeline._set_params(params)  # to preserve depends
                new.runtime_info.pipeline.input_values = dict(self.runtime_info.pipeline.input_values)
        return new

    def deepcopy(self):
        """
        Copy the calculator and full pipeline:

        >>> calculator2 = calculator1.deepcopy()
        # calculator2 lives independently from calculator1
        """
        return copy.deepcopy(self)

    @property
    def params(self):
        """This calculator's specific parameters."""
        #if not self.runtime_info.initialized:
        return self.runtime_info.init.params
        #return self.runtime_info.params

    @params.setter
    def params(self, params):
        """
        Set this calculator's specific parameters; which triggers an automatic call to :meth:`initialize`
        before :meth:`calculate`.
        """
        self.runtime_info.init.params = ParameterCollection(params)

    @property
    def all_params(self):
        """All pipeline parameters."""
        return self.runtime_info.pipeline.params

    @all_params.setter
    def all_params(self, all_params):
        """Set all pipeline parameters."""
        self.runtime_info.pipeline.params = all_params

    @property
    def varied_params(self):
        """Varied pipeline parameters."""
        return self.runtime_info.pipeline.varied_params

    def __clear__(self):
        """Clear instance attributes and ``jax.jit`` cache."""
        self.__dict__ = {name: self.__dict__[name] for name in ['info', 'runtime_info', '_mpicomm'] if name in self.__dict__}
        for funcname in dir(self.__class__):
            # to recompile jit
            getattr(getattr(self.__class__, funcname, None), 'clear_cache', lambda: None)()


class CollectionCalculator(BaseCalculator):

    def initialize(self, calculators=None):
        if hasattr(calculators, 'items'):
            calculators = dict(calculators.items())
        else:
            calculators = {str(i): calc for i, calc in enumerate(calculators)}
        self.names = list(calculators.keys())
        self.calculators = list(calculators.values())
        for calculator in self.calculators:
            if calculator.runtime_info.initialized:
                for param in calculator.all_params:
                    for depname in param.depends.values():
                        param = calculator.all_params[depname]
                        if not any(param in calc.runtime_info.params for calc in calculator.runtime_info.pipeline.calculators):
                            self.init.params[param] = param.clone(drop=True)  # add parameter to this calculator
        self.all_calculators = {name: list(calculator.runtime_info.pipeline.calculators) for name, calculator in zip(self.names, self.calculators)}
        self.all_derived = {}
        for name, calculators in self.all_calculators.items():
            self.all_derived[name] = {}
            for calculator in calculators:
                for param in calculator.runtime_info.derived_params:
                    self.all_derived[name][param.name] = calculator
                    self.params.set(param.clone(name='{}_{}'.format(name, param.name)))
        self.runtime_info.requires = self.calculators

    def __getitem__(self, name):
        try:
            return self.calculators[name]
        except TypeError:
            return self.calculators[self.names.index(name)]

    def __setitem__(self, name, calculator):
        try:
            self.calculators[name] = calculator
        except TypeError:
            self.calculators[self.names.index(name)] = calculator

    def __len__(self):
        return len(self.calculators)

    def __iter__(self):
        return iter(self.calculators)

    def items(self):
        return list(zip(self.names, self.calculators))

    def __getattr__(self, name):
        match = re.match(r'(\d*)_(.*)', name)
        if match:
            calcname, basename = match.group(1), match.group(2)
            if calcname in self.names:
                try:
                    return getattr(self[calcname], basename)
                except AttributeError:
                    return self.all_derived[calcname][basename].runtime_info.derived[basename]
        raise AttributeError('calculator {} has no attribute {};'
                             'have you run any calculation already by calling this calculator or calculators'
                             'that depend on it (typically, a likelihood?)'.format(self.__class__.__name__, name))

    def __getstate__(self):
        state = {}
        for calcname, calculator in zip(self.names, self.calculators):
            for key, value in calculator.__getstate__().items():
                state['{}_{}'.format(calcname, key)] = value
        return state
