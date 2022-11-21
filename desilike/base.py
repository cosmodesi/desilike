import os
import sys

import numpy as np

from . import mpi
from .utils import BaseClass, NamespaceDict, Monitor, OrderedSet, jax, deep_eq
from .io import BaseConfig
from .parameter import Parameter, ParameterCollection, ParameterCollectionConfig, ParameterArray, Samples


namespace_delimiter = '.'


class PipelineError(Exception):

    """Exception raised when issue with pipeline."""


class Info(NamespaceDict):

    """Namespace/dictionary holding calculator static attributes."""


class BasePipeline(BaseClass):

    def __init__(self, calculator):
        self.calculators = [calculator.runtime_info.initialize()]

        def callback(calculator):
            for require in calculator.runtime_info.requires:
                if require in self.calculators:
                    del self.calculators[self.calculators.index(require)]  # we want first dependencies at the end
                self.calculators.append(require.runtime_info.initialize())
                callback(require)

        callback(self.calculators[0])
        self.calculators = self.calculators[::-1]
        self.derived = None
        self.mpicomm = calculator.mpicomm
        for calculator in self.calculators:
            calculator.runtime_info._param_values = None

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        for calculator in self.calculators:
            calculator.mpicomm = mpicomm

    def _set_params(self, params=None, quiet=False):
        params_from_calculator = {}
        ref_params = ParameterCollection(params)
        params = ParameterCollection()
        for calculator in self.calculators:
            for iparam, param in enumerate(calculator.runtime_info.full_params):
                if param in ref_params:
                    calculator.runtime_info.full_params[iparam] = param = ref_params[param]
                if not quiet and param in params:
                    if param.derived and param.fixed:
                        msg = 'Derived parameter {} of {} is already derived in {}.'.format(param, calculator, params_from_calculator[param.name])
                        if param.basename not in calculator.runtime_info.derived_auto and param.basename not in params_from_calculator[param.name].runtime_info.derived_auto:
                            raise PipelineError(msg)
                        elif self.mpicomm.rank == 0:
                            self.log_warning(msg)
                    elif param != params[param]:
                        raise PipelineError('Parameter {} of {} is different from that of {}.'.format(param, calculator, params_from_calculator[param.name]))
                params_from_calculator[param.name] = calculator
                params.set(param)
        for param in ref_params:
            if param not in params:
                raise PipelineError('Parameter {} is not used by any calculator'.format(param))
        self.derived = None
        self._params = params
        self._varied_params = ParameterCollection([param for param in self._params if param.varied and (not param.derived or param.solved)])

    @property
    def params(self):
        if getattr(self, '_params', None) is None:
            self._set_params()
        return self._params

    @property
    def varied_params(self):
        if getattr(self, '_varied_params', None) is None:
            self._set_params()
        return self._varied_params

    @property
    def param_values(self):
        if getattr(self, '_param_values', None) is None:
            self._param_values = {param.name: param.value for param in self.params}
        return self._param_values

    def eval_params(self, params):
        toret = {}
        all_params = {**self.param_values, **params}
        for param in all_params:
            try:
                toret[param] = self.params[param].eval(**all_params)
            except KeyError:
                pass
        return toret

    def calculate(self, **params):
        for name in params:
            if name not in self.varied_params:
                raise PipelineError('Input parameter is not one of varied parameters: {}'.format(self.varied_params))
        self.param_values.update(params)
        params = self.eval_params(params)
        self.derived = Samples()
        for icalc, calculator in enumerate(self.calculators):  # start by first calculator
            runtime_info = calculator.runtime_info
            # print(calculator.__class__.__name__, runtime_info._param_values)
            runtime_info.set_param_values(params, full=True)
            # print(calculator.__class__.__name__, runtime_info.tocalculate, runtime_info._param_values, params)
            result = runtime_info.calculate()
            for param in runtime_info.full_params:
                if param.depends: self.derived.set(ParameterArray(np.asarray(params[param.name]), param=param))
            self.derived.update(runtime_info.derived)
        return result

    def mpicalculate(self, **params):
        size, cshape = 0, ()
        names = self.mpicomm.bcast(list(params.keys()) if self.mpicomm.rank == 0 else None, root=0)
        for name in names:
            array = None
            if self.mpicomm.rank == 0:
                array = np.asarray(params[name])
                cshape = array.shape
                array = array.ravel()
            params[name] = mpi.scatter(array, mpicomm=self.mpicomm, mpiroot=0)
            size = params[name].size
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
        if not cumsizes[-1]:
            try:
                self.derived = self.derived[:0]
            except (AttributeError, TypeError, IndexError):
                self.derived = Samples()
            return
        mpicomm = self.mpicomm
        states = {}
        for ivalue in range(size):
            self.mpicomm = mpi.COMM_SELF
            self.calculate(**{name: value[ivalue] for name, value in params.items()})
            states[ivalue + cumsizes[mpicomm.rank]] = self.derived
        self.mpicomm = mpicomm
        derived = None
        states = self.mpicomm.gather(states, root=0)
        if self.mpicomm.rank == 0:
            derived = {}
            for state in states:
                derived.update(state)
            derived = Samples.concatenate([derived[i][None, ...] for i in range(cumsizes[-1])]).reshape(cshape, current=len(cumsizes))
        self.derived = derived

    def get_cosmo_requires(self):
        from .cosmo import ExternalEngine
        return ExternalEngine.get_requires(*[getattr(calculator, 'cosmo_requires', {}) for calculator in self.calculators])

    def set_cosmo_requires(self, cosmo):
        for calculator in self.calculators:
            if getattr(calculator, 'cosmo_requires', None):
                calculator.cosmo = cosmo

    def jac(self, getter, params=None):

        if jax is None:
            raise PipelineError('jax is required to compute the Jacobian')

        def fun(params):
            params = self.eval_params(params)
            for calculator in self.calculators:  # start by first calculator, and by the last one
                runtime_info = calculator.runtime_info
                runtime_info.set_param_values(params, full=True)
                #runtime_info._updated_param_values = any(use_jax(value) for value in runtime_info.param_values.values())
                runtime_info._calculate()
            return getter()

        jac = jax.jacfwd(fun, argnums=0, has_aux=False, holomorphic=False)

        if params is None:
            params = self._param_values
        elif not isinstance(params, dict):
            params = {str(param): self._param_values[str(param)] for param in params}

        jac = jac(params)
        fun(params)  # TODO find better fix
        return {k: np.asarray(v) for k, v in jac.items()}

    def _classify_derived_auto(self, calculators=None, niterations=3, seed=42):
        if calculators is None:
            calculators = []
            for calculator in self.calculators:
                if any(kw in getattr(calculator.runtime_info, 'derived_auto', OrderedSet()) for kw in ['.varied', '.fixed']):
                    calculators.append(calculator)

        states = [{} for i in range(len(calculators))]
        rng = np.random.RandomState(seed=seed)
        if calculators:
            for ii in range(niterations):
                params = {str(param): param.ref.sample(random_state=rng) for param in self.params.select(varied=True)}
                self.calculate(**params)
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
                if all(deep_eq(value, values[0]) for value in values):
                    fixed[-1][name] = values[0]
                else:
                    varied[-1].append(name)
                    dtype = np.asarray(values[0]).dtype
                    if not np.issubdtype(dtype, np.inexact):
                        raise ValueError('Attribute {} is of type {}, which is not supported (only float and complex supported)'.format(name, dtype))
        return calculators, fixed, varied

    def _set_derived_auto(self, *args, **kwargs):
        calculators, fixed, varied = self._classify_derived_auto(*args, **kwargs)
        for calculator, fixed_names, varied_names in zip(calculators, fixed, varied):
            derived_names = OrderedSet()
            for derived_name in calculator.runtime_info.derived_auto:
                if derived_name == '.fixed':
                    derived_names |= OrderedSet(fixed_names)
                elif derived_name == '.varied':
                    derived_names |= OrderedSet(varied_names)
                else:
                    derived_names.add(derived_name)
            calculator.runtime_info.derived_auto |= derived_names
            for name in derived_names:
                if name not in calculator.runtime_info.base_params:
                    param = Parameter(name, namespace=calculator.runtime_info.namespace, derived=True)
                    calculator.runtime_info.full_params.set(param)
                    calculator.runtime_info.full_params = calculator.runtime_info.full_params
        self._params = None
        return calculators, fixed, varied

    def _set_speed(self, niterations=10, override=False, seed=42):
        seed = mpi.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=10000)[self.mpicomm.rank]  # to get different seeds on each rank
        rng = np.random.RandomState(seed=seed)
        BasePipeline.run(self)  # to set _derived
        for calculator in self.calculators:
            calculator.runtime_info.monitor.reset()
        for ii in range(niterations):
            params = {str(param): param.ref.sample(random_state=rng) for param in self.params.select(varied=True, solved=False)}
            BasePipeline.run(self, **params)
        if self.mpicomm.rank == 0:
            self.log_info('Found speeds:')
        for calculator in self.calculators:
            if calculator.runtime_info.speed is None or override:
                total_time = self.mpicomm.allreduce(calculator.runtime_info.monitor.get('time', average=False))
                counter = self.mpicomm.allreduce(calculator.runtime_info.monitor.counter)
                if counter == 0:
                    calculator.runtime_info.speed = 1e6
                else:
                    calculator.runtime_info.speed = counter / total_time
                if self.mpicomm.rank == 0:
                    self.log_info('- {}: {:.2f} iterations / second'.format(calculator, calculator.runtime_info.speed))

    def block_params(self, params=None, nblocks=None, oversample_power=0, **kwargs):
        from itertools import permutations, chain
        if params is None: params = self.params.select(varied=True)
        else: params = [self.params[param] for param in params]
        # Using same algorithm as Cobaya
        speeds = [calculator.runtime_info.speed for calculator in self.calculators]
        if any(speed is None for speed in speeds) or kwargs:
            self._set_speed(**kwargs)
            speeds = [calculator.runtime_info.speed for calculator in self.calculators]

        footprints = [tuple(param in calculator.runtime_info.full_params for calculator in self.calculators) for param in params]
        unique_footprints = list(set(row for row in footprints))
        param_blocks = [[p for ip, p in enumerate(params) if footprints[ip] == uf] for uf in unique_footprints]
        param_block_sizes = [len(b) for b in param_blocks]

        def sort_parameter_blocks(footprints, block_sizes, speeds, oversample_power=oversample_power):
            footprints = np.array(footprints, dtype='i4')
            block_sizes = np.array(block_sizes, dtype='i4')
            costs = 1. / np.array(speeds, dtype='f8')
            tri_lower = np.tri(len(block_sizes))
            assert footprints.shape[0] == block_sizes.size

            def get_cost_per_param_per_block(ordering):
                return np.minimum(1, tri_lower.T.dot(footprints[ordering])).dot(costs)

            if oversample_power >= 1:
                orderings = [sort_parameter_blocks(footprints, block_sizes, speeds, oversample_power=1 - 1e-3)[0]]
            else:
                orderings = list(permutations(np.arange(len(block_sizes))))

            permuted_costs_per_param_per_block = np.array([get_cost_per_param_per_block(list(o)) for o in orderings])
            permuted_oversample_factors = (permuted_costs_per_param_per_block[..., [0]] / permuted_costs_per_param_per_block)**oversample_power
            total_costs = np.array([(block_sizes[list(o)] * permuted_oversample_factors[i]).dot(permuted_costs_per_param_per_block[i]) for i, o in enumerate(orderings)])
            argmin = np.argmin(total_costs)
            optimal_ordering = orderings[argmin]
            costs = permuted_costs_per_param_per_block[argmin]
            return optimal_ordering, costs, permuted_oversample_factors[argmin].astype('i4')

        # a) Multiple blocks
        if nblocks is None:
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(unique_footprints, param_block_sizes, speeds, oversample_power=oversample_power)
            sorted_blocks = [param_blocks[i] for i in i_optimal_ordering]
        # b) 2-block slow-fast separation
        else:
            if len(param_blocks) < nblocks:
                raise ValueError('Cannot build up {:d} parameter blocks, as we only have {:d}'.format(nblocks, len(param_blocks)))
            # First sort them optimally (w/o oversampling)
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(unique_footprints, param_block_sizes, speeds, oversample_power=0)
            sorted_blocks = [param_blocks[i] for i in i_optimal_ordering]
            sorted_footprints = np.array(unique_footprints)[list(i_optimal_ordering)]
            # Then, find the split that maxes cost LOG-differences.
            # Since costs are already "accumulated down",
            # we need to subtract those below each one
            costs_per_block = costs - np.append(costs[1:], 0)
            # Split them so that "adding the next block to the slow ones" has max cost
            log_differences = np.zeros(len(costs_per_block) - 1, dtype='f8')  # some blocks are costless (no more parameters)
            nonzero = (costs_per_block[:-1] != 0.) & (costs_per_block[1:] != 0.)
            log_differences[nonzero] = np.log(costs_per_block[:-1][nonzero]) - np.log(costs_per_block[1:][nonzero])
            split_block_indices = np.pad(np.sort(np.argsort(log_differences)[-(nblocks - 1):]) + 1, (1, 1), mode='constant', constant_values=(0, len(param_block_sizes)))
            split_block_slices = list(zip(split_block_indices[:-1], split_block_indices[1:]))
            split_blocks = [list(chain(*sorted_blocks[low:up])) for low, up in split_block_slices]
            split_footprints = np.clip(np.array([np.array(sorted_footprints[low:up]).sum(axis=0) for low, up in split_block_slices]), 0, 1)  # type: ignore
            # Recalculate oversampling factor with 2 blocks
            oversample_factors = sort_parameter_blocks(split_footprints, [len(block) for block in split_blocks], speeds,
                                                       oversample_power=oversample_power)[2]
            # Finally, unfold `oversampling_factors` to have the right number of elements,
            # taking into account that that of the fast blocks should be interpreted as a
            # global one for all of them.
            oversample_factors = np.concatenate([np.full(size, factor, dtype='f8') for factor, size in zip(oversample_factors, np.diff(split_block_slices, axis=-1))])
        return sorted_blocks, oversample_factors


class RuntimeInfo(BaseClass):

    """Information about calculator name, requirements, parameters values at a given step, etc."""

    def __init__(self, calculator, init=None):
        """
        initialize :class:`RuntimeInfo`.

        Parameters
        ----------
        calculator : BaseCalculator
            The calculator this :class:`RuntimeInfo` instance is attached to.
        """
        self.calculator = calculator
        self.derived_auto = OrderedSet()
        self.namespace = None
        self.speed = None
        self.monitor = Monitor()
        self.required_by = set()
        if init is None: init = ((), {})
        self.init = tuple(init)
        self.initialized, self.calculated = False, False
        self._updated_param_values = True

    @property
    def requires(self):
        if getattr(self, '_requires', None) is None:
            self._requires = []
            for name, value in self.calculator.__dict__.items():
                if isinstance(value, BaseCalculator):
                    self._requires.append(value)
            self.requires = self._requires
        return self._requires

    @requires.setter
    def requires(self, requires):
        self._requires = list(requires)
        for require in self._requires:
            require.runtime_info.required_by.add(self.calculator)
        self._pipeline = None

    @property
    def pipeline(self):
        if getattr(self, '_pipeline', None) is None:
            self._pipeline = BasePipeline(self.calculator)
        return self._pipeline

    @property
    def full_params(self):
        if getattr(self, '_full_params', None) is None:
            self._full_params = ParameterCollection(self.calculator.params)
        return self._full_params

    @full_params.setter
    def full_params(self, full_params):
        self._full_params = ParameterCollection(full_params)
        self._base_params = self._solved_params = self._derived_params = self._varied_params = self._param_values = None

    @property
    def base_params(self):
        if getattr(self, '_base_params', None) is None:
            self._base_params = {param.basename: param for param in self.full_params}
        return self._base_params

    @property
    def solved_params(self):
        if getattr(self, '_solved_params', None) is None:
            self._solved_params = self.full_params.select(solved=True)
        return self._solved_params

    @property
    def derived_params(self):
        if getattr(self, '_derived_params', None) is None:
            self._derived_params = self.full_params.select(derived=True, solved=False, depends={})
        return self._derived_params

    @property
    def varied_params(self):
        if getattr(self, '_varied_params', None) is None:
            self._varied_params = ParameterCollection([param for param in self.full_params if (not param.drop) and (param.depends or (not param.derived) or param.solved)])
        return self._varied_params

    @property
    def derived(self):
        if getattr(self, '_derived', None) is None:
            self._derived = Samples()
            if self.derived_params:
                state = self.calculator.__getstate__()
                for param in self.derived_params:
                    name = param.basename
                    if name in state: value = state[name]
                    else: value = getattr(self.calculator, name)
                    self._derived.set(ParameterArray(np.asarray(value), param=param))
        return self._derived

    @property
    def toinitialize(self):
        return not self.initialized

    def initialize(self, **kwargs):
        if self.toinitialize:
            self.clear(initialized=True)
            self.calculator.initialize(*self.init[0], **self.init[1])
        return self.calculator

    @property
    def tocalculate(self):
        return self._updated_param_values or any(require.runtime_info.calculated for require in self.requires)

    def _calculate(self, **params):
        self.set_param_values(params)
        if self.tocalculate:
            self.monitor.start()
            try:
                self.calculator.calculate(**self.param_values)
            except Exception as exc:
                raise PipelineError('Error in method calculate of {}'.format(self.calculator)) from exc
            self.monitor.stop()
            self._derived = None
            self.calculated = True
        else:
            self.calculated = False
        self._updated_param_values = False

    def calculate(self, **params):
        self._calculate(**params)
        return self.calculator.get()

    @property
    def param_values(self):
        if getattr(self, '_param_values', None) is None:
            self._param_values = {param.basename: param.value for param in self.varied_params}
        return self._param_values

    def set_param_values(self, param_values, full=False):
        if full:
            for param, value in param_values.items():
                if param in self.varied_params:
                    basename = self.varied_params[param].basename
                    if self.param_values[basename] != value or type(self.param_values[basename]) is not type(value):
                        self._updated_param_values = True
                    self._param_values[basename] = value
        else:
            for basename, value in param_values.items():
                basename = str(basename)
                if basename in self.param_values:
                    if self._param_values[basename] != value or type(self.param_values[basename]) is not type(value):
                        self._updated_param_values = True
                    self._param_values[basename] = value

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

    def deepcopy(self):
        import copy
        new = self.copy()
        new.full_params = copy.deepcopy(self.full_params)
        return new


class BaseCalculator(BaseClass):

    def __new__(cls, *args, **kwargs):
        cls.info = Info(getattr(cls, 'info', {}))
        cls.params = ParameterCollection(getattr(cls, 'params', None))
        if hasattr(cls, 'config_fn'):
            dirname = os.path.dirname(sys.modules[cls.__module__].__file__)
            config = BaseConfig(os.path.join(dirname, cls.config_fn), index={'class': cls.__name__})
            cls.info = Info({**config.get('info', {}), **cls.info})
            params = ParameterCollectionConfig(config.get('params', {})).init()
            params.update(cls.params)
            init = config.get('init', {})
            if init: kwargs = {**init, **kwargs}
        else:
            params = cls.params.deepcopy()
        new = super(BaseCalculator, cls).__new__(cls)
        new.params = params
        new.runtime_info = RuntimeInfo(new, init=((), kwargs))
        new.mpicomm = mpi.COMM_WORLD
        return new

    def __init__(self, *args, **kwargs):
        if args:
            raise SyntaxError('Provide named arguments')
        self.update(**kwargs)

    def update(self, *args, **kwargs):
        if len(args) == 1:
            kwargs = {**args[0], **kwargs}
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        for name, value in kwargs.items():
            self.runtime_info.init[1][name] = value
        self.runtime_info.initialized = False

    def __call__(self, **params):
        return self.runtime_info.pipeline.calculate(**params)

    def initialize(self):
        pass

    def calculate(self):
        pass

    def get(self):
        return self

    def __getstate__(self):
        return {}
