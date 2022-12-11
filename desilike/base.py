import os
import sys
import warnings

import numpy as np

from . import mpi
from .utils import BaseClass, NamespaceDict, Monitor, OrderedSet, jax, deep_eq
from .io import BaseConfig
from .parameter import Parameter, ParameterCollection, ParameterConfig, ParameterCollectionConfig, ParameterArray, Samples


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
            calculator.runtime_info.tocalculate = True
        self._set_params()

    def _set_params(self, params=None):
        params_from_calculator = {}
        params = ParameterCollectionConfig(params)
        self._params = ParameterCollection()
        for calculator in self.calculators:
            for iparam, param in enumerate(calculator.runtime_info.params):
                if param in params:
                    calculator.runtime_info.params[iparam] = param = param.clone(params[param])
                if param in self._params:
                    if param.derived and param.fixed:
                        msg = 'Derived parameter {} of {} is already derived in {}.'.format(param, calculator, params_from_calculator[param.name])
                        if param.basename not in calculator.runtime_info.derived_auto and param.basename not in params_from_calculator[param.name].runtime_info.derived_auto:
                            raise PipelineError(msg)
                        elif self.mpicomm.rank == 0:
                            warnings.warn(msg)
                    elif param != params[param]:
                        raise PipelineError('Parameter {} of {} is different from that of {}.'.format(param, calculator, params_from_calculator[param.name]))
                params_from_calculator[param.name] = calculator
                self._params.set(param)
        self.derived = None
        for param in params:
            if param not in self._params:
                raise PipelineError('Cannot attribute parameter {} to any calculator'.format(param))
        self.varied_params = self._params.select(varied=True, derived=False)
        self.param_values = {param.name: param.value for param in self._params}

    @property
    def params(self):
        if getattr(self, '_params', None) is None:
            self._set_params()
        return self._params

    @params.setter
    def params(self, params):
        self._set_params(params)

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        for calculator in self.calculators:
            calculator.mpicomm = mpicomm

    @property
    def tocalculate(self):
        return any(calculator.runtime_info.tocalculate for calculator in self.calculators)

    @tocalculate.setter
    def tocalculate(self, tocalculate):
        for calculator in self.calculators:
            calculator.runtime_info.tocalculate = True

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
            if name not in self.params:
                raise PipelineError('Input parameter {} is not one of parameters: {}'.format(name, self.params))
        self.param_values.update(params)
        params = self.eval_params(params)
        self.derived = Samples()
        for icalc, calculator in enumerate(self.calculators):  # start by first calculator
            runtime_info = calculator.runtime_info
            # print(calculator.__class__.__name__, runtime_info._param_values)
            runtime_info.set_param_values(params, full=True)
            # print(calculator.__class__.__name__, runtime_info.tocalculate, runtime_info._param_values, params)
            result = runtime_info.calculate()
            self.derived.update(runtime_info.derived)
        for param in self.params:
            if param.depends: self.derived.set(ParameterArray(np.asarray(params[param.name]), param=param))
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
            cosmo_requires = getattr(calculator, 'cosmo_requires', {})
            if cosmo_requires:
                cosmo_params = cosmo_requires.get('params', {})
                if cosmo_params:
                    for basename, param in calculator.runtime_info.base_params.items():
                        if basename in cosmo_params:
                            self.param_values[param.name] = calculator.runtime_info.param_values[basename] = cosmo[basename]
                if set(cosmo_requires.keys()) != {'params'}:
                    calculator.cosmo = cosmo
                calculator.runtime_info.tocalculate = True

    def jac(self, getter, params=None):

        if jax is None:
            raise PipelineError('jax is required to compute the Jacobian')

        def fun(params):
            params = self.eval_params(params)
            for calculator in self.calculators:  # start by first calculator, and by the last one
                runtime_info = calculator.runtime_info
                runtime_info.set_param_values(params, full=True)
                #runtime_info._tocalculate = any(use_jax(value) for value in runtime_info.param_values.values())
                runtime_info._calculate()
            return getter()

        jac = jax.jacfwd(fun, argnums=0, has_aux=False, holomorphic=False)

        if params is None:
            params = self.param_values
        elif not isinstance(params, dict):
            params = {str(param): self.param_values[str(param)] for param in params}

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
                if name not in calculator.runtime_info.params.basenames():
                    param = Parameter(name, namespace=calculator.runtime_info.namespace, derived=True)
                    calculator.runtime_info.params.set(param)
                    calculator.runtime_info.params = calculator.runtime_info.params
        self._params = None
        self._set_params()
        return calculators, fixed, varied

    def _set_speed(self, niterations=10, override=False, seed=42):
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

        footprints = []
        for param in params:
            calculators_to_calculate = []

            def callback(calculator):
                for calculator in calculator.runtime_info.required_by:
                    calculators_to_calculate.append(calculator)
                    callback(calculator)

            for calculator in self.calculators:
                if param in calculator.runtime_info.params:
                    callback(calculator)

            footprints.append(tuple(calculator in calculators_to_calculate for calculator in self.calculators))
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
                # Choose best ordering
                orderings = [sort_parameter_blocks(footprints, block_sizes, speeds, oversample_power=1 - 1e-3)[0]]
                # Then we will recompute costs and oversample_factors
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
    installer = None

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
        if init is None: init = ((), {}, ParameterCollection())
        self.init = list(init)
        self._initialized = False
        self._tocalculate = True
        self.calculated = False
        self.params = ParameterCollection(init[2])
        self.name = self.calculator.__class__.__name__

    def install(self):
        if self.installer is not None:
            try:
                func = self.calculator.install
            except AttributeError:
                return
            func(self.installer)

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
            require.runtime_info.initialize()  # otherwise runtime_info is cleared and required_by is lost
            #assert not require.runtime_info.toinitialize
            require.runtime_info.required_by.add(self.calculator)
        self._pipeline = None

    @property
    def pipeline(self):
        if getattr(self, '_pipeline', None) is None:
            self._pipeline = BasePipeline(self.calculator)
        return self._pipeline

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
        self.base_params = {param.basename: param for param in self._params}
        self.varied_params = ParameterCollection([param for param in self._params if (not param.drop) and (param.depends or (not param.derived) or param.solved)])
        self.derived_params = self._params.select(derived=True, solved=False, depends={})
        self.param_values = {param.basename: param.value for param in self.varied_params}

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
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, initialized):
        self._initialized = initialized
        self._pipeline = None
        if not initialized:
            for calculator in self.required_by:
                calculator.runtime_info.initialized = False

    @property
    def toinitialize(self):
        return not self._initialized

    def initialize(self, **kwargs):
        if self.toinitialize:
            self.clear()
            self.install()
            bak = self.init[2]
            self.init[2] = ParameterCollection(self.init[2]).deepcopy()
            self.calculator.initialize(*self.init[0], **self.init[1])
            self.params = self.init[2]
            self.init[2] = bak
            self._initialized = True
        return self.calculator

    @property
    def tocalculate(self):
        return self._tocalculate or any(require.runtime_info.calculated for require in self.requires)

    @tocalculate.setter
    def tocalculate(self, tocalculate):
        self._tocalculate = tocalculate

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
        self._tocalculate = False

    def calculate(self, **params):
        self._calculate(**params)
        return self.calculator.get()

    def set_param_values(self, param_values, full=False):
        if full:
            for param, value in param_values.items():
                if param in self.varied_params:
                    basename = self.varied_params[param].basename
                    if self.param_values[basename] != value or type(self.param_values[basename]) is not type(value):
                        self._tocalculate = True
                    self.param_values[basename] = value
        else:
            for basename, value in param_values.items():
                basename = str(basename)
                if basename in self.param_values:
                    if self.param_values[basename] != value or type(self.param_values[basename]) is not type(value):
                        self._tocalculate = True
                    self.param_values[basename] = value

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
        new.params = copy.deepcopy(self.params)
        return new


class BaseCalculator(BaseClass):

    def __new__(cls, *args, **kwargs):
        cls_info = Info(getattr(cls, '_info', {}))
        cls_params = ParameterCollection(getattr(cls, '_params', None))
        if hasattr(cls, 'config_fn'):
            dirname = os.path.dirname(sys.modules[cls.__module__].__file__)
            config = BaseConfig(os.path.join(dirname, cls.config_fn), index={'class': cls.__name__})
            cls_info = Info({**config.get('info', {}), **cls_info})
            params = ParameterCollectionConfig(config.get('params', {})).init()
            params.update(cls_params)
            init = config.get('init', {})
            if init: kwargs = {**init, **kwargs}
        else:
            params = cls_params.deepcopy()
        new = super(BaseCalculator, cls).__new__(cls)
        new.info = cls_info
        new.runtime_info = RuntimeInfo(new, init=((), kwargs, params))
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

    def __repr__(self):
        return self.runtime_info.name

    @property
    def params(self):
        if not self.runtime_info.initialized:
            return self.runtime_info.init[2]
        return self.runtime_info.params

    @params.setter
    def params(self, params):
        self.runtime_info.init[2] = params
        self.runtime_info.initialized = False

    @property
    def all_params(self):
        return self.runtime_info.pipeline.params

    @all_params.setter
    def all_params(self, all_params):
        self.runtime_info.pipeline.params = all_params

    @property
    def varied_params(self):
        return self.runtime_info.pipeline.varied_params
