import os
import re
import sys
import copy
import warnings
import functools

import numpy as np

from . import mpi
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

    _attrs = ['_args', '_params', '_updated']  # will be copied

    def __init__(self, *arg, args=None, params=None, **kwargs):
        """
        Initialize :class:`InitConfig`.

        Parameters
        ----------
        params : list, ParameterCollection
            Parameters at initialization.
        """
        self.args = args or ()
        self.params = params or ParameterCollection()
        self._updated = True
        super(InitConfig, self).__init__(*arg, **kwargs)

    @property
    def updated(self):
        """Whether the configuration parameters have been updated (which requires reinitialization of the calculator)."""
        return self._updated or self.params.updated

    @updated.setter
    def updated(self, updated):
        """Set the 'updated' status."""
        self._updated = bool(updated)
        if self._updated:
            try:
                calculator = self.runtime_info.calculator
                if not getattr(self.runtime_info, '_initialization', False):
                    calculator.__dict__ = {name: calculator.__dict__[name] for name in ['_mpicomm', 'info', 'runtime_info']}
            except AttributeError:
                pass
        else:
            self.params.updated = False

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        """Positional arguments to be passed to calculator."""
        self._args = tuple(args)
        self.updated = True

    @property
    def params(self):
        """Parameters."""
        return self._params

    @params.setter
    def params(self, params):
        """Set parameters."""
        self._params = ParameterCollection(params)
        self.updated = True

    def __getstate__(self):
        """Return state."""
        return {name: getattr(self, name) for name in ['data', 'args', 'params', 'updated']}

    def __setstate__(self, state):
        """Set state."""
        for name, value in state.items():
            if name == 'data':
                self.data = state[name]
            else:
                setattr(self, '_' + name, value)


def _make_wrapper(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        toret = func(self, *args, **kwargs)
        self.updated = True
        return toret

    return wrapper


for name in ['__delitem__', '__getitem__', '__setitem__', 'clear', 'fromkeys', 'pop', 'popitem', 'setdefault', 'update']:

    setattr(InitConfig, name, _make_wrapper(getattr(UserDict, name)))


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

        def callback(calculator):
            self.calculators.append(calculator.runtime_info.initialize())
            for require in calculator.runtime_info.requires:
                if require in self.calculators:
                    del self.calculators[self.calculators.index(require)]  # we want first dependencies at the end
                callback(require)

        callback(calculator)
        # To avoid loops created by one calculator, which when updated, requests reinitialization of the calculators which depend on it
        for calculator in self.calculators:
            calculator.runtime_info.initialized = True
        self.calculators = self.calculators[::-1]
        self.mpicomm = calculator._mpicomm
        for calculator in self.calculators:
            calculator.runtime_info.tocalculate = True
        self._params = ParameterCollection()
        self._set_params()
        self.more_derived, self.more_calculate = None, None

    def _set_params(self, params=None):
        # Internal method to reset parameters, based on calculator's :class:`BaseCalculator.runtime_info.params`
        params_from_calculator = {}
        params = ParameterCollectionConfig(params, identifier='name')
        new_params = ParameterCollection()
        for calculator in self.calculators:
            calculator_params = ParameterCollection(ParameterCollectionConfig(calculator.runtime_info.params, identifier='name').clone(params))
            #new_calculator_params = ParameterCollection()
            for iparam, param in enumerate(calculator.runtime_info.params):
                param = calculator_params[param]
                if param in new_params:
                    if param.derived and param.fixed:
                        msg = 'Derived parameter {} of {} is already derived in {}.'.format(param, calculator, params_from_calculator[param.name])
                        if self.mpicomm.rank == 0:
                            warnings.warn(msg)
                    elif param != new_params[param]:
                        raise PipelineError('Parameter {} of {} is different from that of {}.'.format(param, calculator, params_from_calculator[param.name]))
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
        for param in self._params:
            if param not in new_params:
                # Add in previous parameters to be dropped
                if any(param.name in p.depends.values() for p in new_params):
                    new_params.set(param)
        self._params = new_params.deepcopy()
        self._params.updated = False
        self._varied_params = self._params.select(varied=True, derived=False)
        self.input_values = {param.name: param.value for param in self._params}
        self.derived = None

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

    def calculate(self, **params):
        """
        Calculate, i.e. call calculators' :meth:`BaseCalculator.calculate` if their parameters are updated,
        or if they depend on previous calculation that has been updated.
        Derived parameter values are stored in :attr:`derived`.
        """
        for name in params:
            if name not in self.params:
                raise PipelineError('Input parameter {} is not one of parameters: {}'.format(name, self.params))
        self.input_values.update(params)
        params = self.params.eval(**self.input_values)
        self.derived = Samples()
        for param in self._params:
            if param.depends: self.derived.set(ParameterArray(np.asarray(params[param.name]), param=param))
        for calculator in self.calculators:  # start by first calculator
            runtime_info = calculator.runtime_info
            runtime_info.set_input_values(params, full=True)
            result = runtime_info.calculate()
            self.derived.update(runtime_info.derived)
        if self.more_calculate:
            toret = self.more_calculate()
            if toret is not None: result = toret
        if self.more_derived:
            tmp = self.more_derived(0)
            if tmp is not None: self.derived.update(tmp)
        return result

    def mpicalculate(self, **params):
        """MPI-parallel version of the above: one can pass arrays as input parameter values."""
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
        self.derived = Samples()
        mpicomm, more_derived = self.mpicomm, self.more_derived
        self.mpicomm, self.more_derived = mpi.COMM_SELF, None
        states = {}
        for ivalue in range(size):
            self.calculate(**{name: value[ivalue] for name, value in params.items()})
            istate = ivalue + cumsizes[mpicomm.rank]
            states[istate] = self.derived
            if more_derived:
                tmp = more_derived(istate)
                if tmp is not None: states[istate].update(tmp)
        self.mpicomm, self.more_derived = mpicomm, more_derived
        derived = None
        states = self.mpicomm.gather(states, root=0)
        if self.mpicomm.rank == 0:
            derived = {}
            for state in states: derived.update(state)
            derived = Samples.concatenate([derived[i] for i in range(cumsizes[-1])]).reshape(cshape)
        self.derived = derived

    def get_cosmo_requires(self):
        """Return a dictionary mapping section to method's name and arguments,
        e.g. 'background': {'comoving_radial_distance': {'z': z}}."""
        from .cosmo import BaseExternalEngine
        return BaseExternalEngine.get_requires(*[getattr(calculator, 'cosmo_requires', {}) for calculator in self.calculators])

    def set_cosmo_requires(self, cosmo):
        """Set input :class:`cosmoprimo.Cosmology` instance."""
        for calculator in self.calculators:
            cosmo_requires = getattr(calculator, 'cosmo_requires', {})
            if cosmo_requires:
                conversions = {'logA': 'ln10^10A_s'}
                cosmo_params = cosmo_requires.get('params', {})
                if cosmo_params:
                    for basename, name in calculator.runtime_info.base_names.items():
                        if basename in cosmo_params:
                            self.input_values[name] = calculator.runtime_info.input_values[basename] = cosmo[conversions.get(basename, basename)]
                if set(cosmo_requires.keys()) != {'params'}:  # requires a :class:`cosmoprimo.Cosmology` instance as ``cosmo`` attribute
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
                try:
                    eq = all(deep_eq(value, values[0]) for value in values)
                except Exception as exc:
                    raise ValueError('unable to check equality of {} (type: {})'.format(name, type(values[0]))) from exc
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
                    if hasattr(param, 'setdefault'):
                        param = param.copy()
                        param.setdefault('namespace', calculator.runtime_info.namespace)
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
        """
        Group parameters together, and compute their ``oversample_factor``, indicative of the frequency
        at which they should be updated altogether.

        Note
        ----
        Algorithm taken from Cobaya.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to sort into blocks. Defaults to :attr:`varied_params`.

        nblocks : int, default=None
            Number of blocks. If ``None``, parameters are grouped by "footprint",
            i.e. the set of calculators that depend on them (either directly, or indirectly, through calculators' requirements).

        oversample_power : int, default=0
            ``oversample_factor`` is proportional to ``speed ** oversample_power``.

        **kwargs : dict
            Optional arguments for :meth:`_set_speed`, which is called if any :attr:`BaseCalculator.runtime_info.speed`
            of :attr:`calculators` is not set.

        Returns
        -------
        sorted_blocks : list
            List of list of parameter names.

        oversample_factors : list
            List of corresponding oversample factor (for each block).
        """
        from itertools import permutations, chain
        if params is None: params = self.varied_params
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
                for calc in self.calculators:
                    if calculator in calc.runtime_info.requires:
                        calculators_to_calculate.append(calc)
                        callback(calc)

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
            permuted_oversample_factors = (permuted_costs_per_param_per_block[..., [0]] / permuted_costs_per_param_per_block) ** oversample_power
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
        #self.required_by = set()
        if init is None: init = InitConfig()
        self.init = init
        if not isinstance(init, InitConfig):
            self.init = InitConfig(init)
        self.init.runtime_info = self
        self._initialized = False
        self._tocalculate = True
        self.calculated = False
        self._with_namespace = False
        self.params = ParameterCollection(init.params)
        self.name = self.calculator.__class__.__name__

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
    def requires(self):
        """
        Return list of calculators this calculator depends upon.
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
        #for require in self._requires:
            #require.runtime_info.initialize()  # otherwise runtime_info is cleared and required_by is lost
            #assert not require.runtime_info.toinitialize
            #require.runtime_info.required_by.add(self.calculator)
        self._pipeline = None

    @property
    def pipeline(self):
        """Return pipeline for this calculator."""
        if getattr(self, '_pipeline', None) is None:
            self._pipeline = BasePipeline(self.calculator)
        elif any(not calculator.runtime_info.initialized for calculator in self._pipeline.calculators):
            initialized = True
            for calculator in self._pipeline.calculators:
                if not calculator.runtime_info.initialized:
                    initialized = False
                calculator.runtime_info.initialized = initialized
            self._pipeline = BasePipeline(self.calculator)
        return self._pipeline

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
        self.base_names = {(param.name if self._with_namespace else param.basename): param.name for param in self.params}
        self.input_names = {param.name: (param.name if self._with_namespace else param.basename) for param in self.params.select(input=True)}
        self.input_values = {(param.name if self._with_namespace else param.basename): param.value for param in self.params.select(input=True)}
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
                    value = np.asarray(value)
                    param._shape = value.shape  # a bit hacky, but no need to update parameters for this...
                    self._derived.set(ParameterArray(value, param=param))
        return self._derived

    @property
    def initialized(self):
        """Has this calculator been initialized?"""
        if self.init.updated:
            self._initialized = False
        return self._initialized

    @initialized.setter
    def initialized(self, initialized):
        if initialized:
            self.init.updated = False
        #else:
            #self._pipeline = None
        #    for calculator in self.required_by:
        #        calculator.runtime_info.initialized = False
        self._initialized = initialized

    def initialize(self):
        """Initialize calculator (if not already initialized), calling :meth:`BaseCalculator.initialize` with :attr:`init` configuration."""
        if not self.initialized:
            self.clear()
            self._initialization = True   # to avoid infinite loops
            self.calculator.__dict__ = {name: self.calculator.__dict__[name] for name in ['info', 'runtime_info', '_mpicomm']}
            self.install()
            bak = self.init.params
            params_with_namespace = ParameterCollection(self.init.params).deepcopy()
            self._with_namespace = getattr(self.calculator, '_with_namespace', False)
            if not self._with_namespace:
                params_basenames = params_with_namespace.basenames()
                # Pass parameters without namespace
                self.params = self.init.params = params_with_namespace.clone(namespace=None)
            else:
                self.params = self.init.params = params_with_namespace
            try:
                self.calculator.initialize(*self.init.args, **self.init)
            except Exception as exc:
                raise PipelineError('Error in method initialize of {}'.format(self.calculator)) from exc

            if not self._with_namespace:
                for param in self.init.params:
                    if param.basename in params_basenames:  # update namespace
                        param.update(namespace=params_with_namespace[params_basenames.index(param.basename)].namespace)
            self.params = self.init.params
            self.init.params = bak
            self.initialized = True
            self._initialization = False
            if getattr(self, '_requires', None) is None:
                self._requires = []
                for name, value in self.calculator.__dict__.items():
                    if isinstance(value, BaseCalculator):
                        self._requires.append(value)
        return self.calculator

    @property
    def tocalculate(self):
        """Should calculator's :class:`BaseCalculator.calculate` be called?"""
        return self._tocalculate or any(require.runtime_info.calculated for require in self.requires)

    @tocalculate.setter
    def tocalculate(self, tocalculate):
        self._tocalculate = tocalculate

    def calculate(self, **params):
        """
        If calculator's :class:`BaseCalculator.calculate` has not be called with input parameter values, call it,
        keeping track of running time with :attr:`monitor`.
        """
        self.set_input_values(params)
        if self.tocalculate:
            self.monitor.start()
            try:
                self.calculator.calculate(**self.input_values)
            except Exception as exc:
                raise PipelineError('Error in method calculate of {} with calculator parameters {} and pipeline parameters {}'.format(self.calculator, self.input_values, self.pipeline.input_values)) from exc
            self.monitor.stop()
            self._derived = None
            self.calculated = True
        else:
            self.calculated = False
        self._tocalculate = False
        return self.calculator.get()

    def set_input_values(self, input_values, full=False, force=None):
        """Update parameter values; if new, next :meth:`calculate` call will call calculator's :class:`BaseCalculator.calculate`."""
        self.params
        if full:
            for name, value in input_values.items():
                name = str(name)
                if name in self.input_names:
                    basename = self.input_names[name]
                    if force is not None:
                        self._tocalculate = force
                    elif type(self.input_values[basename]) is not type(value) or self.input_values[basename] != value:
                        self._tocalculate = True
                    self.input_values[basename] = value
        else:
            for basename, value in input_values.items():
                basename = str(basename)
                if basename in self.input_values:
                    if force is not None:
                        self._tocalculate = force
                    elif self.input_values[basename] != value or type(self.input_values[basename]) is not type(value):
                        self._tocalculate = True
                    self.input_values[basename] = value

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
        cls_params = ParameterCollection(getattr(cls, '_params', None))
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
        init.params = params
        new.runtime_info = RuntimeInfo(new, init=init)
        new._mpicomm = mpi.COMM_WORLD
        return new

    def __init__(self, *args, **kwargs):
        self.init.args = args
        self.init.update(**kwargs)

    @property
    def mpicomm(self):
        if not self.runtime_info.initialized:
            return self._mpicomm
        return self.runtime_info.pipeline.mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        if not self.runtime_info.initialized:
            self._mpicomm = mpicomm
        self.runtime_info.pipeline.mpicomm = mpicomm

    @property
    def init(self):
        """Return configuration at initialization."""
        return self.runtime_info.init

    def __call__(self, **params):
        """Take all parameters as input, calculate, and return the result of :attr:`get`"""
        return self.runtime_info.pipeline.calculate(**params)

    def initialize(self, **kwargs):
        # Define this method, with takes meta parameters as input. Parameters can be accessed through self.params.
        pass

    def calculate(self, **params):
        # Define this method, which takes parameter values as input.
        pass

    def get(self):
        """Return quantity of main interest, e.g. loglikelihood + logprior if ``self`` is a likelihood."""
        return self

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
                params = self.runtime_info.pipeline.params.deepcopy()
                new.runtime_info.pipeline._set_params(params)  # to preserve depends
                new(**self.runtime_info.pipeline.input_values)
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
                        if not any(param in calculator.runtime_info.params for calculator in calculator.runtime_info.pipeline.calculators):
                            self.params[param] = param.clone(drop=True)  # add parameter to this calculator
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
                    return self.all_derived[calcname][basename].runtime_info.derived[basename][0]  # Samples, so remove the first axis
        raise AttributeError('calculator {} has no attribute {};'
                             'have you run any calculation already by calling this calculator or calculators'
                             'that depend on it (typically, a likelihood?)'.format(self.__class__.__name__, name))

    def __getstate__(self):
        state = {}
        for calcname, calculator in zip(self.names, self.calculators):
            for key, value in calculator.__getstate__().items():
                state['{}_{}'.format(calcname, key)] = value
        return state
