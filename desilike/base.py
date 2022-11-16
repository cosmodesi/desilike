import os
import sys

import numpy as np

from . import mpi, utils
from .utils import BaseClass, NamespaceDict, Monitor
from .io import BaseConfig
from .parameter import ParameterCollection, ParameterArray, ParameterValues


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
                if require not in self.calculators:
                    self.calculators.append(require.runtime_info.initialize())
                    callback(require)

        callback(self.calculators[0])
        self.calculators = self.calculators[::-1]
        self._derived = None
        self.mpicomm = calculator.mpicomm

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        for calculator in self.calculators:
            calculator.mpicomm = mpicomm

    def _get_params(self, params=None, quiet=False):
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
        self._derived = None
        return params

    @property
    def params(self):
        return self._get_params()

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
        to_calculate = self._derived is None
        self.param_values.update(params)
        params = self.eval_params(params)
        for calculator in self.calculators:  # start by first calculator, and by the last one
            for param in calculator.runtime_info.full_params:
                value = params.get(param.name, None)
                if value is not None and param.basename in calculator.runtime_info.param_values and value != calculator.runtime_info.param_values[param.basename]:
                    calculator.runtime_info.param_values[param.basename] = value
                    to_calculate = True
        result = None
        if to_calculate:
            self.derived = ParameterValues()
            for calculator in self.calculators:
                result = calculator.runtime_info.calculate()
                for param in calculator.runtime_info.full_params:
                    if param.depends:
                        self.derived.set(ParameterArray(np.asarray(params[param.name]), param=param))
                self.derived.update(calculator.runtime_info.derived)
        return result

    def get_cosmo_requires(self):
        from .cosmo import ExternalEngine
        return ExternalEngine.get_requires(*[getattr(calculator, 'cosmo_requires', {}) for calculator in self.calculators])

    def set_cosmo_requires(self, cosmo):
        for calculator in self.calculators:
            if getattr(calculator, 'cosmo_requires', None):
                calculator.cosmo = cosmo


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
        self.monitor = Monitor()
        self.required_by = set()
        self.init = dict(init or {})
        self.initialized = False

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
            self._full_params = self.calculator.params
        return self._full_params

    @full_params.setter
    def full_params(self, full_params):
        self._full_params = full_params
        self._base_params = self._solved_params = self._derived_params = self._param_values = self._pipeline = None

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
    def derived(self):
        if getattr(self, '_derived', None) is None:
            self._derived = ParameterValues()
            if self.derived_params:
                state = self.calculator.__getstate__()
                for param in self.derived_params:
                    name = param.basename
                    if name in state: value = state[name]
                    else: value = getattr(self.calculator, name)
                    self._derived.set(ParameterArray(np.asarray(value), param=param), output=True)
        return self._derived

    def initialize(self, **kwargs):
        if self.initialized: return self.calculator
        self.clear(initialized=True)
        self.calculator.initialize(**self.init)
        return self.calculator

    def calculate(self, **params):
        self.param_values.update(**params)
        self.monitor.start()
        try:
            self.result = self.calculator.calculate(**self.param_values)
        except Exception as exc:
            raise PipelineError('Error in method calculate of {}'.format(self.calculator)) from exc
        self.monitor.stop()
        return self.result

    @property
    def param_values(self):
        if getattr(self, '_param_values', None) is None:
            self._param_values = {param.basename: param.value for param in self.full_params if (not param.drop) and (param.depends or (not param.derived) or param.solved)}
        return self._param_values

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
        cls.info = Info(**getattr(cls, 'info', {}))
        cls.params = ParameterCollection(getattr(cls, 'params', None))
        if hasattr(cls, 'config_fn'):
            dirname = os.path.dirname(sys.modules[cls.__module__].__file__)
            config = BaseConfig(os.path.join(dirname, cls.config_fn), index={'class': cls.__name__})
            cls.info = Info(**{**config.get('info', {}), **cls.info})
            params = ParameterCollection(config.get('params', {}))
            params.update(cls.params)
            cls.params = params
            init = config.get('init', {})
            if init: kwargs = {**init, **kwargs}
        new = super(BaseCalculator, cls).__new__(cls)
        new.runtime_info = RuntimeInfo(new, init=kwargs)
        new.mpicomm = mpi.COMM_WORLD
        return new

    def __init__(self, *args, **kwargs):
        if args:
            raise SyntaxError('Provide named arguments')
        self.update(**kwargs)

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.runtime_info.init[name] = value
        self.runtime_info.initialized = False

    def __call__(self, **params):
        return self.runtime_info.pipeline.calculate(**params)

    def calculate(self):
        pass

    def __getstate__(self):
        return {}
