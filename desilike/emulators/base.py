import os
import re

import numpy as np

from desilike import utils, plotting
from desilike.base import BaseCalculator
from desilike.jax import numpy as jnp
from desilike.utils import BaseClass, serialize_class, import_class, expand_dict
from desilike.io import BaseConfig
from desilike.parameter import Parameter, ParameterArray, Samples, ParameterCollection, ParameterConfig


def find_uniques(li):
    toret = []
    for el in li:
        if el not in toret:
            toret.append(el)
    return toret


def get_subsamples(samples, frac=1., nmax=np.inf, seed=42, mpicomm=None):
    """Return a random fraction ``frac`` of input samples."""
    nsamples = mpicomm.bcast(samples.size if mpicomm.rank == 0 else None)
    size = min(max(int(nsamples * frac + 0.5), 1), nmax)
    if size > nsamples:
        raise ValueError('Cannot use {:d} subsamples (> {:d} total samples)'.format(size, nsamples))
    rng = np.random.RandomState(seed=seed)
    toret = None
    if mpicomm.rank == 0:
        toret = samples.ravel()[rng.choice(nsamples, size=size, replace=False)]
    return toret


def _setstate(self, state):
    self.__dict__.update({name: value for name, value in state.items() if name not in self.emulator.in_calculator_state})
    self.__setstate__({name: value for name, value in state.items() if name in self.emulator.in_calculator_state})


class EmulatedCalculator(BaseCalculator):

    def initialize(self, emulator=None, **kwargs):
        self.emulator = emulator
        self.calculate(**{param.basename: param.value for param in self.init.params})

    def calculate(self, **params):
        predict = self.emulator.predict(**params)
        state = {**self.emulator.fixed, **predict}
        _setstate(self, state)

    @classmethod
    def load(cls, filename):
        return Emulator.load(filename).to_calculator()

    def save(self, fn):
        return self.emulator.save(fn)


class Emulator(BaseClass):

    """
    Class to emulate a :class:`BaseCalculator` instance.

    For a :class:`BaseCalculator` to be emulated, it must implement:

    - __getstate__(self): a method returning ``state`` a dictionary of attributes as basic python types and numpy arrays,
      everything required to replace a call to :meth:`BaseCalculator.calculate`
    - __setstate__(self, state): a method setting calculator's state
    """

    def __init__(self, calculator, engine='taylor', mpicomm=None):
        """
        Initialize emulator.

        Parameters
        ----------
        calculator : BaseCalculator
            Input calculator.

        engine : str, dict, BaseEmulatorEngine, default='taylor'
            A dictionary mapping calculator's derived attribute names (including wildcard) to emulator engine,
            which can be a :class:`BaseEmulatorEngine` (type or instance) or one of ['taylor', 'mlp'].
            A single emulator engine can be provided, and used for all calculator's derived attributes.

        mpicomm : MPI communicator, default=None
            Optionally, the MPI communicator.
        """
        self.is_calculator_sequence = utils.is_sequence(calculator)
        if self.is_calculator_sequence:
            self.is_calculator_sequence = len(calculator)
            from desilike.base import CollectionCalculator
            calculator = CollectionCalculator(calculators=calculator)
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.calculator = calculator
        self.pipeline = self.calculator.runtime_info.pipeline
        self.params = self.pipeline.params.clone(namespace=None).deepcopy()
        self.varied_params = self.pipeline.varied_params.clone(namespace=None).names()
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        calculators, fixed, varied = self.pipeline._classify_derived(self.pipeline.calculators)
        self.fixed, self.varied = {}, {}

        params = []
        for cc, ff, vv in zip(calculators[:-1], fixed, varied):
            base_names = cc.runtime_info.base_names
            fff, vvv = {}, {}
            for k, v in ff.items():
                if k in base_names and self.params[base_names[k]].derived: fff[base_names[k]] = v
            for k in vv:
                if k in base_names and self.params[base_names[k]].derived: vvv[base_names[k]] = None
            if self.is_calculator_sequence:
                fff = {name: value for name, value in fff.items() if re.match(r'(\d*)_(.*)', name)}
                vvv = {name: value for name, value in vvv.items() if re.match(r'(\d*)_(.*)', name)}
            self.fixed.update(fff)
            self.varied.update(vvv)
            params.append(list(vvv))
        self.fixed.update(fixed[-1])
        self.varied.update(dict.fromkeys(varied[-1]))
        params.append(varied[-1])
        self.pipeline._set_derived(calculators, params=params)
        self.in_calculator_state = list(fixed[-1]) + varied[-1]
        self.varied = list(self.varied)

        if mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params))
            self.log_info('Found varying {} and fixed {} outputs.'.format(self.varied, list(self.fixed.keys())))
        if not self.varied:
            raise ValueError('Found no varying quantity in provided calculator')

        def get_calculator_info(calculator):
            calculator__class__ = serialize_class(calculator.__class__)
            yaml_data = BaseConfig()
            yaml_data['class'] = calculator.__class__.__name__
            yaml_data['info'] = dict(calculator.info)
            #self.yaml_data['init'] = dict(calculator.runtime_info.init)
            params = {}
            for param in calculator.all_params:
                params[param.basename] = dict(ParameterConfig(param))
                params[param.basename].pop('basename')
                params[param.basename].pop('namespace', None)
            yaml_data['params'] = params
            return calculator__class__, yaml_data

        if self.is_calculator_sequence:
            self.calculator__class__, self.yaml_data = [], []
            for name, calculator in self.calculator.items():
                calculator__class__, yaml_data = get_calculator_info(calculator)
                self.calculator__class__.append(calculator__class__)
                self.yaml_data.append(yaml_data)
                self.fixed[name + '_cosmo_requires'] = calculator.runtime_info.pipeline.get_cosmo_requires()
        else:
            self.calculator__class__, self.yaml_data = get_calculator_info(self.calculator)
            # Add in cosmo_requires
            self.fixed['cosmo_requires'] = self.calculator.runtime_info.pipeline.get_cosmo_requires()

        if not hasattr(engine, 'items'):
            engine = {'*': engine}
        self.engines = engine
        for key, engine in self.engines.items():
            self.engines[key] = get_engine(engine)
        self.engines = utils.expand_dict(self.engines, self.varied)
        for name, engine in self.engines.items():
            if engine is None:
                raise ValueError('Engine not specified for varying attribute {}'.format(name))
            engine.initialize(varied_params=self._get_varied_params(name))

        self.varied_shape = {name: -1 for name in self.engines}
        self.samples, self.diagnostics = {}, {}
        self.mpicomm = mpicomm

    @property
    def mpicomm(self):
        from desilike import mpi
        return getattr(self, '_mpicomm', mpi.COMM_WORLD)

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        try:
            for engine in self.engines.values():
                engine.mpicomm = mpicomm
        except AttributeError:
            pass

    def _get_varied_params(self, name):
        name = str(name)
        if self.is_calculator_sequence:
            index = int(re.match(r'(\d*)(_.*|)$', name).group(1))
            return [name for name in self.varied_params if name in self.yaml_data[index]['params']]
        return self.varied_params.copy()

    def set_samples(self, name=None, samples=None, **kwargs):
        """
        Set samples for :meth:`fit`.

        Parameters
        ----------
        name : str, default=None
            Name of calculator's derived attribute(s) (of :attr:`varied`) these samples apply to.
            If ``None``, samples are set for all attributes.

        samples : Samples, default=None
            Samples containing ``calculator.varied_params`` and calculator's derived attributes :attr:`varied`.
            If ``None``, samples will be generated using engines' :meth:`BaseEmulatorEngine.get_default_samples` methods.

        **kwargs : dict
            If ``samples`` is ``None``, optional arguments for :meth:`BaseEmulatorEngine.get_default_samples`.
        """
        if name is None:
            unique_engines = find_uniques(self.engines.values())
            if len(unique_engines) == 1:
                engine = unique_engines[0]
            else:
                raise ValueError('Provide either attribute name or engine')
        elif isinstance(name, str):
            engine = self.engines[name]
        else:
            engine = name
        if self.mpicomm.bcast(samples is None, root=0):
            samples = engine.get_default_samples(self.calculator, **kwargs)
        elif self.mpicomm.rank == 0:
            samples = samples if isinstance(samples, Samples) else Samples.load(samples)
        tmp = None
        if self.mpicomm.rank == 0:
            tmp = Samples(attrs=samples.attrs)
            for param in self.pipeline.varied_params + self.pipeline.params.select(name=list(self.engines.keys()), derived=True):
                tmp[param.clone(namespace=None)] = samples[param.name]
        for name, eng in self.engines.items():
            if eng is engine:
                self.samples[name] = tmp

    def fit(self, name=None, **kwargs):
        """
        Fit :class:`BaseEmulatorEngine` to samples.

        Parameters
        ----------
        name : str, default=None
            Name of calculator's derived attribute(s) (of :attr:`varied`) these samples apply to.
            If ``None``, fits will be performed for all calculator's derived attributes.

        **kwargs : dict
            Optional arguments for :meth:`BaseEmulatorEngine.fit`.
        """
        def _get_X_Y(samples, yname, with_deriv=False):
            X, Y = None, None
            if self.mpicomm.rank == 0:
                nsamples = samples.size
                X = np.concatenate([samples[name].reshape(nsamples, 1) for name in self._get_varied_params(yname)], axis=-1)
                Y = samples[yname]
                yshape = Y.shape[Y.andim:]
                derivs = Y.derivs
                if derivs is not None:
                    yshape = yshape[1:]
                    if with_deriv:
                        Y = Y.reshape(nsamples, Y.shape[Y.andim], -1)
                    else:
                        Y = Y.zero.reshape(nsamples, -1)
                else:
                    Y = Y.reshape(nsamples, -1)
                Y = Y.clone(param=Y.param.clone(shape=np.prod(yshape)), derivs=derivs)
                self.varied_shape[yname] = yshape
            self.varied_shape[yname] = self.mpicomm.bcast(self.varied_shape[yname], root=0)
            return X, Y

        if name is None:
            name = list(self.engines.keys())
        if not utils.is_sequence(name):
            name = [name]
        names = name

        for name in names:
            self.engines[name] = engine = self.engines[name].copy()
            engine.fit(*_get_X_Y(self.samples[name], name, getattr(engine, '_samples_with_derivs', False)), **kwargs)

    def predict(self, **params):
        X = jnp.array([params[name] for name in self.varied_params])
        return {name: engine.predict(X).reshape(self.varied_shape[name]) for name, engine in self.engines.items()}

    def to_calculator(self, derived=None):
        """
        Export new :class:`EmulatedCalculator` instance,
        that can readily be used in replacement of input calculator.

        Parameters
        ----------
        derived : list, ParameterCollection
            List of parameters to set as derived in returned calculator.

        Returns
        -------
        calculator : EmulatedCalculator
            Emulated calculator.
        """
        def _split_emulator(self):
            emulators = []
            for index in range(self.is_calculator_sequence):
                emulator = self.__class__.__new__(self.__class__)
                emulator.is_calculator_sequence = False
                emulator.yaml_data = self.yaml_data[index]
                emulator.params = self.params.select(basename=list(emulator.yaml_data['params'].keys())).deepcopy()
                emulator.varied_params = self._get_varied_params(index)
                emulator.calculator__class__ = self.calculator__class__[index]
                index = str(index)

                def get_name(name):
                    return name[len(index) + 1:]

                emulator.fixed = {get_name(name): value for name, value in self.fixed.items() if name.startswith(index)}
                emulator.varied_shape = {get_name(name): value for name, value in self.varied_shape.items() if name.startswith(index)}
                emulator.in_calculator_state = [get_name(name) for name in self.in_calculator_state if name.startswith(index)]
                emulator.engines = {get_name(name): engine for name, engine in self.engines.items() if name.startswith(index)}
                emulators.append(emulator)
            return emulators

        if self.is_calculator_sequence:
            return [emulator.to_calculator(derived=derived) for emulator in _split_emulator(self)]

        state = self.__getstate__()

        Calculator = import_class(*state['calculator__class__'])
        new_name = Calculator.__name__

        new_cls = type(EmulatedCalculator)(new_name, (EmulatedCalculator, Calculator),
                                           {'__setstate__': Calculator.__setstate__, 'get': Calculator.get, '__module__': Calculator.__module__, 'config_fn': None})

        new_cls._params = self.params.select(depends={}).deepcopy()
        for param in new_cls._params: param.update(drop=False)

        calculator = new_cls(emulator=self)
        calculator.runtime_info.initialize()  # to initialize
        if derived is not None:
            calculator.runtime_info.pipeline._set_derived([calculator], params=[derived])

        return calculator

    def check(self, mse_stop=None, diagnostics=None, frac=0.1, **kwargs):
        """
        Check emulator against provided samples.

        Parameters
        ----------
        mse_stop : float, default=None
            Mean squared error (MSE).

        diagnostics : default=None
            Dictionary where computed statistics (MSE) are added.
            Default is :attr:`diagnostics`.

        frac : float, default=0.1
            Fraction of samples to select for testing.

        seed : int, default=None
            Random seed for sample downsampling.
        """
        if diagnostics is None:
            diagnostics = self.diagnostics

        if self.mpicomm.rank == 0:
            self.log_info('Diagnostics:')

        toret = True
        calculator = self.to_calculator(derived=['emulator.{}'.format(name) for name in self.engines])
        pipeline = calculator.runtime_info.pipeline

        unique_samples = find_uniques(self.samples.values())
        for samples in unique_samples:
            subsamples = get_subsamples(samples, frac=frac, mpicomm=self.mpicomm, **kwargs)
            pipeline.mpicalculate(**{name: subsamples[name] if self.mpicomm.rank == 0 else None for name in self.varied_params})
            derived = pipeline.derived

            if self.mpicomm.rank == 0:

                def add_diagnostics(name, value):
                    if name not in diagnostics:
                        diagnostics[name] = [value]
                    else:
                        diagnostics[name].append(value)
                    return value

                item = '- '
                mse = {}
                for name in self.samples:
                    if self.samples[name] is samples:
                        mse[name] = np.mean((derived['emulator.' + name] - subsamples[name].zero)**2)
                        msg = '{}mse of {} is {:.3g} (square root = {:.3g})'.format(item, name, mse[name], np.sqrt(mse[name]))
                        if mse_stop is not None:
                            test = mse[name] < mse_stop
                            self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', mse_stop))
                            toret &= test
                        else:
                            self.log_info('{}.'.format(msg))
                add_diagnostics('mse', mse)

            diagnostics.update(self.mpicomm.bcast(diagnostics, root=0))

        return self.mpicomm.bcast(toret, root=0)

    def plot(self, name=None, nmax=100, fn=None, kw_save=None, show=False, **kwargs):
        """
        Plot comparison between input calculator and its emulator.

        Parameters
        ----------
        name : str, default=None
            Name of calculator's derived attribute(s) (of :attr:`varied`) to plot.

        nmax : int, default=100
            Maximum number of samples to plot.

        fn : str, Path, default=None
            Figure path. Should be a list of paths if ``name`` refer to multiple attributes.

        kw_save : dict, default=None
            If ``fn`` is provided, dictionary of arguments to be passed
            to :func:`matplotlib.pyplot.savefig`, e.g. 'dpi'.

        show : bool, default=None
            If ``True``, show figure.

        seed : int, default=None
            Random seed for sample downsampling.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt

        fns, names = fn, name
        if names is None:
            names = list(self.engines.keys())
        if utils.is_sequence(names):
            if fns is None:
                fns = [None] * len(names)
            elif not utils.is_sequence(fns):
                fns = [fns.replace('*', '{}').format(name) for name in names]
        else:
            fns, names = [fns], [names]
        fns = {name: ff for name, ff in zip(names, fns)}

        calculator = self.to_calculator(derived=['emulator.{}'.format(name) for name in self.engines])
        pipeline = calculator.runtime_info.pipeline
        unique_samples = find_uniques(self.samples.values())

        figs = []
        for samples in unique_samples:
            samples_names = [name for name, s in self.samples.items() if s is samples]
            if samples_names:
                subsamples = get_subsamples(samples, nmax=nmax, mpicomm=self.mpicomm, **kwargs)
                pipeline.mpicalculate(**{name: subsamples[name] if self.mpicomm.rank == 0 else None for name in self.varied_params})
                derived = pipeline.derived

                if self.mpicomm.rank == 0:
                    for name in samples_names:
                        plt.close(plt.gcf())
                        fig, lax = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': (2, 1)}, figsize=(6, 6), squeeze=True)
                        fig.subplots_adjust(hspace=0)
                        for d, s in zip(derived['emulator.' + name].zero, subsamples[name].zero):
                            lax[0].plot(d.ravel(), color='k', marker='+', markersize=1, alpha=0.2)
                            lax[1].plot((d - s).ravel(), color='k', marker='+', markersize=1, alpha=0.2)
                        lax[0].set_ylabel(name)
                        lax[1].set_ylabel(r'$\Delta$ {}'.format(name))
                        for ax in lax: ax.grid(True)
                        figs.append(fig)
                        if fn is not None:
                            plotting.savefig(fn, fig=fig, **(kw_save or {}))
                        if show:
                            plt.show()
        return figs

    def __getstate__(self):
        state = {'engines': {}}
        for name, engine in self.engines.items():
            state['engines'][name] = {'__class__': serialize_class(engine.__class__), **engine.__getstate__()}
        for name in ['varied_params', 'fixed', 'varied_shape', 'in_calculator_state', 'calculator__class__', 'is_calculator_sequence']:
            state[name] = getattr(self, name)
        state['yaml_data'] = [yaml_data.data for yaml_data in self.yaml_data] if self.is_calculator_sequence else self.yaml_data.data
        state['params'] = self.params.__getstate__()
        return state

    def save(self, filename, yaml=True):
        state = self.__getstate__()
        if self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            if yaml:
                state['config_fn'] = fn = os.path.splitext(filename)[0] + '.yaml'
                BaseConfig.write(self.yaml_data, fn)
            np.save(filename, state, allow_pickle=True)

    def __setstate__(self, state):
        super(Emulator, self).__setstate__(state)
        self.is_calculator_sequence = getattr(self, 'is_calculator_sequence', False)
        self.yaml_data = [BaseConfig(yaml_data) for yaml_data in self.yaml_data] if self.is_calculator_sequence else BaseConfig(self.yaml_data)
        self.params = ParameterCollection.from_state(state['params'])
        for name, state in self.engines.items():
            state = state.copy()
            self.engines[name] = import_class(*state.pop('__class__')).from_state(state)


class RegisteredEmulatorEngine(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls

        def wrapper(func):
            from functools import wraps

            @wraps(func)
            def initialize(self, *args, **kwargs):
                init = getattr(self, 'init', ((), {}))
                return func(self, *(tuple(init[0]) + args), **{**init[1], **kwargs})

            return initialize

        cls.initialize = wrapper(cls.initialize)

        return cls


def get_engine(engine):
    """
    Return engine for emulation.

    Parameters
    ----------
    engine : type, BaseEmulatorEngine, str
        Engine (type or instance) or one of ['taylor', 'mlp'].

    Returns
    -------
    engine : BaseEmulatorEngine
    """
    if isinstance(engine, str):
        engine = engine.lower()
        if engine == 'taylor':
            from . import taylor
        elif engine == 'mlp':
            from . import mlp

        try:
            engine = BaseEmulatorEngine._registry[engine]()
        except KeyError:
            raise ValueError('Unknown engine {}.'.format(engine))

    if isinstance(engine, type):
        engine = engine()
    return engine


class BaseEmulatorEngine(BaseClass, metaclass=RegisteredEmulatorEngine):

    """Base class for emulator engine."""

    name = 'base'

    def __init__(self, *args, **kwargs):
        self.init = (args, kwargs)

    def initialize(self, varied_params):
        pass

    def get_default_samples(self, calculator):
        raise NotImplementedError

    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class PointEmulatorEngine(BaseEmulatorEngine):

    """Basic emulator that returns constant prediction."""
    name = 'point'

    def get_default_samples(self, calculator):
        from desilike.samplers import GridSampler
        sampler = GridSampler(calculator, size=1)
        sampler.run()
        return sampler.samples

    def fit(self, X, Y):
        self.point = np.asarray(self.mpicomm.bcast(Y[0] if self.mpicomm.rank == 0 else None, root=0))

    def predict(self, X):
        # Dumb prediction
        return self.point

    def __getstate__(self):
        state = {}
        for name in ['point']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
