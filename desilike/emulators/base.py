import os

import numpy as np

from desilike import utils, plotting
from desilike.base import BaseCalculator
from desilike.utils import BaseClass, OrderedSet, serialize_class, import_class, jnp
from desilike.io import BaseConfig
from desilike.parameter import Parameter, ParameterArray, Samples, ParameterCollection, ParameterConfig, find_names


def find_uniques(li):
    toret = []
    for el in li:
        if el not in toret:
            toret.append(el)
    return toret


def get_subsamples(samples, frac=1., nmax=np.inf, seed=42, mpicomm=None):
    nsamples = mpicomm.bcast(samples.size if mpicomm.rank == 0 else None)
    size = min(int(nsamples * frac + 0.5), nmax)
    if size > nsamples:
        raise ValueError('Cannot use {:d} subsamples (> {:d} total samples)'.format(size, nsamples))
    rng = np.random.RandomState(seed=seed)
    toret = None
    if mpicomm.rank == 0:
        toret = samples.ravel()[rng.choice(nsamples, size=size, replace=False)]
    return toret


def _setstate(self, state):
    calc_state = {name: value for name, value in state.items() if name in self.emulator.in_calculator}
    self.__dict__.update(state)
    self.__setstate__(calc_state)


class EmulatedCalculator(BaseCalculator):

    def initialize(self, **kwargs):
        pass

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

    def __init__(self, calculator, engine='taylor', mpicomm=None):
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.calculator = calculator
        self.pipeline = self.calculator.runtime_info.pipeline

        self.params = self.pipeline.params.deepcopy()
        for param in self.params: param.update(drop=False)  # dropped params become actual params
        self.varied_params = self.params.names(varied=True, derived=False)
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')

        calculators = []
        for calculator in self.pipeline.calculators:
            if calculator.runtime_info.derived_params and calculator is not self.calculator:
                calculators.append(calculator)

        self.calculator.runtime_info.derived_auto = OrderedSet('.fixed', '.varied')
        calculators.append(self.calculator)
        calculators, fixed, varied = self.pipeline._set_derived_auto(calculators)
        self.fixed, self.varied = {}, OrderedSet()

        for cc, ff, vv in zip(calculators, fixed, varied):
            bp = cc.runtime_info.base_params
            self.fixed.update({k: v for k, v in ff.items() if k in bp and bp[k].derived})
            self.varied |= OrderedSet(k for k in vv if k in bp and bp[k].derived)
        self.in_calculator = set(ff.keys()) | set(vv)
        self.varied = list(self.varied)

        if mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params))
            self.log_info('Found varying {} and fixed {} outputs.'.format(self.varied, list(self.fixed.keys())))

        # Add in cosmo_requires
        self.fixed['cosmo_requires'] = self.calculator.runtime_info.pipeline.get_cosmo_requires()

        if not isinstance(engine, dict):
            engine = {'*': engine}

        self.engines = {name: None for name in self.varied}
        for template, engine in engine.items():
            engine = get_engine(engine)
            for tmpname in find_names(self.varied, template):
                self.engines[tmpname] = engine
        for name, engine in self.engines.items():
            if engine is None:
                raise ValueError('Engine not specified for varying attribute {}'.format(name))
            engine.initialize(varied_params=self.varied_params)

        self.calculator__class__ = serialize_class(calculator.__class__)
        self.yaml_data = BaseConfig()
        self.yaml_data['class'] = calculator.__class__.__name__
        self.yaml_data['info'] = dict(calculator.info)
        #self.yaml_data['init'] = dict(calculator.runtime_info.init)
        params = {}
        for param in self.params:
            params[param.name] = dict(ParameterConfig(param))
            params[param.name].pop('basename')
        self.yaml_data['params'] = params
        self.varied_shape = {name: -1 for name in self.engines}
        self.samples, self.diagnostics = {}, {}
        self.mpicomm = mpicomm

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        try:
            for engine in self.engines.values():
                engine.mpicomm = mpicomm
        except AttributeError:
            pass

    def set_samples(self, name=None, samples=None, **kwargs):
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
            for param in self.pipeline.params.select(varied=True, derived=False):
                tmp.set(ParameterArray(samples[param], param=param))
            for param in self.pipeline.params.select(name=list(self.engines.keys()), derived=True):
                tmp.set(ParameterArray(samples[param], param=param))
        for name, eng in self.engines.items():
            if eng is engine:
                self.samples[name] = tmp

    def fit(self, name=None, **kwargs):

        def _get_X_Y(samples, yname):
            X, Y = None, None
            if self.mpicomm.rank == 0:
                nsamples = samples.size
                X = np.concatenate([samples[name].reshape(nsamples, 1) for name in self.varied_params], axis=-1)
                self.varied_shape[name] = samples[yname].shape[samples.ndim:]
                Y = samples[yname].reshape(nsamples, -1)
            self.varied_shape[name] = self.mpicomm.bcast(self.varied_shape[name], root=0)
            return X, Y

        if name is None:
            name = list(self.engines.keys())
        if not utils.is_sequence(name):
            name = [name]
        names = name

        for name in names:
            self.engines[name] = engine = self.engines[name].copy()
            engine.fit(*_get_X_Y(self.samples[name], name), **kwargs)

    def predict(self, **params):
        X = jnp.array([params[name] for name in self.varied_params])
        return {name: engine.predict(X).reshape(self.varied_shape[name]) for name, engine in self.engines.items()}

    def to_calculator(self, derived=None):

        state = self.__getstate__()
        Calculator = import_class(*state['calculator__class__'])
        new_name = Calculator.__name__

        new_cls = type(EmulatedCalculator)(new_name, (EmulatedCalculator, Calculator),
                                           {'__setstate__': Calculator.__setstate__, 'get': Calculator.get, '__module__': Calculator.__module__})
        try:
            new_cls.config_fn = Calculator.config_fn
        except AttributeError:
            pass

        calculator = new_cls()
        calculator.emulator = self
        params = self.params.deepcopy()
        if derived is not None:
            for param in derived:
                param = Parameter(param, derived=True)
                if param not in params:
                    params.set(param)
        calculator.params = params
        _setstate(calculator, self.fixed)
        return calculator

    def check(self, mse_stop=None, diagnostics=None, frac=0.1, **kwargs):

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
                        mse[name] = np.mean((derived['emulator.' + name] - subsamples[name])**2)
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

    def plot(self, fn=None, name=None, kw_save=None, nmax=100, show=False, **kwargs):
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

        toret = []
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
                        for d, s in zip(derived['emulator.' + name], subsamples[name]):
                            lax[0].plot(d.ravel(), color='k', marker='+', markersize=1, alpha=0.2)
                            lax[1].plot((d - s).ravel(), color='k', marker='+', markersize=1, alpha=0.2)
                        lax[0].set_ylabel(name)
                        lax[1].set_ylabel(r'$\Delta$ {}'.format(name))
                        for ax in lax: ax.grid(True)
                        toret.append(lax)
                        if fn is not None:
                            plotting.savefig(fn, fig=fig, **(kw_save or {}))
                        if show is None:
                            plt.show()
        return toret

    def __getstate__(self):
        state = {'engines': {}}
        for name, engine in self.engines.items():
            state['engines'][name] = {'__class__': serialize_class(engine.__class__), **engine.__getstate__()}
        for name in ['varied_params', 'fixed', 'varied_shape', 'in_calculator', 'calculator__class__']:
            state[name] = getattr(self, name)
        state['yaml_data'] = self.yaml_data.data
        state['params'] = self.params.__getstate__()
        return state

    def save(self, filename, yaml=True):
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        if yaml:
            state['config_fn'] = fn = os.path.splitext(filename)[0] + '.yaml'
            self.yaml_data.write(fn)
        np.save(filename, state, allow_pickle=True)

    def __setstate__(self, state):
        super(Emulator, self).__setstate__(state)
        self.yaml_data = BaseConfig(self.yaml_data)
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
    Return engine (class) for cosmological calculation.

    Parameters
    ----------
    engine : BaseEngine, string
        Engine or one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'eisenstein_hu_nowiggle_variants', 'bbks'].

    Returns
    -------
    engine : BaseEngine
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

    name = 'point'

    def get_default_samples(self, calculator):
        from desilike.samplers import GridSampler
        sampler = GridSampler(calculator, ngrid=2)
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
