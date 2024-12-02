import os
import re
import itertools

import numpy as np
from cosmoprimo.emulators.tools import *
from cosmoprimo.emulators import tools

from desilike import utils, plotting, mpi
from desilike.base import CollectionCalculator, BaseCalculator, vmap
from desilike.utils import serialize_class, import_class
from desilike.io import BaseConfig
from desilike.parameter import ParameterConfig, ParameterCollection, Deriv


def find_uniques(li):
    toret = []
    for el in li:
        if el not in toret:
            toret.append(el)
    return toret


def _get_subsamples(samples, frac=1., nmax=np.inf, seed=42, mpicomm=None):
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


def _get_calculator_info(calc, collection_calculator=None):
    if collection_calculator is None: collection_calculator = calc
    calculator__class__ = serialize_class(calc.__class__)
    yaml_data = BaseConfig()
    yaml_data['class'] = calc.__class__.__name__
    yaml_data['info'] = dict(calc.info)
    #self.yaml_data['init'] = dict(calculator.runtime_info.init)
    params = {}
    for param in collection_calculator.all_params:
        if param in calc.all_params or not any(param in cc.all_params for cc in collection_calculator.calculators if cc is not calc):  # last one is fore parameters defined as calculator.all_params = ...
            basename = param.basename
            params[basename] = dict(ParameterConfig(param))
            params[basename].pop('basename')
            params[basename].pop('namespace', None)
    yaml_data['params'] = params
    return calculator__class__, yaml_data


class Emulator(tools.Emulator):

    """Subclass :class:`tools.Emulator` to be able to provide a desilike calculator as calculator."""

    def __init__(self, calculator, samples=None, engine=None, xoperation=None, yoperation=None, mpicomm=None, **kwargs):
        if mpicomm is None:
            mpicomm = calculator[0].mpicomm if utils.is_sequence(calculator) else calculator.mpicomm
        self._calculator, self._params, self._varied, self._fixed = self._get_calculator(calculator, **kwargs)
        super().__init__(samples=samples, engine=engine, xoperation=xoperation, yoperation=yoperation, mpicomm=mpicomm, **kwargs)

    def _get_calculator(self, calculator, **kwargs):
        is_calculator_sequence = utils.is_sequence(calculator)
        if is_calculator_sequence:
            is_calculator_sequence = len(calculator)
            calculator = CollectionCalculator(calculators=calculator)
        if isinstance(calculator, CollectionCalculator):
            is_calculator_sequence = len(calculator.all_calculators)
        pipeline = calculator.runtime_info.pipeline
        params = pipeline.params.clone(namespace=None).deepcopy()
        varied_params = pipeline.varied_params.clone(namespace=None).names()

        if not varied_params:
            raise ValueError('No parameters to be varied!')
        calculators, _fixed, _varied = pipeline._classify_derived(pipeline.calculators)
        fixed, varied = {}, {}

        toderive = []
        for cc, ff, vv in zip(calculators[:-1], _fixed, _varied):
            base_names = cc.runtime_info.base_names
            fff, vvv = {}, {}
            for k, v in ff.items():
                if k in base_names and params[base_names[k]].derived: fff[base_names[k]] = v
            for k in vv:
                if k in base_names and params[base_names[k]].derived: vvv[base_names[k]] = None
            if is_calculator_sequence:
                fff = {name: value for name, value in fff.items() if re.match(r'(\d*)_(.*)', name)}
                vvv = {name: value for name, value in vvv.items() if re.match(r'(\d*)_(.*)', name)}
            fixed.update(fff)
            varied.update(vvv)
            toderive.append(list(vvv))
        fixed.update(_fixed[-1])
        varied.update(dict.fromkeys(_varied[-1]))
        toderive.append(_varied[-1])
        pipeline._set_derived(calculators, params=toderive)

        in_calculator_state = list(_fixed[-1]) + _varied[-1]
        varied = list(varied)

        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(varied_params))
            self.log_info('Found varying {} and fixed {} outputs.'.format(varied, list(fixed)))
        if not varied:
            raise ValueError('Found no varying quantity in provided calculator')

        if is_calculator_sequence:
            calculator__class__, yaml_data = [], []
            for name, calc in calculator.items():
                _calculator__class__, _yaml_data = _get_calculator_info(calc, calculator)
                calculator__class__.append(_calculator__class__)
                yaml_data.append(_yaml_data)
                fixed[name + '_cosmo_requires'] = calc.runtime_info.pipeline.get_cosmo_requires()
        else:
            calculator__class__, yaml_data = _get_calculator_info(calculator)
            # Add in cosmo_requires
            fixed['cosmo_requires'] = calculator.runtime_info.pipeline.get_cosmo_requires()

        self.__dict__.update({'is_calculator_sequence': is_calculator_sequence, 'yaml_data': yaml_data, 'all_params': params,
                              'varied_params': varied_params, 'calculator__class__': calculator__class__, 'in_calculator_state': in_calculator_state})
        return calculator, {name: None for name in varied_params}, varied, fixed

    def _get_samples(self, samples):
        from desilike.samples import Samples
        samples = samples if isinstance(samples, Samples) else Samples.load(samples)
        return samples.ravel()

    def _get_X_Y_from_samples(self, samples, **kwargs):
        from cosmoprimo.emulators import Samples
        varied_names = samples.names(varied=True, derived=False)
        samples = Samples({'X.' + name: samples[name] for name in varied_names} | {'Y.' + name: samples[name] for name in samples.names() if name not in varied_names})
        return super()._get_X_Y_from_samples(samples, **kwargs)

    def _sort_varied_fixed(self, samples, subsample=None):
        varied, fixed = super()._sort_varied_fixed(samples, subsample)
        for name in fixed:
            if getattr(fixed[name], 'derivs', []):
                varied[name] = fixed.pop(name).shape
        return varied, fixed

    def _get_varied_params(self, name):
        name = str(name)
        if self.is_calculator_sequence:
            index = int(re.match(r'(\d*)(_.*|)$', name).group(1))
            return [name for name in self.varied_params if name in self.yaml_data[index]['params']]
        return self.varied_params.copy()

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
                emulator.xoperations = [operation.copy() for operation in self.xoperations]
                emulator.yoperations = [operation.copy() for operation in self.yoperations]
                emulator.defaults = dict(self.defaults)
                emulator.is_calculator_sequence = False
                emulator.yaml_data = self.yaml_data[index]
                emulator.all_params = self.all_params.select(basename=list(emulator.yaml_data['params'])).deepcopy()
                emulator.varied_params = self._get_varied_params(index)
                emulator.calculator__class__ = self.calculator__class__[index]
                index = str(index)

                def get_name(name):
                    return name[len(index) + 1:]

                emulator.fixed = {get_name(name): value for name, value in self.fixed.items() if name.startswith(index)}
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

        new_cls._params = self.all_params.select(depends={}).deepcopy()

        for param in new_cls._params: param.update(drop=False)

        calculator = new_cls(emulator=self)
        calculator.runtime_info.initialize()  # to initialize
        if derived is not None:
            calculator.runtime_info.pipeline._set_derived([calculator], params=[derived])

        return calculator

    def __getstate__(self):
        state = super().__getstate__()
        for name in ['varied_params', 'in_calculator_state', 'calculator__class__', 'is_calculator_sequence']:
            state[name] = getattr(self, name)
        state['yaml_data'] = [yaml_data.data for yaml_data in self.yaml_data] if self.is_calculator_sequence else self.yaml_data.data
        state['all_params'] = self.all_params.__getstate__()
        return state

    def __setstate__(self, state):
        # Backward-compatibility
        for name in ['xoperations', 'yoperations']:
            if name not in state: state[name] = []
        if 'params' in state:
            state['all_params'] = state.pop('params')
        varied_shape = state.pop('varied_shape', None)
        if varied_shape is not None:
            varied_params = list(state['varied_params'])
            for name, engine in state['engines'].items():
                engine['yshape'] = varied_shape[name]
                engine['xshape'] = (len(varied_params),)
                engine['name'] = engine.pop('__class__')[0].split('.')[-1][:-len('EmulatorEngine')].lower()
                engine['params'] = {param: None for param in varied_params}
                engine['xoperations'] = []
                engine['yoperations'] = []
            state['defaults'] = {param: None for param in varied_params}
        super().__setstate__(state)
        self.is_calculator_sequence = getattr(self, 'is_calculator_sequence', False)
        self.yaml_data = [BaseConfig(yaml_data) for yaml_data in self.yaml_data] if self.is_calculator_sequence else BaseConfig(self.yaml_data)
        self.all_params = ParameterCollection.from_state(state['all_params'])

    def save(self, filename, yaml=True):
        state = self.__getstate__()
        if self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            if yaml:
                state['config_fn'] = fn = os.path.splitext(filename)[0] + '.yaml'
                BaseConfig.write(self.yaml_data, fn)
            np.save(filename, state, allow_pickle=True)

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
            diagnostics = self.diagnostics = getattr(self, 'diagnostics', {})

        if self.mpicomm.rank == 0:
            self.log_info('Diagnostics:')

        toret = True
        calculator = self.to_calculator(derived=['emulator.{}'.format(name) for name in self.engines])
        calculator()

        unique_samples = find_uniques(self._samples.values())
        for samples in unique_samples:
            subsamples = _get_subsamples(samples, frac=frac, mpicomm=self.mpicomm, **kwargs)
            vcalculate = vmap(calculator, backend='mpi', return_derived=True)
            derived = vcalculate({name: subsamples[name] if self.mpicomm.rank == 0 else None for name in self.varied_params})[1]

            if self.mpicomm.rank == 0:

                def add_diagnostics(name, value):
                    if name not in diagnostics:
                        diagnostics[name] = [value]
                    else:
                        diagnostics[name].append(value)
                    return value

                item = '- '
                mse = {}
                for name in self._samples:
                    if self._samples[name] is samples:
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
        calculator()
        unique_samples = find_uniques(self._samples.values())

        figs = []
        for samples in unique_samples:
            samples_names = [name for name, s in self._samples.items() if s is samples]
            if samples_names:
                subsamples = _get_subsamples(samples, nmax=nmax, mpicomm=self.mpicomm, **kwargs)
                vcalculate = vmap(calculator, backend='mpi', return_derived=True)
                derived = vcalculate({name: subsamples[name] if self.mpicomm.rank == 0 else None for name in self.varied_params})[1]

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


def _setstate(self, state):
    self.__dict__.update({name: value for name, value in state.items() if name not in self.emulator.in_calculator_state})
    self.__setstate__({name: value for name, value in state.items() if name in self.emulator.in_calculator_state})


from desilike.base import BaseCalculator


class EmulatedCalculator(BaseCalculator):

    def initialize(self, emulator=None, **kwargs):
        self.emulator = emulator
        try:
            super()._emulator_initialize()
        except AttributeError:
            pass
        self.calculate(**{param.basename: param.value for param in self.init.params})
        # Hack to enforce parameters that are dropped in the non-emulated pipeline to be passed here
        for param in self.init.params:
            if param.name in self.emulator.varied_params:
                param.update(drop=False)

    def calculate(self, **params):
        _setstate(self, self.emulator.predict(params))

    @classmethod
    def load(cls, filename):
        if not os.path.exists(filename):  # pre-saved emulators
            filename = os.path.join(os.path.dirname(__file__), 'train', filename, 'emulator.npy')
        return Emulator.load(filename).to_calculator()

    def save(self, fn):
        return self.emulator.save(fn)


class PointEmulatorEngine(tools.PointEmulatorEngine):

    def get_default_samples(self, calculator, **kwargs):
        from desilike.samplers import GridSampler
        sampler = GridSampler(calculator, size=1)
        sampler.run()
        return sampler.samples


class TaylorEmulatorEngine(tools.TaylorEmulatorEngine):

    def __init__(self, *args, order=3, accuracy=2, method=None, delta_scale=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_options = dict(order=order, accuracy=accuracy, method=method, delta_scale=delta_scale)
        assert not self.xoperations, 'xoperations not supported for {} engine'.format(self.name)
        assert not self.yoperations, 'yoperations not supported for {} engine'.format(self.name)

    def get_default_samples(self, calculator, **kwargs):
        """
        Returns samples with derivatives.

        Parameters
        ----------
        order : int, dict, default=3
            A dictionary mapping parameter name (including wildcard) to maximum derivative order.
            If a single value is provided, applies to all varied parameters.

        accuracy : int, dict, default=2
            A dictionary mapping parameter name (including wildcard) to derivative accuracy (number of points used to estimate it).
            If a single value is provided, applies to all varied parameters.
            Not used if ``method = 'auto'``  for this parameter.

        delta_scale : float, default=1.
            Parameter grid ranges for the estimation of finite derivatives are inferred from parameters' :attr:`Parameter.delta`.
            These values are then scaled by ``delta_scale`` (< 1. means smaller ranges).
        """
        from desilike import Differentiation
        kwargs.pop('params', None)
        options = {**self.sampler_options, **kwargs}
        differentiation = Differentiation(calculator, **options, mpicomm=self.mpicomm)
        samples = differentiation(**differentiation._grid_center)
        return samples

    def fit(self, X, Y, attrs, **kwargs):
        # print('pre', Y.shape)
        if self.mpicomm.rank == 0:
            xshape, yshape = X.shape[1:], Y.shape[2:]
        self.xshape, self.yshape = self.mpicomm.bcast((xshape, yshape) if self.mpicomm.rank == 0 else None, root=0)
        self._fit_no_operation(X, Y, attrs, **kwargs)

    def _fit_no_operation(self, X, Y, attrs):
        if self.mpicomm.bcast(Y.derivs is None if self.mpicomm.rank == 0 else None, root=0):
            raise ValueError('Please provide samples with derivatives computed')
        self.center, self.derivatives, self.powers = None, None, None
        if self.mpicomm.rank == 0:
            self.center = np.array([np.median(np.unique(xx)) for xx in X.T])
            Y = Y[0]  # only need one element
            self.derivatives, self.powers = [], []
            ndim = len(self.params)
            max_order, max_param_order = 0, [0 for i in range(ndim)]
            for deriv in Y.derivs:
                for iparam, param in enumerate(self.params):
                    max_param_order[iparam] = max(max_param_order[iparam], deriv[param])
                    max_order = max(max_order, deriv.total())
            prefactor, degrees = 1., []
            for order in range(0, max_order + 1):
                if order: prefactor /= order
                for indices in itertools.product(range(ndim), repeat=order):
                    orders = np.bincount(indices, minlength=ndim).astype('i4')
                    if order and sum(orders) > min(order for o, order in zip(orders, max_param_order) if o):
                        continue
                    degree = Deriv(dict(zip(self.params, orders)))
                    if degree not in Y.derivs:
                        import warnings
                        warnings.warn("Derivative {} is missing, let's assume it is 0".format(degree))
                        continue
                    value = prefactor * Y[degree]
                    if degree in degrees:
                        self.derivatives[degrees.index(degree)] += value
                    else:
                        degrees.append(degree)
                        self.powers.append(orders)
                        self.derivatives.append(value)
            self.derivatives, self.powers = np.array(self.derivatives), np.array(self.powers)
        self.derivatives = mpi.bcast(self.derivatives if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0)
        self.powers = self.mpicomm.bcast(self.powers, root=0)
        self.center = self.mpicomm.bcast(self.center, root=0)


class MLPEmulatorEngine(tools.MLPEmulatorEngine):

    def get_default_samples(self, calculator, **kwargs):
        """
        Returns samples.

        Parameters
        ----------
        order : int, dict, default=3
            A dictionary mapping parameter name (including wildcard) to maximum derivative order.
            If a single value is provided, applies to all varied parameters.

        engine : str, default='rqrs'
            QMC engine, to choose from ['sobol', 'halton', 'lhs', 'rqrs'].

        niterations : int, default=300
            Number of samples to draw.
        """
        from desilike.samplers import QMCSampler
        kwargs.pop('params', None)
        options = dict(engine='rqrs', niterations=int(1e5)) | kwargs
        sampler = QMCSampler(calculator, engine=options['engine'], mpicomm=self.mpicomm)
        sampler.run(niterations=options['niterations'])
        return sampler.samples