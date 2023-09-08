import numpy as np

from desilike.emulators import Emulator, MLPEmulatorEngine
from desilike.base import BaseCalculator
from desilike import setup_logging


class LinearModel(BaseCalculator):

    _params = {'a': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}, 'b': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}}
    #_params = {'a': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}}
    #_params = {'b': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}}

    def initialize(self):
        self.x = np.linspace(0.1, 1.1, 11)

    def calculate(self, a=0., b=0.):
        self.model = a * self.x + b

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['x', 'model']}


def test_mlp_linear(plot=False):
    calculator = LinearModel()
    emulator = Emulator(calculator, engine=MLPEmulatorEngine(nhidden=(), npcs=3))
    emulator.set_samples(niterations=int(1e5))
    emulator.fit(batch_sizes=(10000,), epochs=1000, learning_rates=None)
    emulator.check(frac=0.5)
    emulated_calculator = emulator.to_calculator()

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for i, dx in enumerate(np.linspace(-1., 1., 5)):
            calculator(**{str(param): param.value + dx for param in calculator.varied_params})
            emulated_calculator(**{str(param): param.value + dx for param in emulated_calculator.varied_params})
            color = 'C{:d}'.format(i)
            ax.plot(calculator.x, calculator.model, color=color, linestyle='--')
            ax.plot(emulated_calculator.x, emulated_calculator.model, color=color, linestyle='-')
        plt.show()


def test_mlp(plot=False):
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    calculator = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate())
    power_bak = calculator().copy()
    emulator = Emulator(calculator, engine='mlp')
    emulator.set_samples(niterations=int(1e3))
    emulator.fit()
    emulator.check(frac=1.)

    calculator = emulator.to_calculator()
    calculator(**{str(param): param.value for param in calculator.varied_params})
    calculator(**{str(param): param.value * 1.1 for param in calculator.varied_params})
    assert not np.allclose(calculator.power, power_bak)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(calculator.ells):
            color = 'C{:d}'.format(ill)
            ax.plot(calculator.k, calculator.k * power_bak[ill], color=color, label=r'$\ell = {:d}$'.format(ell))
            ax.plot(calculator.k, calculator.k * calculator.power[ill], color=color, linestyle='--')
        plt.show()


if __name__ == '__main__':

    setup_logging()
    test_mlp_linear(plot=True)
    test_mlp(plot=True)
