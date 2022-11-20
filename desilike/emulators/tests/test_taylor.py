import numpy as np

from desilike.parameter import Parameter
from desilike.emulators import Emulator, TaylorEmulatorEngine
from desilike.base import BaseCalculator
from desilike import setup_logging


class PowerModel(BaseCalculator):

    def initialize(self, order=4):
        self.x = np.linspace(0.1, 1.1, 11)
        self.order = order
        for i in range(self.order):
            self.params.set(Parameter('a{:d}'.format(i), value=0.5, ref={'limits': [-2., 2.]}))

    def calculate(self, **kwargs):
        self.model = sum(kwargs['a{:d}'.format(i)] * self.x**i for i in range(self.order))

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['x', 'model']}


def test_taylor_power(plot=False):
    for order in range(3, 5):
        calculator = PowerModel()
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=order))
        emulator.set_samples()
        emulator.fit()
        emulator.check()

        emulated_calculator = emulator.to_calculator()

        if plot:
            from matplotlib import pyplot as plt
            ax = plt.gca()
            for i, dx in enumerate(np.linspace(-1., 1., 5)):
                calculator(**{str(param): param.value + dx for param in calculator.runtime_info.full_params if param.varied})
                emulated_calculator(**{str(param): param.value + dx for param in emulated_calculator.runtime_info.full_params if param.varied})
                color = 'C{:d}'.format(i)
                ax.plot(calculator.x, calculator.model, color=color, linestyle='--')
                ax.plot(emulated_calculator.x, emulated_calculator.model, color=color, linestyle='-')
            plt.show()


def test_taylor(plot=False):
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    calculator = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate())
    power_bak = calculator().power.copy()
    emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples()
    emulator.fit()

    calculator = emulator.to_calculator()
    calculator(**{str(param): param.value for param in calculator.params if param.varied})
    assert np.allclose(calculator.power, power_bak)
    calculator(**{str(param): param.value * 1.1 for param in calculator.params if param.varied})
    assert not np.allclose(calculator.power, power_bak)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(calculator.ells):
            color = 'C{:d}'.format(ill)
            ax.plot(calculator.k, calculator.k * power_bak[ill], color=color, label=r'$\ell = {:d}$'.format(ell))
            ax.plot(calculator.k, calculator.k * calculator.power[ill], color=color, linestyle='--')
        plt.show()


def test_likelihood():
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))

    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrum
    from desilike.likelihoods import GaussianLikelihood
    kwargs = dict(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                  data='../../tests/_pk/data.npy', mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy')
    observable = ObservedTracerPowerSpectrum(theory=theory, **kwargs)
    likelihood = GaussianLikelihood(observables=[observable])
    likelihood() # needed to set everything up

    emulator = Emulator(theory, engine=TaylorEmulatorEngine(order=2))
    emulator.set_samples()
    emulator.fit()
    theory = emulator.to_calculator()

    observable = ObservedTracerPowerSpectrum(theory=theory, **kwargs)
    likelihood2 = GaussianLikelihood(observables=[observable])

    for param in likelihood.varied_params:
        for scale in [-0.1, 0.1]:
            params = {param.name: param.value + param.proposal * scale}
            print(param, (likelihood(**params), likelihood2(**params)))



if __name__ == '__main__':

    setup_logging()
    #test_taylor_power(plot=False)
    #test_taylor(plot=False)
    test_likelihood()
