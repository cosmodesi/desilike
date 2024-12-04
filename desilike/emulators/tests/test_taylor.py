import numpy as np

from desilike.jax import numpy as jnp
from desilike.parameter import Parameter
from desilike.emulators import Emulator, TaylorEmulatorEngine
from desilike.base import BaseCalculator
from desilike import setup_logging


class PowerModel(BaseCalculator):

    def initialize(self, order=2):
        self.x = np.linspace(0.1, 1.1, 11)
        self.order = order
        for i in range(self.order):
            self.params.set(Parameter('a{:d}'.format(i), value=1.5, ref={'limits': [1., 3.]}, delta=0.1))

    def calculate(self, **kwargs):
        #self.model = sum(kwargs['a{:d}'.format(i)] * self.x**i for i in range(self.order))
        self.model = jnp.prod(jnp.array([kwargs['a{:d}'.format(i)]**(i + 2) for i in range(self.order)])) * self.x

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['x', 'model']}


def test_taylor_power(plot=False):

    for order in [3, 4][:1]:
        calculator = PowerModel()
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=order))
        emulator.set_samples()
        emulator.fit()
        emulator.check()
        emulator.plot(show=plot)

        emulated_calculator = emulator.to_calculator()

        #from desilike import Differentiation

        #def getter():
        #    return emulated_calculator.model

        #d = Differentiation(emulated_calculator, getter, order=1)()
        #assert not np.isnan(d).any()

        mpicomm = emulated_calculator.mpicomm
        if mpicomm.rank == 0 and plot:
            from matplotlib import pyplot as plt
            ax = plt.gca()
            for i, dx in enumerate(np.linspace(-1., 1., 5)):
                calculator(**{str(param): param.value + dx for param in calculator.varied_params})
                emulated_calculator(**{str(param): param.value + dx for param in emulated_calculator.varied_params})
                color = 'C{:d}'.format(i)
                ax.plot(calculator.x, calculator.model, color=color, linestyle='--')
                ax.plot(emulated_calculator.x, emulated_calculator.model, color=color, linestyle='-')
            plt.show()

    for order in [3, 4, 7][2:]:
        calculator = PowerModel()
        for param in calculator.all_params: param.update(value=1.1, prior={'limits': [1., 2.]})
        #calculator.all_params['a1'].update(fixed=True)
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=order, accuracy={'*': 2, 'a1': 4}))
        emulator.set_samples(method='finite')
        emulator.fit()
        emulator.check()

        emulated_calculator = emulator.to_calculator()
        from desilike import Differentiation

        def getter():
            return emulated_calculator.model

        deriv = Differentiation(emulated_calculator, getter, order=1)()
        mpicomm = emulated_calculator.mpicomm
        if mpicomm.rank == 0:
            assert np.isfinite(deriv).all()

            if plot:
                from matplotlib import pyplot as plt
                ax = plt.gca()
                for i, dx in enumerate(np.linspace(1., 4., 5)):
                    calculator(**{str(param): param.value + dx for param in calculator.varied_params})
                    emulated_calculator(**{str(param): param.value + dx for param in emulated_calculator.varied_params})
                    color = 'C{:d}'.format(i)
                    ax.plot(calculator.x, calculator.model, color=color, linestyle='--')
                    ax.plot(emulated_calculator.x, emulated_calculator.model, color=color, linestyle='-')
                plt.show()


def test_taylor(plot=False):
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    calculator = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate())
    power_bak = calculator().copy()
    emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples()
    emulator.fit()

    calculator(**{str(param): param.value for param in calculator.params if param.varied})
    assert np.allclose(calculator.power, power_bak)
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

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    kwargs = dict(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                  data='../../tests/_pk/data.npy', covariance='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy')
    observable = TracerPowerSpectrumMultipolesObservable(theory=theory, **kwargs)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()  # needed to set everything up

    emulator = Emulator(theory, engine=TaylorEmulatorEngine(order=2))
    emulator.set_samples()
    emulator.fit()
    theory = emulator.to_calculator()

    observable = TracerPowerSpectrumMultipolesObservable(theory=theory, **kwargs)
    likelihood2 = ObservablesGaussianLikelihood(observables=[observable])

    for param in likelihood.varied_params:
        for scale in [-0.1, 0.1]:
            params = {param.name: param.value + param.proposal * scale}
            print(param, (likelihood(**params), likelihood2(**params)))


def test_pt():
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.theories import Cosmoprimo
    cosmo = Cosmoprimo(fiducial='DESI', h=0.7)
    #cosmo.init.params['h'].update(delta=(0.7,) + cosmo.init.params['h'].delta[1:])
    temp = DirectPowerSpectrumTemplate(cosmo=cosmo, z=0.8)
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=temp)
    theory()

    emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=1))
    emulator.set_samples()
    emulator.fit()
    pt = emulator.to_calculator()



if __name__ == '__main__':

    setup_logging()

    test_pt()
    test_taylor_power(plot=True)
    test_taylor(plot=True)
    test_likelihood()
    #test_pt()
