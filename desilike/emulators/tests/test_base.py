import os

from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
from desilike.emulators.base import Emulator, EmulatedCalculator
from desilike import setup_logging


def test_base():
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    calculator = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate())

    emulator = Emulator(calculator, engine='point')
    emulator.set_samples()
    emulator.fit()
    emulator.save(fn)
    calculator = emulator.to_calculator()

    emulator = EmulatedCalculator.load(fn)
    emulator.runtime_info.initialize()
    print(emulator.runtime_info.varied_params)
    print(emulator.runtime_info.param_values)
    emulator()
    #print(emulator.params, emulator.runtime_info.init[2], emulator.runtime_info.params)
    emulator().shape
    emulator.save(fn)

    emulator = EmulatedCalculator.load(fn)


if __name__ == '__main__':

    setup_logging()
    test_base()
