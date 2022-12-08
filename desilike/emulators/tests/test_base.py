import os

from desilike.theories.galaxy_clustering import (KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles,
                                                 FullPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate)
from desilike.emulators.base import Emulator, EmulatedCalculator
from desilike import setup_logging


def test_base():
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')

    for Template in [FullPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate]:
        calculator = KaiserTracerPowerSpectrumMultipoles(template=Template())

        emulator = Emulator(calculator, engine='point')
        emulator.set_samples()
        emulator.fit()
        emulator.save(fn)
        calculator = emulator.to_calculator()

        emulator = EmulatedCalculator.load(fn)
        emulator.runtime_info.initialize()
        print(emulator.runtime_info.varied_params)
        print(emulator.runtime_info.param_values)
        print(emulator.runtime_info.pipeline.get_cosmo_requires())
        emulator()
        #print(emulator.params, emulator.runtime_info.init[2], emulator.runtime_info.params)
        emulator().shape
        emulator.save(fn)

        emulator = EmulatedCalculator.load(fn)

    calculator = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate())
    calculator()
    pt = calculator.pt
    emulator = Emulator(pt, engine='point')
    emulator.set_samples()
    emulator.fit()
    emulator.save(fn)
    pt = EmulatedCalculator.load(fn)
    calculator.update(pt=pt)
    calculator(f=0.8)


if __name__ == '__main__':

    setup_logging()
    test_base()
