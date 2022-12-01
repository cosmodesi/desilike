if __name__ == '__main__':

    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator
    from desilike.bindings.y1kp45_mock_challenge import AbacusSummitLRGFullPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles

    save_emulator = False
    save_external = True

    emulator_fn = '_tests/emulator.npy'
    if save_emulator:
        AbacusSummitLRGShapeFitPowerSpectrumMultipoles(emulator_fn=emulator_fn, save_emulator=True)

    if save_external:
        likelihoods = [AbacusSummitLRGFullPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles]
        setup_logging('info')
        CobayaLikelihoodGenerator()(likelihoods)
        CosmoSISLikelihoodGenerator()(likelihoods)
        MontePythonLikelihoodGenerator()(likelihoods)
