if __name__ == '__main__':
    
    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator
    from desilike.bindings.y1kp45_mock_challenge import AbacusSummitLRGFullPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles


    for cls in [AbacusSummitLRGFullPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles]:
        setup_logging('info')
        CobayaLikelihoodGenerator()(cls)
        CosmoSISLikelihoodGenerator()(cls)
        MontePythonLikelihoodGenerator()(cls)