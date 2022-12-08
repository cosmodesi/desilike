if __name__ == '__main__':

    import os

    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator
    from desilike.bindings.y1kp45_mock_challenge import AbacusSummitLRGFullPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles

    save_emulator = True
    save_external = True
    dirname = os.path.dirname(__file__)

    Likelihoods = [AbacusSummitLRGFullPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles]
    emulator_fns = [os.path.join(dirname, Like.__name__ + '.npy') for Like in Likelihoods]
    kw_likes = [{'emulator_fn': emulator_fn} for emulator_fn in emulator_fns]
    #kw_likes = None  # to deactivate emulators

    if save_emulator:
        for Like, emulator_fn in zip(Likelihoods, emulator_fns):
            Like(emulator_fn=emulator_fn, save_emulator=True)

    if save_external:
        setup_logging('info')
        CobayaLikelihoodGenerator()(Likelihoods, kw_likes=kw_likes)
        CosmoSISLikelihoodGenerator()(Likelihoods, kw_likes=kw_likes)
        MontePythonLikelihoodGenerator()(Likelihoods, kw_likes=kw_likes)
