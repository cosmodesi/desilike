def AbacusSummitLRGDirectPowerSpectrumMultipoles(cosmo='external', solve=None, save_emulator=False, emulator_fn=None):

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    if save_emulator or emulator_fn is None:
        from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
        template = DirectPowerSpectrumTemplate(z=0.8, cosmo=None if save_emulator and cosmo == 'external' else cosmo)
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
        #from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
        #theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    else:
        from desilike.emulators import EmulatedCalculator
        pt = EmulatedCalculator.load(emulator_fn)
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt=pt)
    if solve is None:
        from desilike.utils import jax
        solve = jax is not None
    if solve and not save_emulator:
        for param in theory.params.select(name=['alpha*', 'sn*']): param.update(derived='.marg')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    """
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, kstep=0.005,
                                                         data='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Pk/Pre/jmena/nmesh_512/pypower_format/Pk_AbacusSummit_base_*.npy',
                                                         mocks='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CubicBox/LRG/Pk/jmena/nmesh_512/pypower_format/Pk_EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed*.npy',
                                                         wmatrix='/global/cfs/cdirs/desi/users/adematti/desi_mock_challenge/FirstGenMocks/AbacusSummit/CubicBox/ELG/z1.100/window_nmesh512_los-x.npy',
                                                         theory=theory)
    """
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, kstep=0.005,
                                                         data='../../tests/_pk/data.npy',
                                                         mocks='../../tests/_pk/mock_*.npy',
                                                         wmatrix='../../tests/_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    if save_emulator:
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        likelihood()
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.check()
        emulator.save(emulator_fn)
    return likelihood


def AbacusSummitLRGShapeFitPowerSpectrumMultipoles(solve=None, save_emulator=False, emulator_fn=None):

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    if save_emulator or emulator_fn is None:
        from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
        template = ShapeFitPowerSpectrumTemplate(z=0.8)
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
        #from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
        #theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    else:
        from desilike.emulators import EmulatedCalculator
        pt = EmulatedCalculator.load(emulator_fn)
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt=pt)
    if solve is None:
        from desilike.utils import jax
        solve = jax is not None
    if solve and not save_emulator:
        for param in theory.params.select(name=['alpha*', 'sn*']): param.update(derived='.marg')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    """
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, kstep=0.005,
                                                         data='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Pk/Pre/jmena/nmesh_512/pypower_format/Pk_AbacusSummit_base_*.npy',
                                                         mocks='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CubicBox/LRG/Pk/jmena/nmesh_512/pypower_format/Pk_EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed*.npy',
                                                         wmatrix='/global/cfs/cdirs/desi/users/adematti/desi_mock_challenge/FirstGenMocks/AbacusSummit/CubicBox/ELG/z1.100/window_nmesh512_los-x.npy',
                                                         theory=theory)
    """
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, kstep=0.005,
                                                         data='../../tests/_pk/data.npy',
                                                         mocks='../../tests/_pk/mock_*.npy',
                                                         wmatrix='../../tests/_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    if save_emulator:
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        likelihood()
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.check()
        emulator.save(emulator_fn)
    return likelihood


if __name__ == '__main__':

    import os

    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator

    save_emulator = False
    save_external = True
    emulator_dir = os.path.join(os.path.dirname(__file__), 'emulators')

    Likelihoods = [AbacusSummitLRGDirectPowerSpectrumMultipoles, AbacusSummitLRGShapeFitPowerSpectrumMultipoles]
    emulator_fns = [os.path.join(emulator_dir, Like.__name__ + '.npy') for Like in Likelihoods]
    kw_likes = [{'emulator_fn': emulator_fn} for emulator_fn in emulator_fns]
    kw_likes = None  # to deactivate emulators

    if save_emulator:
        for Like, emulator_fn in zip(Likelihoods, emulator_fns):
            Like(emulator_fn=emulator_fn, save_emulator=True)

    if save_external:
        setup_logging('info')
        CobayaLikelihoodGenerator()(Likelihoods, kw_like=kw_likes)
        CosmoSISLikelihoodGenerator()(Likelihoods, kw_like=kw_likes)
        MontePythonLikelihoodGenerator()(Likelihoods, kw_like=kw_likes)
