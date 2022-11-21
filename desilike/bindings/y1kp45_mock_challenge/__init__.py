def AbacusSummitLRGFullPowerSpectrumMultipoles(cosmo='external', solve=None):

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, FullPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood

    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=FullPowerSpectrumTemplate(z=0.8, cosmo=cosmo))
    if solve is None:
        from desilike.utils import jax
        solve = jax is not None
    if solve:
        for param in theory.params.select(name=['alpha*', 'sn*']): param.derived = '.marg'
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, kstep=0.005,
                                                       data='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Pk/Pre/jmena/nmesh_512/pypower_format/Pk_AbacusSummit_base_*.npy',
                                                       mocks='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CubicBox/LRG/Pk/jmena/nmesh_512/pypower_format/Pk_EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed*.npy',
                                                       wmatrix='/global/cfs/cdirs/desi/users/adematti/desi_mock_challenge/FirstGenMocks/AbacusSummit/CubicBox/ELG/z1.100/window_nmesh512_los-x.npy',
                                                       theory=theory)
    return GaussianLikelihood(observables=[observable])


def AbacusSummitLRGShapeFitPowerSpectrumMultipoles(solve=None):

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood

    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.8))
    if solve is None:
        from desilike.utils import jax
        solve = jax is not None
    if solve:
        for param in theory.params.select(name=['alpha*', 'sn*']): param.derived = '.marg'
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.02, 0.2], 2: [0.02, 0.2]}, kstep=0.005,
                                                       data='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/LRG/Pk/Pre/jmena/nmesh_512/pypower_format/Pk_AbacusSummit_base_*.npy',
                                                       mocks='/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CubicBox/LRG/Pk/jmena/nmesh_512/pypower_format/Pk_EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed*.npy',
                                                       wmatrix='/global/cfs/cdirs/desi/users/adematti/desi_mock_challenge/FirstGenMocks/AbacusSummit/CubicBox/ELG/z1.100/window_nmesh512_los-x.npy',
                                                       theory=theory)
    return GaussianLikelihood(observables=[observable])
