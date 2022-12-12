from desilike import setup_logging
from desilike.samplers import EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler


def test_ensemble():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import ObservedTracerPowerSpectrumMultipoles
    from desilike.likelihoods import GaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.marg')
    observable = ObservedTracerPowerSpectrumMultipoles(klim={0: [0.05, 0.2], 2: [0.05, 0.2]}, kstep=0.01,
                                                       data='../../tests/_pk/data.npy', mocks='../../tests/_pk/mock_*.npy', wmatrix='../../tests/_pk/window.npy',
                                                       theory=theory)
    likelihood = GaussianLikelihood(observables=[observable])
    for Sampler in [EmceeSampler, ZeusSampler, PocoMCSampler, MCMCSampler, StaticDynestySampler, DynamicDynestySampler, PolychordSampler]:
        sampler = Sampler(likelihood, save_fn='./_tests/chain_*.npy')
        sampler.run(max_iterations=100, check=True)


if __name__ == '__main__':

    setup_logging()
    test_ensemble()
