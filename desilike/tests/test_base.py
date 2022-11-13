from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate


def test_base():

    theory = KaiserTracerPowerSpectrumMultipoles()
    exit()
    print(theory.runtime_info.pipeline.params)
    theory(sigma8=0.9, b1=1.).power
    theory.template(b1=1.).pk_tt
    print(theory.template.runtime_info.pipeline.params)
    theory.template = ShapeFitPowerSpectrumTemplate(k=theory.kin)
    print(theory.runtime_info.pipeline.params)
    print(theory(dm=0.01, qpar=0.99).power)


if __name__ == '__main__':

    test_base()
