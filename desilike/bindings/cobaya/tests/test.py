from desilike import setup_logging

from desilike.bindings.cobaya.factory import CobayaLikelihoodFactory, CobayaLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood#, TestGalaxyPowerSpectrumLikelihood


def test_simple():
    CobayaLikelihood = CobayaLikelihoodFactory(TestSimpleLikelihood)
    params = CobayaLikelihood().params
    info = {'likelihood': {'test': {'external': CobayaLikelihood}}, 'theory': {}, 'params': params}
    # info = {'likelihood': {'test': {'external': CobayaLikelihood}}}

    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'a': 2., 'b': 1.}))


def test_galaxy_power_spectrum():
    CobayaLikelihood = CobayaLikelihoodFactory(TestGalaxyPowerSpectrumLikelihood)
    params = CobayaLikelihood().params
    info = {'likelihood': {'test': {'external': CobayaLikelihood}}, 'params': params}

    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'b': 2.}))


def test_generate_likelihood():

    setup_logging('debug')
    CobayaLikelihoodGenerator()(TestSimpleLikelihood)
    info = {'likelihood': {'desilike.bindings.cobaya.tests.TestSimpleLikelihood': None}}

    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'a': 2., 'b': 1.}))


if __name__ == '__main__':

    # test_simple()
    # test_galaxy_power_spectrum()
    test_generate_likelihood()
