from desilike import setup_logging

from desilike.bindings.cosmosis.factory import CosmoSISLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood, TestEmulatedFullKaiserLikelihood


def test_generate_likelihood():

    setup_logging('warning')
    CosmoSISLikelihoodGenerator()([TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood, TestEmulatedFullKaiserLikelihood])


if __name__ == '__main__':

    test_generate_likelihood()
