from desilike import setup_logging

from desilike.bindings.montepython.factory import MontePythonLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood, TestEmulatedFullKaiserLikelihood


def test_generate_likelihood():

    setup_logging('warning')
    MontePythonLikelihoodGenerator()([TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood, TestEmulatedFullKaiserLikelihood])


if __name__ == '__main__':

    test_generate_likelihood()
