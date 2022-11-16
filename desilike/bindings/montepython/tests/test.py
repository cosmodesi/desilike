from desilike import setup_logging

from desilike.bindings.montepython.factory import MontePythonLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood


def test_generate_likelihood():

    for cls in [TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood]:
        setup_logging('warning')
        MontePythonLikelihoodGenerator()(cls)


if __name__ == '__main__':

    test_generate_likelihood()
