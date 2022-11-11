from desilike import setup_logging

from desilike.bindings.montepython.factory import MontePythonLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood


def test_generate_likelihood():

    setup_logging('debug')
    MontePythonLikelihoodGenerator()(TestSimpleLikelihood)


if __name__ == '__main__':

    test_generate_likelihood()
