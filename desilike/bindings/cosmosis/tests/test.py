from desilike import setup_logging

from desilike.bindings.cosmosis.factory import CosmoSISLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood


def test_generate_likelihood():

    setup_logging('debug')
    CosmoSISLikelihoodGenerator()(TestSimpleLikelihood)


if __name__ == '__main__':

    test_generate_likelihood()
