# NOTE: This is automatically generated code by desilike.bindings.cosmosis.factory.CosmoSISLikelihoodGenerator
from desilike.bindings.cosmosis.factory import CosmoSISLikelihoodFactory

from desilike.bindings.tests import TestSimpleLikelihood
TestSimpleLikelihood = CosmoSISLikelihoodFactory(TestSimpleLikelihood, {}, __name__)

setup, execute, cleanup = TestSimpleLikelihood.build_module()

