# NOTE: This is automatically generated code by desilike.bindings.cosmosis.factory.CosmoSISLikelihoodGenerator
from desilike.bindings.cosmosis.factory import CosmoSISLikelihoodFactory

from desilike.bindings.tests import TestFullKaiserLikelihood
TestFullKaiserLikelihood = CosmoSISLikelihoodFactory(TestFullKaiserLikelihood, {}, __name__)

setup, execute, cleanup = TestFullKaiserLikelihood.build_module()

