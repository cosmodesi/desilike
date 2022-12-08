# NOTE: This is automatically generated code by desilike.bindings.cosmosis.factory.CosmoSISLikelihoodGenerator
from desilike.bindings.cosmosis.factory import CosmoSISLikelihoodFactory

from desilike.bindings.tests import TestEmulatedFullKaiserLikelihood
TestEmulatedFullKaiserLikelihood = CosmoSISLikelihoodFactory(TestEmulatedFullKaiserLikelihood, {}, __name__)

setup, execute, cleanup = TestEmulatedFullKaiserLikelihood.build_module()

