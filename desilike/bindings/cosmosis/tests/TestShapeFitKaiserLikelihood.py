# NOTE: This is automatically generated code by desilike.bindings.cosmosis.factory.CosmoSISLikelihoodGenerator
from desilike.bindings.cosmosis.factory import CosmoSISLikelihoodFactory

from desilike.bindings.tests import TestShapeFitKaiserLikelihood
TestShapeFitKaiserLikelihood = CosmoSISLikelihoodFactory(TestShapeFitKaiserLikelihood, __name__)

setup, execute, cleanup = TestShapeFitKaiserLikelihood.build_module()

