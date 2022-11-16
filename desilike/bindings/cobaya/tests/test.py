from desilike import setup_logging

from desilike.bindings.cobaya.factory import CobayaLikelihoodFactory, CobayaLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood


def test_external_likelihood():

    for cls in [TestSimpleLikelihood, TestShapeFitKaiserLikelihood]:
        CobayaLikelihood = CobayaLikelihoodFactory(cls)
        like = CobayaLikelihood()
        like.initialize()
        info = {'likelihood': {'test': {'external': CobayaLikelihood}}, 'theory': {}, 'params': like._params}
        # info = {'likelihood': {'test': {'external': CobayaLikelihood}}}

        from cobaya.model import get_model
        model = get_model(info)
        print(model.loglike({param.name: param.value for param in like.like.runtime_info.pipeline.params.select(varied=True, derived=False)}))


def test_generate_likelihood():

    for cls in [TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood][1:2]:
        setup_logging('warning')
        CobayaLikelihoodGenerator()(cls)
        info = {}
        if 'Full' in cls.__name__:
            info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
            info['params'] = {'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                              'ombh2': 0.02242,
                              'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
        info['likelihood'] = {'desilike.bindings.cobaya.tests.' + cls.__name__: None}
        #from desilike.io import BaseConfig
        #info = dict(BaseConfig('config_TestFullKaiserLikelihood.yaml'))
        from cobaya.model import get_model
        model = get_model(info)
        like = cls()
        print(model.loglike({param.name: param.value for param in like.runtime_info.pipeline.params.select(varied=True, derived=False)}))


if __name__ == '__main__':

    #test_external_likelihood()
    test_generate_likelihood()
