from desilike import setup_logging

from desilike.bindings.cobaya.factory import CobayaLikelihoodFactory, CobayaLikelihoodGenerator
from desilike.bindings.tests import TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood, TestEmulatedFullKaiserLikelihood


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

    setup_logging('warning')
    allcls = [TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestFullKaiserLikelihood, TestEmulatedFullKaiserLikelihood]
    CobayaLikelihoodGenerator()(allcls)
    for cls in [TestSimpleLikelihood, TestShapeFitKaiserLikelihood]:
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
        print(model.loglike({param.name: param.value for param in like.varied_params}))

    cls = TestFullKaiserLikelihood
    info = {}
    info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'ombh2': 0.02242,
                      'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.cobaya.tests.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'b1': 2., 'sn0': 0., 'H0': 69., 'As': 2e-9, 'omch2': 0.12}))

    cls = TestEmulatedFullKaiserLikelihood
    info = {}
    info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'ombh2': 0.02242,
                      'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.cobaya.tests.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'b1': 2., 'sn0': 0., 'H0': 69., 'As': 2e-9, 'omch2': 0.12}))

    info = {}
    info['theory'] = {'classy': {'extra_args': {'non linear': 'hmcode', 'nonlinear_min_k_max': 20, 'N_ncdm': 1, 'N_ur': 2.0328}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'A_s': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'omega_b': 0.02242,
                      'omega_cdm': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.cobaya.tests.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'b1': 2., 'sn0': 0., 'H0': 69., 'A_s': 2e-9, 'omega_cdm': 0.12}))


if __name__ == '__main__':

    #test_external_likelihood()
    test_generate_likelihood()
