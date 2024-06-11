from desilike import setup_logging

from desilike.bindings.cobaya import CobayaLikelihoodFactory, CobayaLikelihoodGenerator
from desilike.bindings.tests.test_generator import TestSimpleLikelihood, TestShapeFitKaiserLikelihood, TestDirectKaiserLikelihood, TestEmulatedDirectKaiserLikelihood


def test_external_likelihood():

    for cls in [TestSimpleLikelihood, TestShapeFitKaiserLikelihood]:
        CobayaLikelihood = CobayaLikelihoodFactory(cls, {})
        like = CobayaLikelihood()
        like.initialize()
        from desilike.io import BaseConfig
        CobayaLikelihood.params = params = BaseConfig(cls.__name__ + '.yaml')['params']
        info = {'likelihood': {'test': {'external': CobayaLikelihood}}, 'theory': {}, 'params': params}
        # info = {'likelihood': {'test': {'external': CobayaLikelihood}}}

        from cobaya.model import get_model
        model = get_model(info)
        print(model.loglike({param.name: param.value for param in like.like.runtime_info.pipeline.params.select(varied=True, derived=False)}))


def test_generated_likelihood():

    setup_logging('warning')

    # Test caching
    cls = TestDirectKaiserLikelihood
    info = {}
    info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'ombh2': 0.02242,
                      'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.tests.cobaya_bindings.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'LRG.b1': 2., 'LRG.sn0': 0., 'H0': 69., 'As': 2e-9, 'omch2': 0.12}))
    print(model.loglike({'LRG.b1': 2., 'LRG.sn0': 0.2, 'H0': 69., 'As': 2e-9, 'omch2': 0.12}))
    print(model.loglike({'LRG.b1': 2., 'LRG.sn0': 0., 'H0': 69., 'As': 3e-9, 'omch2': 0.12}))

    for cls in [TestSimpleLikelihood, TestShapeFitKaiserLikelihood]:
        info = {}
        if 'Direct' in cls.__name__:
            info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
            info['params'] = {'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                              'ombh2': 0.02242,
                              'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
        info['likelihood'] = {'desilike.bindings.tests.cobaya_bindings.' + cls.__name__: None}
        #from desilike.io import BaseConfig
        #info = dict(BaseConfig('config_TestDirectKaiserLikelihood.yaml'))
        from cobaya.model import get_model
        model = get_model(info)
        like = cls()
        print(model.loglike({param.name: param.value for param in like.varied_params}))

    cls = TestDirectKaiserLikelihood
    info = {}
    info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'ombh2': 0.02242,
                      'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.tests.cobaya_bindings.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'LRG.b1': 2., 'LRG.sn0': 0., 'H0': 69., 'As': 2e-9, 'omch2': 0.12}))

    cls = TestEmulatedDirectKaiserLikelihood
    info = {}
    info['theory'] = {'camb': {'stop_at_error': True, 'extra_args': {'num_massive_neutrinos': 1, 'halofit_version': 'mead'}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'As': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'ombh2': 0.02242,
                      'omch2': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.tests.cobaya_bindings.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'ELG.b1': 2., 'ELG.sn0': 0., 'H0': 69., 'As': 2e-9, 'omch2': 0.12}))

    info = {}
    info['theory'] = {'classy': {'extra_args': {'non linear': 'hmcode', 'nonlinear_min_k_max': 20, 'N_ncdm': 1, 'N_ur': 2.0328}}}
    info['params'] = {'H0': {'prior': {'min': 50., 'max': 100.}, 'ref': 70., 'proposal': 1.},
                      'A_s': {'prior': {'min': 5.0e-10, 'max': 8.0e-09}, 'ref': 2.0e-09, 'proposal': 1.0e-10},
                      'omega_b': 0.02242,
                      'omega_cdm': {'prior': {'min': 0.05, 'max': 0.2}, 'ref': 0.11933, 'proposal': 0.01}}
    info['likelihood'] = {'desilike.bindings.tests.cobaya_bindings.' + cls.__name__: None}
    from cobaya.model import get_model
    model = get_model(info)
    print(model.loglike({'ELG.b1': 2., 'ELG.sn0': 0., 'H0': 69., 'A_s': 2e-9, 'omega_cdm': 0.12}))


def test_convert_params():

    from cosmoprimo.fiducial import DESI
    cosmo = DESI()

    params = {'Omega_cdm': {'prior': {'limits': [0.01, 0.9]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_cdm'], 'scale': 0.01}, 'latex': '\Omega_{cdm}'},
              'Omega_b': {'prior': {'limits': [0.001, 0.3]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_b'], 'scale': 0.001}, 'latex': '\Omega_{b}'},
              'H0': {'prior': {'limits': [20., 100]}, 'ref': {'dist': 'norm', 'loc': cosmo['H0'], 'scale': 0.01}, 'latex': 'H_{0}'},
              'Omega_k': {'prior': {'limits': [-0.8, 0.8]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_k'], 'scale': 0.0065}, 'latex': '\Omega_{k}'},
              'w0_fld': {'prior': {'limits': [-3., 1.]}, 'ref': {'dist': 'norm', 'loc': cosmo['w0_fld'], 'scale': 0.08}, 'latex': 'w_{0}'},
              'wa_fld': {'prior': {'limits': [-3., 2.]}, 'ref': {'dist': 'norm', 'loc': cosmo['wa_fld'], 'scale': 0.3}, 'latex': 'w_{a}'},
              'logA': {'prior': {'limits': [1.61, 3.91]}, 'ref': {'dist': 'norm', 'loc': cosmo['logA'], 'scale': 0.014}, 'delta': 0.01, 'latex': '\ln(10^{10} A_{s})'},
              'n_s': {'prior': {'limits': [0.8, 1.2]}, 'ref': {'dist': 'norm', 'loc': cosmo['n_s'], 'scale': 0.0042}, 'delta': 0.004, 'latex': 'n_{s}'},
              'tau_reio': {'prior': {'limits': [0.01, 0.8]}, 'ref': {'dist': 'norm', 'loc': cosmo['tau_reio'], 'scale': 0.008}, 'delta': 0.004, 'latex': r'\tau'},
              'm_ncdm': {'prior': {'limits': [0., 5.]}, 'ref': {'dist': 'norm', 'loc': cosmo['m_ncdm_tot'], 'scale': 0.012}, 'delta': 0.01, 'latex': 'm_{ncdm}'},
              'Omega_m': {'derived': True, 'latex': '\Omega_{m}'}}
    params['n_s'].update(fixed=True)

    from desilike import ParameterCollection
    params = ParameterCollection(params)
    from desilike.bindings.cobaya import desilike_to_cobaya_params
    camb_params = desilike_to_cobaya_params(params, engine='camb')
    #print(camb_params['omch2'], camb_params['Omega_m'], camb_params['ns'])
    print(camb_params)
    classy_params = desilike_to_cobaya_params(params, engine='classy')
    #print(classy_params['Omega_m'], classy_params['n_s'])
    print(classy_params)


if __name__ == '__main__':

    #test_external_likelihood()
    test_generated_likelihood()
    test_convert_params()
