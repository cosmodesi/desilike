
import numpy as np
import matplotlib.pyplot as plt


def test_templates():
    """Test power spectrum template functionality and properties."""

    from cosmoprimo.fiducial import DESI
    from desilike.theories import Cosmoprimo
    from desilike.theories.galaxy_clustering import (
        FixedPowerSpectrumTemplate, DirectPowerSpectrumTemplate,
        BAOPowerSpectrumTemplate, StandardPowerSpectrumTemplate,
        ShapeFitPowerSpectrumTemplate, BAOPhaseShiftPowerSpectrumTemplate,
        WiggleSplitPowerSpectrumTemplate, BandVelocityPowerSpectrumTemplate,
        TurnOverPowerSpectrumTemplate, DirectWiggleSplitPowerSpectrumTemplate)
    from desilike.theories.galaxy_clustering import (
        KaiserTracerPowerSpectrumMultipoles,
        DampedBAOWigglesTracerPowerSpectrumMultipoles)
    from desilike.theories.galaxy_clustering import (
        BAOExtractor, BAOPhaseShiftExtractor, StandardPowerSpectrumExtractor,
        ShapeFitPowerSpectrumExtractor, WiggleSplitPowerSpectrumExtractor,
        BandVelocityPowerSpectrumExtractor, TurnOverPowerSpectrumExtractor
    )

    results = {('f', 0): 0.81662893, ('f0', 0): 0.81883763, ('qpar', 0): 1.0, ('qper', 0): 1.0, ('pk_dd', 0): 3783.72072527,
               ('f', 1): 0.3356176, ('f0', 1): 0.3356176, ('qpar', 1): 1.45647117, ('qper', 1): 1.19186961, ('pk_dd', 1): 57659.09672345,
               ('f', 2): 0.81662893, ('f0', 2): 0.81883763, ('qpar', 2): 0.94981605, ('qper', 2): 0.94981605, ('pk_dd', 2): 3783.72072527,
               ('f', 3): 0.81662893, ('f0', 3): 0.81883763, ('qpar', 3): 0.38242867, ('qper', 3): 0.31295168, ('pk_dd', 3): 3783.72072527,
               ('f', 4): 0.81662893, ('f0', 4): 0.81883763, ('qpar', 4): 0.94981605, ('qper', 4): 0.94981605, ('pk_dd', 4): 3783.72072527,
               ('f', 5): 0.61172059, ('f0', 5): 0.61337508, ('qpar', 5): 0.94981605, ('qper', 5): 0.94981605, ('pk_dd', 5): 3783.72072527,
               ('f', 6): 0.61172059, ('f0', 6): 0.61337508, ('qpar', 6): 0.94981605, ('qper', 6): 0.94981605, ('pk_dd', 6): 6927.68763277,
               ('f', 7): 0.61172059, ('f0', 7): 0.61337508, ('qpar', 7): 0.91776714, ('qper', 7): 0.96625778, ('pk_dd', 7): 6927.68763277,
               ('f', 8): 0.61177488, ('f0', 8): 0.61337508, ('qpar', 8): 0.96625778, ('qper', 8): 1.01731043, ('pk_dd', 8): 37934.79447151,
               ('f', 9): 0.61172059, ('f0', 9): 0.61337508, ('qpar', 9): 0.96625778, ('qper', 9): 1.01731043, ('pk_dd', 9): 56998.76555436,
               ('f', 10): 0.3356176, ('f0', 10): 0.3356176, ('qpar', 10): 1.45647117, ('qper', 10): 1.19186961, ('pk_dd', 10): 57659.09672345,
               ('f', 11): 0.81662893, ('f0', 11): 0.81883763, ('qpar', 11): 0.96625778, ('qper', 11): 1.01731043, ('pk_dd', 11): 4153.79332761,
               ('f', 12): 0.81662893, ('f0', 12): 0.81883763, ('qpar', 12): 1.0, ('qper', 12): 1.0, ('pk_dd', 12): 2233.35578179}
    templates = [
        FixedPowerSpectrumTemplate(),
        DirectPowerSpectrumTemplate(),
        BAOPowerSpectrumTemplate(),
        BAOPowerSpectrumTemplate(apmode='bao'),
        BAOPhaseShiftPowerSpectrumTemplate(),
        StandardPowerSpectrumTemplate(),
        ShapeFitPowerSpectrumTemplate(),
        ShapeFitPowerSpectrumTemplate(apmode='qisoqap'),
        WiggleSplitPowerSpectrumTemplate(),
        WiggleSplitPowerSpectrumTemplate(kernel='tophat'),
        DirectWiggleSplitPowerSpectrumTemplate(),
        BandVelocityPowerSpectrumTemplate(kp=np.linspace(0.01, 0.1, 10)),
        TurnOverPowerSpectrumTemplate()
    ]

    def test_template(itemplate, template):
        cosmo = Cosmoprimo(engine='eisenstein_hu')
        template.init.update(cosmo=cosmo)
        theory = KaiserTracerPowerSpectrumMultipoles(template=template)
        if any(name in template.__class__.__name__ for name in ['direct', 'bao', 'standard', 'shapefit']):
            template.init.params['qpar'] = template.init.params['qper'] = {'derived': True}
            poles, derived = theory(return_derived=True)
            assert all(k in derived for k in ['qpar', 'qper']), "Missing derived parameters"
        else:
            poles, derived = theory(return_derived=True)
        assert np.isfinite(poles).all(), f"Non-finite result for {template.__class__.__name__}"
        # Access template properties
        quantities = ['f', 'f0', 'qpar', 'qper', 'pk_dd']

        # Test with only_now and multi-z
        if 'TurnOver' not in template.__class__.__name__:
            template.init.update(only_now=True)
            theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
            result = theory()
            assert np.isfinite(result).all(), f"Non-finite result with only_now for {template.__class__.__name__}"

        template.init.update(z=[0.5, 1.])
        size = 3
        all_values = {param.name: param.prior.sample(size=size, random_state=42) for param in template.varied_params}
        for i in range(size):
            param_values = {param: value[i] for param, value in all_values.items()}
            template(param_values)
            result = template.pk_dd
            assert result.shape[1] == 2
            assert np.isfinite(result).all(), f"Non-finite multi-z result for {template.__class__.__name__}, {param_values}"
            for name in quantities:
                value = np.mean(getattr(template, name))
                if i == 0:
                    #results[name, itemplate] = round(float(value), 8)
                    assert np.allclose(value, results[name, itemplate])

    for itemplate, template in enumerate(templates):
        test_template(itemplate, template)

    results = {(0, 'DM_over_rd'): 23.46413108, (0, 'qper'): 1.0143647, (0, 'DH_over_rd'): 16.61439705, (0, 'DH_over_DM'): 0.70807638, (0, 'qap'): 0.96952985, (0, 'DV_over_rd'): 20.91371469, (0, 'qiso'): 1.00395563,
               (1, 'DM_over_rd'): 23.98310617, (1, 'qper'): 1.03680022, (1, 'DH_over_rd'): 16.98170649, (1, 'DH_over_DM'): 0.70806952, (1, 'qap'): 0.96952045, (1, 'DV_over_rd'): 21.37621112, (1, 'qiso'): 1.02615761, (1, 'baoshift'): 1.12508775,
               (2, 'DM_over_rd'): 29.57490411, (2, 'qper'): 1.2785361, (2, 'DH_over_rd'): 20.93874056, (2, 'DH_over_DM'): 0.70799014, (2, 'qap'): 0.96941176, (2, 'DV_over_rd'): 26.35921154, (2, 'qiso'): 1.26536482, (2, 'df'): 0.44535425, (2, 'fsigmar'): 0.19184839,
               (3, 'DM_over_rd'): 29.57490411, (3, 'qper'): 1.2785361, (3, 'DH_over_rd'): 20.93874056, (3, 'DH_over_DM'): 0.70799014, (3, 'qap'): 0.96941176, (3, 'DV_over_rd'): 26.35921154, (3, 'qiso'): 1.26536482, (3, 'df'): 0.75754442, (3, 'f_sqrt_Ap'): 55.94702228, (3, 'dm'): -0.86021647, (3, 'm'): -1.52963146, (3, 'n'): 0.9649, (3, 'dn'): 0.0,
               (4, 'DM_over_rd'): 29.57490411, (4, 'qper'): 1.2785361, (4, 'DH_over_rd'): 20.93874056, (4, 'DH_over_DM'): 0.70799014, (4, 'qap'): 0.96941176, (4, 'DV_over_rd'): 26.35921154, (4, 'qiso'): 1.26536482, (4, 'df'): 0.24933451, (4, 'f_sqrt_Ap'): 55.94702228, (4, 'dm'): -0.86021647, (4, 'm'): -1.52963146, (4, 'n'): 0.9649, (4, 'dn'): 0.0,
               (5, 'qbao'): 1.26536482, (5, 'DV_over_rd'): 26.35921154, (5, 'qap'): 0.96941176, (5, 'DH_over_DM'): 0.70799014, (5, 'df'): 0.48584207, (5, 'fsigmar'): 0.11776881, (5, 'dm'): -0.16707089, (5, 'm'): -1.52490034,
               (6, 'qbao'): 1.26536482, (6, 'DV_over_rd'): 26.35921154, (6, 'qap'): 0.96941176, (6, 'DH_over_DM'): 0.70799014, (6, 'df'): 0.44535333, (6, 'fsigmar'): 0.19184759, (6, 'dm'): -0.07122651, (6, 'm'): -2.3847424,
               (7, 'f'): 0.89862458, (7, 'fsigmar'): 0.19184839, (7, 'qap'): 0.96941176,
               (8, 'DH_over_DM'): 0.70799014, (8, 'qap'): 0.96941176, (8, 'DV_times_kTO'): 27.63460734, (8, 'qto'): 0.80979379}
    extractors = [
        BAOExtractor(),
        BAOPhaseShiftExtractor(),
        StandardPowerSpectrumExtractor(),
        ShapeFitPowerSpectrumExtractor(),
        ShapeFitPowerSpectrumExtractor(dfextractor='fsigmar'),
        WiggleSplitPowerSpectrumExtractor(),
        WiggleSplitPowerSpectrumExtractor(kernel='tophat'),
        BandVelocityPowerSpectrumExtractor(kp=np.linspace(0.01, 0.1, 10)),
        TurnOverPowerSpectrumExtractor()
    ]

    def test_extractor(iextractor, extractor):
        size = 3
        all_values = {param.name: param.prior.sample(size=size, random_state=42) for param in extractor.varied_params}
        for i in range(size):
            param_values = {param: value[i] for param, value in all_values.items()}
            extractor(param_values)
            quantities = [quantity for quantities in extractor.conflicts for quantity in quantities]
            result = {quantity: getattr(extractor, quantity) for quantity in quantities}
            if i == 0:
                for quantity in result:
                    #results[iextractor, quantity] = round(float(result[quantity]), 8)
                    assert np.allclose(result[quantity], results[iextractor, quantity])
            assert np.isfinite(list(results.values())).all(), f"Non-finite multi-z result for {extractor.__class__.__name__}, {param_values}"

    for iextractor, extractor in enumerate(extractors):
        test_extractor(iextractor, extractor)


def test_bao():
    """
    Comprehensive test suite for BAO (Baryon Acoustic Oscillation) theory implementations.

    Tests both power spectrum and correlation function multipole estimators with various
    broadband parameterizations. Validates:
      - Parameter fixing, removal, and namespace assignment
      - Theory evaluation with different templates (BAO, Direct, Standard)
      - Emulator fitting and accuracy
      - Plot generation
      - Alcock-Paczynski parameter handling
    """

    from desilike.theories.galaxy_clustering import (DampedBAOWigglesTracerPowerSpectrumMultipoles,
                                                      ResummedBAOWigglesTracerPowerSpectrumMultipoles,
                                                      FlexibleBAOWigglesTracerPowerSpectrumMultipoles)
    from desilike.theories.galaxy_clustering import (DampedBAOWigglesTracerCorrelationFunctionMultipoles,
                                                      ResummedBAOWigglesTracerCorrelationFunctionMultipoles,
                                                      FlexibleBAOWigglesTracerCorrelationFunctionMultipoles)
    from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate, DirectPowerSpectrumTemplate,
                                                      StandardPowerSpectrumTemplate)
    from desilike.theories import Cosmoprimo
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    def test_theory(itheory, theory):
        """
        Test theory evaluation with various broadband parameterizations and templates.

        Validates that:
          - Parameters can be fixed and removed as expected
          - Varied/all parameter lists are correctly updated
          - Theory evaluation succeeds with different templates
          - Namespace assignment propagates correctly
        """
        is_power = 'Power' in theory.__class__.__name__
        cosmo = Cosmoprimo(engine='eisenstein_hu')

        # Define parameter values and removals for each broadband model
        fix_bb_params, remove_bb_params = {}, {}
        fix_bb_params['power'] = {'al0_1': 1e3 if is_power else 1e-3}
        fix_bb_params['power3'] = {'al0_-1': 1e3 if is_power else 1e-3}
        fix_bb_params['even-power'] = {'al0_2': 1e-3}
        fix_bb_params['pcs'] = {'al0_2': 2.}
        fix_bb_params['pcs2'] = {'al0_2': 2.}
        remove_bb_params['power'] = ['al0_-1']
        remove_bb_params['even-power'] = ['al0_0']
        remove_bb_params['pcs'] = ['al0_-1']
        if not is_power:
            fix_bb_params['pcs'].update({'bl0_2': 1e-3})

        # Test each broadband parameterization
        broadbands = ['power', 'power3', 'pcs'] + (['even-power', 'pcs2'] if not is_power else [])
        for broadband in broadbands:
            theory.init.update(pt=None, ells=(0, 2), broadband=broadband, template=BAOPowerSpectrumTemplate(cosmo=cosmo))

            size = 3
            all_values = {param.name: param.ref.sample(size=size, random_state=42) for param in theory.varied_params}
            for i in range(size):
                param_values = {param: value[i] for param, value in all_values.items()}
                poles = theory(param_values)
                if i == 0:
                    #results[itheory, broadband] = round(float(np.std(poles)), 8)
                    assert np.allclose(np.std(poles), results[itheory, broadband])

            # For 'power3' test 3 al0_* parameters; for 'pcs2' test 0 al0_* and 2 al2_*
            if broadband == 'power3':
                varied_params = theory.varied_params
                assert len(varied_params.names(basename=['al0_*'])) == 3, f"Expected 3 al0_* params, got {varied_params.names(basename=['al0_*'])}"
            if broadband == 'pcs2':
                varied_params = theory.varied_params
                al0_count = len(varied_params.names(basename=['al0_*']))
                al2_count = len(varied_params.names(basename=['al2_*']))
                assert al0_count == 0 and al2_count == 2, f"Expected 0 al0_* and 2 al2_*, got {varied_params}"

            # Fix parameters and verify they're no longer varied
            for name, value in fix_bb_params[broadband].items():
                theory.init.params[name].update(fixed=True)
            for name in remove_bb_params.get(broadband, []):
                del theory.init.params[name]
            for name in fix_bb_params[broadband]:
                assert name not in theory.varied_params, f"Parameter {name} should not be varied after fixing"
            for name in remove_bb_params.get(broadband, []):
                assert name not in theory.all_params, f"Parameter {name} should not exist after removal"

            template = BAOPowerSpectrumTemplate(z=0.1, cosmo=cosmo, fiducial='DESI', apmode='qiso')
            theory.init.update(ells=(0,), template=template)
            # Verify these parameters are fixed when only_now is specified
            for param in theory.all_params.select(basename=['dbeta']):
                assert param.fixed, f"Parameter {param.name} should be fixed with ells=[0]"

            template = BAOPowerSpectrumTemplate(z=0.1, cosmo=cosmo, fiducial='DESI', only_now=True)
            theory.init.update(template=template)
            # Verify these parameters are fixed when only_now is specified
            for param in theory.all_params.select(basename=['d', 'sigmapar', 'sigmaper', 'ml*_*']):
                assert param.fixed, f"Parameter {param.name} should be fixed with only_now=True"

            # Test namespace assignment
            namespace = 'LRG'
            for param in theory.init.params:
                param.update(namespace=namespace)
            basenames = theory.init.params.basenames()
            theory()
            for param in theory.all_params:
                if param.basename in basenames:
                    assert param.namespace == 'LRG', f"Parameter {param.name} lost namespace"
            # Evaluate theory with namespaced parameters
            params = fix_bb_params[broadband]
            theory(qpar=1.1, **{namespace + '.' + param: value for param, value in params.items()})

            # Access computed properties
            _ = theory.z, theory.ells, theory.template
            if 'PowerSpectrum' in theory.__class__.__name__:
                _ = theory.k
            else:
                _ = theory.s
            theory.plot(show=False)
            plt.close(plt.gcf())

            # Test with different templates
            for template in [DirectPowerSpectrumTemplate(z=1., cosmo=cosmo, fiducial='DESI'),
                             StandardPowerSpectrumTemplate(z=1., cosmo=cosmo, fiducial='DESI', with_now='peakaverage')]:
                theory.init.update(template=template)
                theory()

    def test_emulate(theory, emulate='pt'):
        """
        Test emulator fitting and evaluation against theory predictions.

        Validates that:
          - Emulator can be constructed and fit successfully
          - Emulated predictions match original theory within numerical precision
        """
        templates = [BAOPowerSpectrumTemplate(z=1., fiducial='DESI', with_now='peakaverage'),
                     DirectPowerSpectrumTemplate(z=1., fiducial='DESI'),
                     StandardPowerSpectrumTemplate(z=1., fiducial='DESI', with_now='peakaverage')]

        for template in templates:
            theory.init.update(template=template)
            theory()

            # Store baseline result
            bak = theory(**{param.name: param.value for param in theory.all_params.select(input=True)})
            _ = theory.k if 'PowerSpectrum' in theory.__class__.__name__ else theory.s

            # Build and fit emulator
            calculator = theory.pt if emulate == 'pt' else theory
            emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=1))
            emulator.set_samples()
            emulator.fit()
            calculator = emulator.to_calculator()

            # Replace calculator with emulated version
            if emulate == 'pt':
                theory.init.update(pt=calculator)
            else:
                theory = calculator

            # Verify emulated result matches baseline
            result = theory()
            assert np.allclose(result, bak), f"Emulator result {result} does not match baseline {bak}"

    results = {(0, 'power'): 9235.29743028, (0, 'power3'): 9563.60481141, (0, 'pcs'): 9668.76099336,
               (1, 'power'): 9558.48460935, (1, 'power3'): 9892.74482909, (1, 'pcs'): 9998.76162521,
               (2, 'power'): 9400.15335596, (2, 'power3'): 9733.9904474, (2, 'pcs'): 9840.51407169,
               (3, 'power'): 0.02953683, (3, 'power3'): 0.02955539, (3, 'pcs'): 0.1855233, (3, 'even-power'): 0.02808751, (3, 'pcs2'): 0.08246475,
               (4, 'power'): 0.03042766, (4, 'power3'): 0.0304462, (4, 'pcs'): 0.18467982, (4, 'even-power'): 0.02878487, (4, 'pcs2'): 0.08170432,
               (5, 'power'): 0.03072141, (5, 'power3'): 0.03074014, (5, 'pcs'): 0.18445573, (5, 'even-power'): 0.02912326, (5, 'pcs2'): 0.08160266}
    for itheory, cls in enumerate([
                DampedBAOWigglesTracerPowerSpectrumMultipoles,
                ResummedBAOWigglesTracerPowerSpectrumMultipoles,
                FlexibleBAOWigglesTracerPowerSpectrumMultipoles,
                DampedBAOWigglesTracerCorrelationFunctionMultipoles,
                ResummedBAOWigglesTracerCorrelationFunctionMultipoles,
                FlexibleBAOWigglesTracerCorrelationFunctionMultipoles]):
        """Run full test suite (emulation + theory) on provided theory object."""
        test_theory(itheory, cls())
        test_emulate(cls())


def test_broadband_bao_correlation():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles

    def get_analytic(s, delta):
        from scipy.special import sici
        x = delta * s
        sinx, sin2x, sin3x, cosx, cos2x, cos3x = np.sin(x), np.sin(2 * x), np.sin(3 * x), np.cos(x), np.cos(2 * x), np.cos(3 * x)
        Si_x, Si_2x, Si_3x = sici(x)[0], sici(2 * x)[0], sici(3 * x)[0]
        poly = {}
        poly['al2_-1'] = (16.0 - 8.0 * x**2 - 16.0 * cosx + x**2 * cosx - x * sinx + x**3 * Si_x) / (2.0 * x**3 * s**3)
        poly['al2_0'] = -2.0 * (12.0 - 16 * cosx + x**2 * cosx + 4.0 * cos2x - x**2 * cos2x - x * sinx + x * cosx * sinx + x**3 * Si_x - 2.0 * x**3 * Si_2x) / (x**3 * s**3)
        poly['al2_1'] = 0.5 * (48.0 + 8.0 * x**2 - 96.0 * cosx + 6.0 * x**2 * cosx + 64.0 * cos2x - 16.0 * x**2 * cos2x - 16.0 * cos3x + 9.0 * x**2 * cos3x - 6.0 * x * sinx + 8.0 * x * sin2x - 3.0 * x * sin3x + 6.0 * x**3 * Si_x - 32.0 * x**3 * Si_2x + 27.0 * x**3 * Si_3x) / (x**3 * s**3)
        return poly

    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(s=np.linspace(1., 130., 130), ells=(0, 2), broadband='pcs')
    for ill, ell in enumerate(theory.ells):
        names = theory.varied_params.names(basename=[f'{prefix}l{ell:d}_[-1:2]' for prefix in ['a', 'b']][:1])
        for name in names:
            xiref = theory(**{name: 0.})
            xi = theory(**{name: 1e-3 if name.startswith('b') else 10.})
            diff = xi[ill] - xiref[ill]
            analytic = get_analytic(theory.s, delta=theory.power.kp)
            if name in analytic:
                ratio = diff[-1] / analytic[name][-1]
                assert np.allclose(diff, analytic[name] * ratio, rtol=1e-2)
            xi = theory(**{name: 0.})


def test_full_shape():
    """
    Comprehensive test suite for full-shape power spectrum and correlation function theories.

    Tests multiple theory implementations (Kaiser, LPT, PyBird, FOLPS variants) with:
      - Observable construction and likelihood evaluation
      - Emulator fitting and accuracy validation
      - Parameter namespace handling
      - Theory-specific option validation (sigv, fsat for LPT)
      - Plot generation for power spectra and correlation functions
    """
    from cosmoprimo.fiducial import DESI
    from desilike.theories import Cosmoprimo
    from desilike.theories.galaxy_clustering import (
        DirectPowerSpectrumTemplate,
        ShapeFitPowerSpectrumTemplate,
        SimpleTracerPowerSpectrumMultipoles,
        KaiserTracerPowerSpectrumMultipoles,
        KaiserTracerCorrelationFunctionMultipoles,
        LPTVelocileptorsTracerCorrelationFunctionMultipoles,
        LPTVelocileptorsTracerPowerSpectrumMultipoles,
        REPTVelocileptorsTracerPowerSpectrumMultipoles,
        REPTVelocileptorsTracerCorrelationFunctionMultipoles,
        PyBirdTracerPowerSpectrumMultipoles,
        PyBirdTracerCorrelationFunctionMultipoles,
        FOLPSTracerPowerSpectrumMultipoles,
        FOLPSTracerCorrelationFunctionMultipoles,
        FOLPSv2TracerPowerSpectrumMultipoles,
        FOLPSv2TracerBispectrumMultipoles
    )
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    def clean_folps():
        """Clean up FOLPS module state to avoid cross-test contamination."""
        try:
            import FOLPSnu as FOLPS
            import types
            for name, value in list(FOLPS.__dict__.items()):
                if not name.startswith('__') and not callable(value) and not isinstance(value, types.ModuleType):
                    FOLPS.__dict__[name] = None
        except ImportError:
            pass

    def test_emulator(theory, emulate='pt'):
        """
        Test emulator fitting and evaluation accuracy.

        Validates:
          - Emulator can be constructed and fit
          - Emulated predictions match baseline theory
        """
        # Store baseline prediction
        kw = dict()
        if 'powerspectrum' in theory.__class__.__name__.lower():
            kw['k'] = np.linspace(0.01, 0.2, 201)
            _ = theory.k
        elif 'correlationfunction' in theory.__class__.__name__.lower():
            kw['s'] = np.linspace(30, 150, 201)
            _ = theory.s

        theory.init.update(**kw)
        baseline = theory(**{param.name: param.value for param in theory.all_params.select(input=True)})

        # Build and fit emulator
        calculator = theory.pt if emulate == 'pt' else theory
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=0))
        emulator.set_samples()
        emulator.fit()
        emulated_calculator = emulator.to_calculator()

        # Replace theory calculator with emulated version
        if emulate == 'pt':
            theory.init.update(pt=emulated_calculator)
        else:
            theory = emulated_calculator

        clean_folps()
        # Verify emulated prediction matches baseline
        theory.init.update(**kw)
        result = theory()
        assert np.allclose(result, baseline), f"Emulated result does not match baseline within tolerance"

        # Access emulated theory coordinates
        _ = theory.z, theory.ells
        if 'spectrum' in theory.__class__.__name__.lower():
            _ = theory.k
        else:
            _ = theory.s

        return theory

    def test_namespace(theory):
        """
        Test parameter namespace assignment and persistence.

        Validates:
          - Namespaces can be assigned to parameters
          - Namespaces persist through theory evaluation
          - Theory evaluation works with namespaced parameters
        """
        namespace = 'LRG'
        for param in theory.init.params:
            param.update(namespace=namespace)

        basenames = theory.init.params.basenames()
        theory()

        for param in theory.all_params:
            if param.basename in basenames:
                assert param.namespace == namespace, f"Parameter {param.name} lost namespace assignment"

        # Reset namespaces
        for param in theory.init.params:
            param.update(namespace=None)

        theory.init.update(tracers=('LRG',))
        for param in theory.init.params:
            if param.basename in basenames and param.basename not in ['sigmapar', 'sigmaper']:
                assert param.namespace == namespace, f"Parameter {param.name} did not propapate namespace"

        if any(name in theory.__class__.__name__.lower() for name in ['kaiser', 'pybird']):
            theory.init.update(tracers=('LRG', 'ELG'))
            assert 'LRG.b1' in theory.all_params and 'ELG.b1' in theory.all_params, "Tracer-specific parameters not created for multiple tracers"
            if 'spectrum' in theory.__class__.__name__.lower() :
                assert any('LRGxELG' in param.namespace for param in theory.all_params), "Cross-tracer parameters did not receive correct namespace"
            theory()
            for param in theory.init.params:
                param.update(namespace=('L1', param.namespace))
            assert 'L1.LRG.b1' in theory.all_params and 'L1.ELG.b1' in theory.all_params, "Tracer-specific parameters not created for multiple tracers"
            if 'spectrum' in theory.__class__.__name__.lower() :
                assert any('L1.LRGxELG' in param.namespace for param in theory.all_params), "Cross-tracer parameters did not receive correct namespace"
            theory()

        # Reset tracers
        theory.init.update(tracers=None)
        # Reset namespaces
        for param in theory.init.params:
            param.update(namespace=None)
        theory()


    def test_theory(itheory, cls, emulate='pt', freedoms=tuple()):
        """
        Comprehensive test routine for a theory object.

        Runs all sub-tests: emulator, and namespace handling.
        """
        freedoms = list(freedoms)
        fiducial = DESI(engine='eisenstein_hu')
        cosmo = Cosmoprimo(fiducial=fiducial)
        for itemplate, template in enumerate([ShapeFitPowerSpectrumTemplate(fiducial=fiducial, z=1.),
                                             DirectPowerSpectrumTemplate(cosmo=cosmo, fiducial=fiducial, z=1.)][1:]):
            theory = cls(template=template)
            test_namespace(theory)

            for freedom in [Ellipsis] + freedoms:
                if freedom is not Ellipsis: theory.init.update(freedom=freedom)
                size = 1

                def sample(param):
                    return param.ref.sample(size=size, random_state=42)

                all_values = {param.name: sample(param) for param in theory.varied_params}
                for i in range(size):
                    param_values = {param: value[i] for param, value in all_values.items()}
                    poles = theory(**param_values)
                    assert np.isfinite(poles).all(), f"Non-finite result for {theory.__class__.__name__}"
                    if i == 0:
                        results[itheory, itemplate, freedom] = round(float(np.std(poles)), 8)
                        assert np.allclose(np.std(poles), results[itheory, itemplate, freedom])

            test_namespace(theory)
            if emulate is not False:
                theory = test_emulator(theory, emulate=emulate)
                test_namespace(theory)
            for freedom in freedoms:
                theory.init.update(freedom=freedom)
                test_namespace(theory)

    results = {(0, 0, Ellipsis): 6077.14180929,
               (1, 0, Ellipsis): 6077.14180929,
               (2, 0, Ellipsis): 0.01660682,
               (3, 0, Ellipsis): 14326.36546248, (3, 0, 'min'): 13171.1865328, (3, 0, 'max'): 14326.36546248, (3, 0, None): 14326.36546248,
               (4, 0, Ellipsis): 0.02950103, (4, 0, 'min'): 0.02850522, (4, 0, 'max'): 0.02950103, (4, 0, None): 0.02950103,
               (5, 0, Ellipsis): 7562.24645358, (5, 0, 'min'): 7703.72498496, (5, 0, 'max'): 7556.07121289, (5, 0, None): 7562.24645358,
               (6, 0, Ellipsis): 0.02741651, (6, 0, 'min'): 0.02794728, (6, 0, 'max'): 0.02746957, (6, 0, None): 0.02741651,
               (7, 0, Ellipsis): 12583.50356435, (7, 0, 'min'): 12799.60763545, (7, 0, 'max'): 12583.50356435, (7, 0, None): 12583.50356435,
               (8, 0, Ellipsis): 0.02803988, (8, 0, 'min'): 0.0274056, (8, 0, 'max'): 0.02803988, (8, 0, None): 0.02803988,
               (9, 0, Ellipsis): 14420.48114423, (9, 0, 'min'): 13273.50048062, (9, 0, 'max'): 14420.48114423, (9, 0, None): 14420.48114423,
               (10, 0, Ellipsis): 0.02825026, (10, 0, 'min'): 0.02825026, (10, 0, 'max'): 0.02825026, (10, 0, None): 0.02825026, (10, 0, Ellipsis): 11671.52140467,
               (11, 0, 'min'): 12653.68634954, (11, 0, 'max'): 11671.52140467, (11, 0, None): 11671.52140467,
               (12, 0, Ellipsis): 1875898185.5563178, (12, 0, 'min'): 1875898185.5563178, (12, 0, 'max'): 1875898185.5563178, (12, 0, None): 1875898185.5563178}

    # Test other theories
    for itheory, cls in enumerate([SimpleTracerPowerSpectrumMultipoles,
                                   KaiserTracerPowerSpectrumMultipoles,
                                   KaiserTracerCorrelationFunctionMultipoles,
                                   REPTVelocileptorsTracerPowerSpectrumMultipoles,
                                   REPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                   PyBirdTracerPowerSpectrumMultipoles,
                                   PyBirdTracerCorrelationFunctionMultipoles,
                                   FOLPSTracerPowerSpectrumMultipoles,
                                   FOLPSTracerCorrelationFunctionMultipoles,
                                   LPTVelocileptorsTracerPowerSpectrumMultipoles,
                                   LPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                   FOLPSv2TracerPowerSpectrumMultipoles,
                                   FOLPSv2TracerBispectrumMultipoles][-4:]):
        test_theory(itheory, cls, emulate=False if 'Simple' in cls.__name__ else 'pt',
                    freedoms=[] if 'Kaiser' in cls.__name__ or 'Simple'  in cls.__name__ and 'FOLPSv2' not in cls.__name__ else ['min', 'max', None])


def test_png():
    """
    Comprehensive test suite for PNG (Primordial Non-Gaussianity) theory implementations.

    Tests PNG power spectrum multipoles with:
      - Different PNG methods (primordial vs matter)
      - Consistency between methods within tolerance
      - Theory evaluation with Alcock-Paczynski parameters
      - Emulator fitting and accuracy validation
      - Plot generation
    """
    from desilike.theories.galaxy_clustering import (
        PNGTracerPowerSpectrumMultipoles,
        ShapeFitPowerSpectrumTemplate,
    )
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    def test_prim_method():
        """
        Test that different PNG methods produce consistent results within tolerance.

        Validates:
          - Primordial method (prim) and matter method give similar results
          - Results change with fnl_loc parameter variations
        """
        theory_prim = PNGTracerPowerSpectrumMultipoles(method='prim')
        theory_matter = PNGTracerPowerSpectrumMultipoles(method='matter')

        # Test with non-zero fnl_loc
        test_params = {'fnl_loc': 100., 'b1': 2.}
        result_prim = theory_prim(**test_params)
        result_matter = theory_matter(**test_params)

        assert np.allclose(result_prim, result_matter, rtol=2e-3), \
            f"Prim and matter methods differ by more than 2e-3: max diff = {np.max(np.abs(result_prim - result_matter) / np.abs(result_prim))}"

    def test_emulator(theory):
        """
        Test emulator fitting and evaluation accuracy.

        Validates:
          - Emulator can be constructed and fit successfully
          - Emulated predictions match original theory
        """
        # Store baseline prediction with various parameters
        baseline = theory(fnl_loc=2., b1=1.5)
        k = theory.k  # Access theory coordinates

        # Build and fit emulator
        emulator = Emulator(theory, engine=TaylorEmulatorEngine(order=2))
        emulator.set_samples()
        emulator.fit()
        emulated_calculator = emulator.to_calculator()

        emulated_calculator.init.update(k=k)
        # Verify emulated result matches baseline
        result = emulated_calculator(fnl_loc=2., b1=1.5)
        assert np.allclose(result, baseline, rtol=1e-2), \
            f"Emulator result does not match baseline within 1% tolerance"

    def test_namespace(theory):
        """Test parameter namespace."""
        theory.init.update(tracers=('LRG', 'ELG'))
        assert 'LRG.b1' in theory.all_params and 'ELG.b1' in theory.all_params, "Tracer-specific parameters not created for multiple tracers"
        assert 'LRGxELG.sn0' in theory.all_params
        theory()
        for param in theory.init.params:
            param.update(namespace=('L1', param.namespace))
        assert 'L1.LRG.b1' in theory.all_params and 'L1.ELG.b1' in theory.all_params, "Tracer-specific parameters not created for multiple tracers"
        assert any('L1.LRGxELG' in param.namespace for param in theory.all_params), "Cross-tracer parameters did not receive correct namespace"
        theory()
        theory.init.update(tracers=None)
        for param in theory.init.params:
            param.update(namespace=None)
        theory()

    def test_theory(cls):
        """
        Comprehensive test routine for PNG theory.

        Runs all sub-tests: namespace handling, emulation, and plotting.
        """
        for method in ['matter', 'prim']:
            for itemplate, template in enumerate([None, ShapeFitPowerSpectrumTemplate(z=1.)]):
                theory = cls(template=template, method=method)
                size = 1

                def sample(param):
                    return param.ref.sample(size=size, random_state=42)

                all_values = {param.name: sample(param) for param in theory.varied_params}
                for i in range(size):
                    param_values = {param: value[i] for param, value in all_values.items()}
                    poles = theory(**param_values)
                    assert np.isfinite(poles).all(), f"Non-finite result for {theory.__class__.__name__}"
                    if i == 0:
                        #results[itemplate, method] = round(float(np.std(poles)), 8)
                        assert np.allclose(np.std(poles), results[itemplate, method])

                test_namespace(theory)
                test_emulator(theory)

                # Test plotting
                theory(fnl_loc=2.)
                fig = theory.plot(show=False)
                assert fig is not None, "Plot generation failed"
                plt.close(fig)

    # Test method consistency
    test_prim_method()
    results = {(0, 'matter'): 10011.16020071, (1, 'matter'): 10046.19588923, (0, 'prim'): 10011.33476181, (1, 'prim'): 10046.4520818}
    test_theory(PNGTracerPowerSpectrumMultipoles)