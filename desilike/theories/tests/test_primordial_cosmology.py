"""Tests for primordial cosmology calculators."""

from pathlib import Path

import numpy as np
import jax
import pytest


class TestCosmoprimoCosmology:

    def test_derived_param(self):
        """A Parameter('Omega_m', derived=True) added to params is computed in __call__
        (matching a plain cosmoprimo clone with the same inputs), correctly reacts to a
        shift in omega_cdm, and stays correct under jax.jit."""
        from desilike.base import build, get_params
        from desilike.parameter import Parameter, VariableCollection
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo0 = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
        vc = get_params(cosmo0)
        vc.set(Parameter('Omega_m', value=0.0, derived=True))
        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI', params=vc)

        pipe = build(cosmo)
        defaults = {p.name: p._value for p in get_params(cosmo)}

        import cosmoprimo
        fiducial = cosmoprimo.fiducial.DESI(engine='eisenstein_hu')

        def expected_omega_m(params):
            kw = {name: value for name, value in params.items() if name != 'Omega_m'}
            return fiducial.clone(base='input', **kw)['Omega_m']

        # eager, at defaults
        _, deriveds = pipe(defaults, return_derived=True)
        assert np.isclose(float(deriveds['Omega_m']), expected_omega_m(defaults), rtol=1e-6)
        assert np.isclose(float(pipe.params['Omega_m'].value), expected_omega_m(defaults), rtol=1e-6)

        # eager, sensitivity to omega_cdm
        shifted = {**defaults, 'omega_cdm': defaults['omega_cdm'] * 1.1}
        _, deriveds_shifted = pipe(shifted, return_derived=True)
        assert not np.isclose(float(deriveds_shifted['Omega_m']), float(deriveds['Omega_m']))
        assert np.isclose(float(deriveds_shifted['Omega_m']), expected_omega_m(shifted), rtol=1e-6)

        # jit: wrap in a lambda so that return_derived=True is a Python constant.
        pipe_rd = lambda p: pipe(p, return_derived=True)
        _, deriveds_jit = jax.jit(pipe_rd)(shifted)
        assert np.isclose(float(deriveds_jit['Omega_m']), expected_omega_m(shifted), rtol=1e-6)

    def test_of_string_matches_add_requirements_tuple(self):
        """get() with a bare-string 'of' must hit the same spec_key as add_requirements()
        registered (regression test: add_requirements() normalizes 'of' to a 2-tuple but
        get() previously did not, so this used to raise KeyError)."""
        from desilike.base import build
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        k = np.linspace(0.01, 0.2, 10)
        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI',
                                    requirements={'fourier.pk': [{'of': 'delta_cb', 'z': 1., 'k': k}]})
        build(cosmo)()
        result = cosmo.get('fourier.pk', of='delta_cb', z=1., k=k)
        assert result is not None

    def test_section_proxy(self):
        """cosmo.get_fourier().pk(...)/get_background().efunc(...)/etc. match the equivalent
        flat cosmo.get(...) calls exactly."""
        from desilike.base import build
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        k = np.linspace(0.01, 0.2, 10)
        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI', requirements={
            'fourier.pk': [{'of': 'delta_cb', 'z': 1., 'k': k}],
            'fourier.sigma8_z': [{'of': 'delta_cb', 'z': 1.}],
            'background.efunc': [{'z': 1.}],
            'background.comoving_transverse_distance': [{'z': 1.}],
            'params.N_eff': None,
            'thermodynamics.rs_drag': None,
        })
        build(cosmo)()

        np.testing.assert_allclose(cosmo.get_fourier().pk(of='delta_cb', z=1., k=k),
                                    cosmo.get('fourier.pk', of='delta_cb', z=1., k=k))
        np.testing.assert_allclose(cosmo.get_fourier().sigma8_z(of='delta_cb', z=1.),
                                    cosmo.get('fourier.sigma8_z', of='delta_cb', z=1.))
        np.testing.assert_allclose(cosmo.get_background().efunc(z=1.),
                                    cosmo.get('background.efunc', z=1.))
        np.testing.assert_allclose(cosmo.get_background().comoving_transverse_distance(z=1.),
                                    cosmo.get('background.comoving_transverse_distance', z=1.))
        np.testing.assert_allclose(cosmo['N_eff'],
                                    cosmo.get('params.N_eff'))
        np.testing.assert_allclose(cosmo.get_thermodynamics().rs_drag,
                                    cosmo.get('thermodynamics.rs_drag'))

    def test_external_engine_invalid_input_raises_eager_nans_under_jit(self):
        """External engines (camb, class) run through pure_callback with concrete values, so
        cosmoprimo's usual 'raise outside jax tracing, NaN inside' fallback (exception_or_nan)
        can never see a real Tracer there and always raises, even under jax.jit. __call__
        mirrors that same eager-raise / traced-NaN contract explicitly (via
        node_state['is_tracing'], threaded in by base.py's _run_graph/_run_or_cache): an
        unphysical point (e.g. omega_cdm < 0) still raises in plain eager use (matching the
        JAX-native engine's behavior, see test below -- though here the underlying
        CosmologyInputError comes back wrapped by pure_callback, e.g. as
        jax.errors.JaxRuntimeError or ValueError depending on the JAX/backend version; assert
        broadly on Exception rather than pin an exact wrapper type), but under jax.jit falls
        back to the (valid) fiducial cosmology for shapes and NaNs every result instead of
        crashing.

        Reads results via return_derived=True rather than cosmo.get(...) after the call:
        base.py's _run_graph resets a traced node's __dict__ back to its pre-call snapshot
        once the trace finishes (to avoid leaking JAX Tracers into later eager calls), and
        jax.pure_callback only actually *invokes* its Python callback at program execution
        time -- which happens after that reset. So a post-call attribute read would observe
        stale (pre-call) state, not the fresh computation; only the value threaded back
        through the compiled pipeline's own return path is reliable."""
        from desilike.base import build, get_params
        from desilike.parameter import Parameter
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo0 = CosmoprimoCosmology(engine='camb', fiducial='DESI')
        vc = get_params(cosmo0)
        vc.set(Parameter('Omega_m', value=0.0, derived=True))
        cosmo = CosmoprimoCosmology(engine='camb', fiducial='DESI', params=vc)
        pipe = build(cosmo)
        defaults = {p.name: float(p._value) for p in get_params(cosmo)}

        # Sanity: a valid point gives a finite derived Omega_m.
        _, deriveds = pipe(defaults, return_derived=True)
        assert np.isfinite(float(deriveds['Omega_m']))

        # Unphysical point, eager: raises (loud, useful for direct/debugging use). pure_callback
        # wraps the original CosmologyInputError, even outside jax.jit.
        bad_eager = {**defaults, 'omega_cdm': -0.05}
        with pytest.raises(Exception):
            pipe(bad_eager, return_derived=True)

        # Same shape of unphysical point (distinct value: the failed eager call above already
        # marked bad_eager as this node's "last params" before raising, since that bookkeeping
        # happens before node() runs -- reusing the same dict here would hit that stale cache
        # and skip re-execution instead of actually exercising the jit path), under jax.jit:
        # the full graph is always built regardless of any prior gate, so this must degrade to
        # NaN instead of crashing.
        bad_jit = {**defaults, 'omega_cdm': -0.06}
        pipe_rd = lambda p: pipe(p, return_derived=True)
        _, deriveds_jit = jax.jit(pipe_rd)(bad_jit)
        assert np.isnan(float(deriveds_jit['Omega_m']))

    def test_native_engine_invalid_input_raises_eager_nans_under_jit(self):
        """JAX-native engines (eisenstein_hu) need no special handling: tracers survive
        end-to-end (no pure_callback boundary), so cosmoprimo's own exception_or_nan already
        raises in eager and NaNs under jax.jit by itself. Regression guard that the
        base.py/_run_requirements refactor for external engines left this path unaffected."""
        from cosmoprimo import CosmologyInputError
        from desilike.base import build, get_params
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        k = np.linspace(0.01, 0.2, 5)
        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI',
                                    requirements={'fourier.pk': [{'of': 'delta_cb', 'z': 0.5, 'k': k}]})
        pipe = build(cosmo)
        defaults = {p.name: float(p._value) for p in get_params(cosmo)}
        bad = {**defaults, 'omega_cdm': -0.05}

        with pytest.raises(CosmologyInputError):
            pipe(bad, return_derived=True)

        jit_out = jax.jit(pipe)(bad)
        assert jit_out is None  # __call__ returns None; no crash is the point of this test


class TestACECosmology:

    import desilike as _desilike
    emulator_base_dir = Path(_desilike.__file__).parent.parent.parent / 'ace-emulators'

    def test_ace(self):
        from desilike.base import build, get_params
        from desilike.parameter import Parameter, VariableCollection
        from desilike.theories.primordial_cosmology import ACECosmology

        cosmo = ACECosmology(engine='isitgr', base_dir=self.emulator_base_dir, fiducial='DESI')
        params = get_params(cosmo)
        for name in ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']:
            params.set(Parameter(name, value=1.0, ref={'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                                  fixed=True, prior={'dist': 'uniform', 'limits': [-3., 3.]}))
        k = np.linspace(0.001, 0.1, 20)
        cosmo.update(params=params, requirements={
            'background.comoving_transverse_distance': [{'z': 0.1}],
            'fourier.pk': [{'of': 'delta_cb', 'z': 0.1, 'k': k}],
            'fourier.pk_now': [{'of': 'delta_cb', 'z': 0.1, 'k': k}],
            'fourier.sigma8_z': [{'of': 'delta_cb', 'z': 0.1}]})
        build(cosmo)()

    def test_section_proxy(self):
        """cosmo.get_fourier().pk(...)/get_background().comoving_transverse_distance(...) match
        the equivalent flat cosmo.get(...) calls exactly, for a second PrimordialCosmology subclass."""
        from desilike.base import build, get_params
        from desilike.parameter import Parameter
        from desilike.theories.primordial_cosmology import ACECosmology

        cosmo = ACECosmology(engine='isitgr', base_dir=self.emulator_base_dir, fiducial='DESI')
        params = get_params(cosmo)
        for name in ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']:
            params.set(Parameter(name, value=1.0, ref={'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                                  fixed=True, prior={'dist': 'uniform', 'limits': [-3., 3.]}))
        k = np.linspace(0.001, 0.1, 20)
        cosmo.update(params=params, requirements={
            'fourier.pk': [{'of': 'delta_cb', 'z': 0.1, 'k': k}],
            'background.comoving_transverse_distance': [{'z': 0.1}]})
        build(cosmo)()

        np.testing.assert_allclose(cosmo.get_fourier().pk(of='delta_cb', z=0.1, k=k),
                                    cosmo.get('fourier.pk', of='delta_cb', z=0.1, k=k))
        np.testing.assert_allclose(cosmo.get_background().comoving_transverse_distance(z=0.1),
                                    cosmo.get('background.comoving_transverse_distance', z=0.1))

    def test_packaged(self):
        """engine='ace' serves DirectSpectrum2Template's and the CMB likelihoods' requirements
        from the packaged jaxace / jaxmapse / jaxcapse trained emulators, matching cosmoprimo
        (class for pk / sigma8_z / rs_drag, camb for the Cl) at the DESI fiducial."""
        from desilike.base import build
        from desilike.theories.primordial_cosmology import ACECosmology

        ellmax = 2508
        z_test = 1.
        k = np.geomspace(1e-3, 1., 30)
        # engine='ace' includes the derived sigma8_m and rs_drag parameters by default.
        cosmo = ACECosmology(engine='ace', fiducial='DESI', requirements={
            'fourier.pk': [{'of': 'delta_cb', 'z': z_test, 'k': k}, {'of': 'theta_cb', 'z': z_test, 'k': k}],
            'fourier.sigma8_z': [{'of': 'delta_cb', 'z': z_test}, {'of': 'theta_cb', 'z': z_test}],
            'background.efunc': [{'z': z_test}],
            'background.comoving_transverse_distance': [{'z': z_test}],
            'harmonic.lensed_cl': [{'ellmax': ellmax}],
            'harmonic.lens_potential_cl': [{'ellmax': ellmax}],
        })
        build(cosmo)()

        import cosmoprimo
        fiducial = cosmoprimo.fiducial.DESI(engine='class')
        fo = fiducial.get_fourier()

        # derived params (default with engine='ace'): rs_drag in Mpc/h, sigma8_m at z = 0
        rs_drag = float(cosmo.derived_params['rs_drag'].value)
        assert np.isclose(rs_drag, fiducial.get_thermodynamics().rs_drag, rtol=2e-4)
        sigma8_m = float(cosmo.derived_params['sigma8_m'].value)
        assert np.isclose(sigma8_m, fo.sigma8_z(0., of='delta_m'), rtol=1e-3)

        # fourier: sigma8_z is the top-hat sigma8 of the emulated P_cb, so genuinely cb (the
        # network's own sigma8 output is total-matter, 0.4% away at this fiducial); fsigma8 =
        # f_z * sigma8_z.  These tolerances are the ones that catch a growth-normalisation
        # mistake: the emulated pk carries D^2 with whatever normalisation D was given, and the
        # conventions on offer differ by 3.3% (see _ace_background).  Measured: 1.3e-4 / 1.2e-4.
        sigma8 = cosmo.get_fourier().sigma8_z(of='delta_cb', z=z_test)
        fsigma8 = cosmo.get_fourier().sigma8_z(of='theta_cb', z=z_test)
        assert np.isclose(float(sigma8), fo.sigma8_z(z_test, of='delta_cb'), rtol=1e-3)
        assert np.isclose(float(fsigma8), fo.sigma8_z(z_test, of='theta_cb'), rtol=1e-3)

        # fourier: linear pk (delta_cb), and pk_tt = f_z^2 pk_dd with f_z = fsigma8 / sigma8
        pk_dd = np.asarray(cosmo.get_fourier().pk(of='delta_cb', z=z_test, k=k))
        pk_tt = np.asarray(cosmo.get_fourier().pk(of='theta_cb', z=z_test, k=k))
        np.testing.assert_allclose(pk_dd, fo.pk_interpolator(of='delta_cb')(k, z=z_test), rtol=2e-3)
        np.testing.assert_allclose(pk_tt / pk_dd, float(fsigma8 / sigma8)**2, rtol=1e-6)

        # background (analytic jaxace, unchanged by this feature; sanity only)
        assert np.isclose(float(cosmo.get_background().efunc(z=z_test)), fiducial.efunc(z_test), rtol=1e-3)
        assert np.isclose(float(cosmo.get_background().comoving_transverse_distance(z=z_test)),
                          fiducial.comoving_transverse_distance(z_test), rtol=1e-3)

        # harmonic: raw dimensionless Cl, matching CosmoprimoCosmology's convention (camb)
        cosmo_camb = fiducial.clone(engine='camb', lensing=True, ellmax_cl=ellmax + 500, non_linear='mead')
        cl_ref = cosmo_camb.get_harmonic().lensed_cl(ellmax=ellmax)
        clpp_ref = cosmo_camb.get_harmonic().lens_potential_cl(ellmax=ellmax)
        cl = cosmo.get_harmonic().lensed_cl(ellmax=ellmax)
        clpp = cosmo.get_harmonic().lens_potential_cl(ellmax=ellmax)
        ells = np.arange(ellmax + 1)
        for name in ['tt', 'ee']:
            np.testing.assert_allclose(np.asarray(cl[name])[30:], cl_ref[name][30:], rtol=6e-3)
        # te crosses zero: compare at the Dl level with an absolute tolerance
        scale = ells * (ells + 1) * (fiducial['T_cmb'] * 1e6)**2 / (2 * np.pi)
        np.testing.assert_allclose(np.asarray(cl['te'])[30:] * scale[30:], cl_ref['te'][30:] * scale[30:],
                                   atol=2e-3 * np.max(np.abs(cl_ref['te'][30:] * scale[30:])))
        np.testing.assert_allclose(np.asarray(cl['bb']), 0.)
        np.testing.assert_allclose(np.asarray(clpp['pp'])[30:1000], clpp_ref['pp'][30:1000], rtol=2e-2)

        # requesting more than the emulator's training range must raise
        with pytest.raises(ValueError, match='ellmax'):
            cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': 6000}]})

    def test_capse_local_dir(self, tmp_path):
        """A Capse-style Cl emulator directory under base_dir (per-spectrum TT/TE/EE/PP network
        subdirs, free-text metadata) is auto-introspected and gives results identical to the
        packaged jaxcapse path, here using the very same cached camb_lcdm networks."""
        import shutil
        from desilike.base import build
        from desilike.theories.primordial_cosmology import ACECosmology

        cached_dir = Path.home() / '.jaxcapse_data' / 'emulators'
        if not (cached_dir / 'TT' / 'nn_setup.json').is_file():
            pytest.skip('cached camb_lcdm networks not available')
        local_dir = tmp_path / 'capse_local'
        for name in ['TT', 'TE', 'EE', 'PP']:
            shutil.copytree(cached_dir / name, local_dir / name, ignore=shutil.ignore_patterns('__pycache__'))

        ellmax = 500
        results = {}
        for label, engine, base_dir in [('packaged', 'ace', None),
                                        ('local', {'harmonic': 'capse_local', 'background': 'ACE_mnuw0wacdm_ln10As_basis'}, tmp_path)]:
            cosmo = ACECosmology(engine=engine, base_dir=base_dir, fiducial='DESI',
                                 requirements={'harmonic.lensed_cl': [{'ellmax': ellmax}],
                                               'harmonic.lens_potential_cl': [{'ellmax': ellmax}]})
            build(cosmo)()
            results[label] = (cosmo.get_harmonic().lensed_cl(ellmax=ellmax), cosmo.get_harmonic().lens_potential_cl(ellmax=ellmax), cosmo)

        for name in ['tt', 'ee', 'bb', 'te']:
            np.testing.assert_array_equal(np.asarray(results['local'][0][name]), np.asarray(results['packaged'][0][name]))
        np.testing.assert_array_equal(np.asarray(results['local'][1]['pp']), np.asarray(results['packaged'][1]['pp']))

        # introspected metadata: parsed inputs, training ranges (drive the NaN guard) and ellmax
        local_cosmo = results['local'][2]
        metadata = local_cosmo._emulator_metadata[str(local_dir)]
        assert metadata['inputs'] == ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio']
        assert metadata['ellmax'] == 5000
        assert np.isclose(local_cosmo._param_clip_ranges['tau_reio'][0], 0.02, atol=1e-3)
        with pytest.raises(ValueError, match='ellmax'):
            local_cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': 6000}]})

    def test_capse_w0wa_dir(self):
        """The local capse_mnuw0wacdm_250001 w0waCDM Cl emulator (Dl muK^2 / phiphi conventions
        assumed identical to camb_lcdm, verified against CAMB): matches a camb w0waCDM run at
        a shifted (w0, wa, mnu-in-range) point, and responds to w0."""
        from desilike.base import build, get_params
        from desilike.theories.primordial_cosmology import ACECosmology

        # The artifact sits next to the desilike checkout (on NERSC it lives under the
        # default base_dir, Installer().install_dir / 'ace-emulators').
        base_dir = self.emulator_base_dir.parent
        emulator_dir = base_dir / 'capse_mnuw0wacdm_250001'
        if not (emulator_dir / 'TT' / 'nn_setup.json').is_file():
            pytest.skip('capse_mnuw0wacdm_250001 not available')

        ellmax = 2500
        cosmo = ACECosmology(engine={'harmonic': 'capse_mnuw0wacdm_250001', 'background': 'ACE_mnuw0wacdm_ln10As_basis'},
                             base_dir=base_dir, fiducial='DESI',
                             requirements={'harmonic.lensed_cl': [{'ellmax': ellmax}]})
        pipe = build(cosmo)
        metadata = cosmo._emulator_metadata[str(emulator_dir)]
        assert metadata['inputs'] == ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio', 'm_ncdm', 'w0_fld', 'wa_fld']
        assert metadata['ellmax'] == 2999

        defaults = {param.name: param._value for param in get_params(cosmo)}
        point = {**defaults, 'w0_fld': -0.9, 'wa_fld': -0.3}
        pipe(point)
        cl_tt = np.asarray(cosmo.get_harmonic().lensed_cl(ellmax=ellmax)['tt'])

        import cosmoprimo.fiducial
        fiducial = cosmoprimo.fiducial.DESI(engine='camb')
        cosmo_camb = fiducial.clone(lensing=True, ellmax_cl=ellmax + 500, non_linear='mead', w0_fld=-0.9, wa_fld=-0.3)
        cl_ref = cosmo_camb.get_harmonic().lensed_cl(ellmax=ellmax)
        np.testing.assert_allclose(cl_tt[30:], cl_ref['tt'][30:], rtol=1e-2)

        # w0 sensitivity: shifting w0 changes the emulated Cl (atol=0: raw Cl are ~1e-10)
        pipe(defaults)
        cl_tt_fiducial = np.asarray(cosmo.get_harmonic().lensed_cl(ellmax=ellmax)['tt'])
        assert not np.allclose(cl_tt[2:], cl_tt_fiducial[2:], rtol=1e-4, atol=0.)

    def test_packaged_out_of_range(self):
        """Parameters outside the packaged emulators' training ranges yield NaN results
        (eager and jit) instead of a non-finite crash in downstream spline solves, and a
        warning flags priors wider than the training range at build time."""
        import warnings as _warnings
        import jax
        from desilike.base import build, get_params
        from desilike.theories.primordial_cosmology import ACECosmology

        k = np.geomspace(1e-3, 1., 20)
        z_test = 1.
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            cosmo = ACECosmology(engine='ace', fiducial='DESI', requirements={
                'fourier.pk': [{'of': 'delta_cb', 'z': z_test, 'k': k}],
                'fourier.pk_now': [{'of': 'delta_cb', 'engine': 'peakaverage', 'z': z_test, 'k': k}],
                'harmonic.lensed_cl': [{'ellmax': 100}],
            })
            pipe = build(cosmo)
        # h prior [0.1, 10] etc. extend beyond the training ranges: warned at build.
        assert any('training range' in str(warning.message) for warning in caught)

        defaults = {param.name: param._value for param in get_params(cosmo)}

        def run(params):
            pipe(params)
            return (cosmo.get('fourier.pk', of='delta_cb', z=z_test, k=k),
                    cosmo.get('fourier.pk_now', of='delta_cb', engine='peakaverage', z=z_test, k=k),
                    cosmo.get('harmonic.lensed_cl', ellmax=100)['tt'])

        results = run(defaults)
        assert all(np.all(np.isfinite(np.asarray(result))) for result in results)
        results = run({**defaults, 'h': 3.})  # far outside the ACE training range: NaN, no crash
        assert all(np.all(np.isnan(np.asarray(result))) for result in results)

        # jit path, through a downstream consumer (results must be read off a pipeline output,
        # not off cosmo._results, which lives inside the compiled pipe's own trace)
        from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template
        template = DirectSpectrum2Template(z=z_test, fiducial='DESI', cosmo=ACECosmology(engine='ace', fiducial='DESI'))
        pipe_template = jax.jit(build(template))
        defaults = {param.name: param._value for param in get_params(template)}
        assert np.all(np.isfinite(np.asarray(pipe_template(defaults))))
        assert np.all(np.isnan(np.asarray(pipe_template({**defaults, 'h': 3.}))))

    def test_training_ranges_accepts_a_cosmology(self):
        """A cosmology can be handed over directly, so no caller has to branch on how far it got.

        Before a build nothing is loaded, so the answer comes from the engine spec the instance
        was constructed with; after build it is what the instance actually enforces. The two
        agree here because the same engine is loaded either way -- what matters is that the
        caller does not have to know which case it is in.
        """
        from desilike.theories.primordial_cosmology import ACECosmology, CosmoprimoCosmology

        declared = ACECosmology.training_ranges(engine='ace')
        cosmology = ACECosmology(engine='ace', requirements={'fourier.sigma8_z': [{'of': 'delta_cb', 'z': 0.}]})
        # Uncompiled: falls back to the declared spec, and reads `base_dir`/`engine` off `_init`.
        assert ACECosmology.training_ranges(engine=cosmology) == declared

        # Compiled: the ranges the out-of-range guard enforces. The 'cosmo' basis drops the
        # networks' native `H0`, which the guard itself keeps alongside `h`.
        from desilike.base import build
        build(cosmology)()
        enforced = ACECosmology.training_ranges(engine=cosmology)
        assert enforced and 'H0' not in enforced
        assert 'H0' in cosmology._param_clip_ranges and 'h' in cosmology._param_clip_ranges

        # Any cosmology may be passed: one without packaged emulators declares no ranges.
        assert ACECosmology.training_ranges(engine=CosmoprimoCosmology()) == {}

    def test_truncate_priors(self):
        """truncate_priors intersects the priors with the packaged emulators' training ranges
        (H0 ranges applied to h, scaled by 1/100), leaves non-matching priors and distribution
        attrs untouched, and returns the collection for chaining."""
        from desilike import Parameter, VariableCollection
        from desilike.theories.primordial_cosmology import ACECosmology

        params = VariableCollection()
        params.set(Parameter('h', value=0.6736, prior=dict(limits=[0.1, 10.])))
        params.set(Parameter('omega_cdm', value=0.12, prior=dict(limits=[0.01, 0.99])))
        params.set(Parameter('n_s', value=0.9649, prior=dict(dist='norm', loc=0.9649, scale=0.042)))
        params.set(Parameter('m_ncdm', value=0.06, prior=dict(limits=[0., 0.3])))  # already narrower than the training range
        params.set(Parameter('Omega_m', value=0.31, prior=dict(limits=[0.01, 0.99])))  # not an emulator input

        # training_ranges: intersected across the packaged set; the 'cosmo' basis (default)
        # reports H0 as h (/100), the 'emulator' basis keeps the networks' native names.
        ranges = ACECosmology.training_ranges(engine='ace')
        assert ranges['h'] == (0.5, 0.9) and 'H0' not in ranges
        assert ranges['omega_cdm'] == (0.08, 0.18)
        ranges_emulator = ACECosmology.training_ranges(engine='ace', basis='emulator')
        assert ranges_emulator['H0'] == (50., 90.) and 'h' not in ranges_emulator
        assert ACECosmology.training_ranges(engine='does_not_exist') == {}
        with pytest.raises(ValueError, match='basis'):
            ACECosmology.training_ranges(engine='ace', basis='unknown')

        returned = ACECosmology.truncate_priors(params, engine='ace')
        assert returned is params
        # jaxace/jaxmapse training box: H0 in (50, 90) -> h in (0.5, 0.9); omega_cdm in (0.08, 0.18).
        assert params['h'].prior.limits == (0.5, 0.9)
        assert params['omega_cdm'].prior.limits == (0.08, 0.18)
        # Gaussian prior keeps its distribution and attrs, gains the training-range limits
        # (n_s in (0.8, 1.1) from jaxace, tightened by the camb_lcdm Cl emulator).
        assert params['n_s'].prior.dist == 'norm' and params['n_s'].prior.attrs['loc'] == 0.9649
        assert params['n_s'].prior.limits[0] >= 0.8 and params['n_s'].prior.limits[1] <= 1.1
        # Narrower existing limits and non-emulator-input parameters are untouched.
        assert params['m_ncdm'].prior.limits == (0., 0.3)
        assert params['Omega_m'].prior.limits == (0.01, 0.99)

        # Non-packaged engine names (no training ranges known) leave everything untouched.
        params = VariableCollection()
        params.set(Parameter('h', value=0.6736, prior=dict(limits=[0.1, 10.])))
        ACECosmology.truncate_priors(params, engine='does_not_exist')
        assert params['h'].prior.limits == (0.1, 10.)

    def test_packaged_direct_template(self):
        """DirectSpectrum2Template(cosmo=ACECosmology(engine='ace')) compiles and runs as pure
        JAX: qpar = qper = 1 and f consistent between fk, f0 and fsigma8 / sigma8 at the
        fiducial, with finite gradients with respect to cosmological parameters."""
        import jax
        from desilike.base import build, get_params
        from desilike.theories.primordial_cosmology import ACECosmology
        from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template

        cosmo = ACECosmology(engine='ace', fiducial='DESI')
        template = DirectSpectrum2Template(z=0.8, fiducial='DESI', cosmo=cosmo)
        pipe = build(template)
        defaults = {param.name: param._value for param in get_params(template)}
        pipe(defaults)
        assert np.isclose(float(template.qpar), 1., atol=5e-3)
        assert np.isclose(float(template.qper), 1., atol=5e-3)
        f = float(template.fsigma8 / template.sigma8)
        np.testing.assert_allclose(np.asarray(template.fk), f, rtol=1e-6)
        assert np.isclose(float(template.f0), f, rtol=1e-6)
        assert np.all(np.isfinite(np.asarray(template.pk_dd)))

        # differentiability: d(sum pk_dd)/d(logA) is finite and positive
        grad = jax.grad(lambda p: jax.numpy.sum(pipe(p)))(defaults)
        assert np.isfinite(float(grad['logA'])) and float(grad['logA']) > 0.

    def test_fiducial_from_calculator(self):
        """_get_fiducial(name, calculator=cosmo) re-runs cosmo's own pipeline at the named
        fiducial's parameter values (cosmoprimo-recognized ones only, e.g. h, omega_cdm, ...),
        keeps cosmo's own extra/nuisance params (mu1, Sigma1, ...) unchanged, and returns an
        ACECosmology reflecting the fiducial point (note: not necessarily cosmo itself, since
        the value travels back through the compiled graph's JAX pytree tree_flatten/unflatten
        machinery, which reconstructs a fresh-but-value-equal instance)."""
        import cosmoprimo
        from desilike.base import build, get_params
        from desilike.parameter import Parameter
        from desilike.theories.primordial_cosmology import ACECosmology, _get_fiducial

        cosmo = ACECosmology(engine='isitgr', base_dir=self.emulator_base_dir, fiducial='DESI')
        params = get_params(cosmo)
        for name in ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']:
            params.set(Parameter(name, value=1.0, ref={'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                                  fixed=True, prior={'dist': 'uniform', 'limits': [-3., 3.]}))
        cosmo.update(params=params, requirements={'background.comoving_transverse_distance': [{'z': 0.1}]})
        build(cosmo)()

        cosmo2 = _get_fiducial('DESI', calculator=cosmo)
        assert isinstance(cosmo2, ACECosmology)

        desi = cosmoprimo.fiducial.DESI()
        assert np.isclose(cosmo2._param_values['h'], desi['h'])
        assert np.isclose(cosmo2._param_values['mu1'], 1.0)  # extra param kept unchanged

        result = cosmo2.get_background().comoving_transverse_distance(z=0.1)
        np.testing.assert_allclose(result, cosmo.get('background.comoving_transverse_distance', z=0.1))


if __name__ == '__main__':

    test = TestACECosmology()
    test.test_ace()


# ── emulating a cosmology ─────────────────────────────────────────────────────
#
# What is exact is asserted as exact (the ratio emulated / exact does not move with the
# parameter, to machine precision), and what is a preconditioner is asserted against a
# measured number. The growth routing is checked against CLASS: cosmoprimo's eisenstein_hu
# engine grows with the Carroll-Press-Turner approximation, whose response to w0/wa is not the
# physical one, so it can validate the amplitude, tilt and dilation but not the growth.

K = np.geomspace(1e-3, 0.5, 300)
Z = 0.8
BOUNDS = {'h': (0.63, 0.72), 'omega_cdm': (0.11, 0.13), 'omega_b': (0.021, 0.024),
          'logA': (2.9, 3.2), 'n_s': (0.94, 0.99), 'w0_fld': (-1.2, -0.8), 'wa_fld': (-0.6, 0.4)}


def _fourier_cosmology(engine='eisenstein_hu', free=('w0_fld', 'wa_fld'), harmonic=False, derived=False,
                       background=False, age=False, extra=None):
    from desilike.parameter import Parameter
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology

    params = CosmoprimoCosmology.propose_params(fiducial='DESI')
    for param in params:
        if param.basename in free:
            param.update(fixed=False)
    if derived:
        params.set(Parameter('sigma8_cb', value=0., derived=True))
    requirements = {
        'fourier.pk': [{'of': 'delta_cb', 'z': Z, 'k': K}, {'of': 'theta_cb', 'z': Z, 'k': K}],
        'primordial.pk': [{'k': K}]}
    if background:
        requirements.update({'background.efunc': [{'z': Z}],
                             'background.comoving_transverse_distance': [{'z': Z}],
                             'thermodynamics.rs_drag': None})
    if age:
        requirements['background.age'] = None
    if harmonic:
        requirements['harmonic.unlensed_cl'] = [{'ellmax': 60}]
    requirements.update(extra or {})
    return CosmoprimoCosmology(engine=engine, fiducial='DESI', params=params, requirements=requirements)


def _space(*names):
    from desilike.emulators import Space

    return Space(bounds={name: BOUNDS[name] for name in names})


def _relative(emulator, point, leaf='fourier.pk|of=delta_cb,delta_cb', kmin=0., kmax=np.inf):
    predicted, exact = emulator.predict(**point)[leaf], emulator.compute(point)[leaf]
    ratio = np.asarray(predicted) / np.asarray(exact)
    return ratio[..., (K >= kmin) & (K <= kmax)] if 'pk' in leaf else ratio


class TestFourierEmulator:

    def test_leaves_are_named_by_requirement(self):
        """A leaf is named by what it is, and the name does not move when a requirement is
        appended -- which is what lets two calculators serving one requirement be stitched."""
        from desilike.emulators import Emulator
        from desilike.theories.primordial_cosmology import FourierEmulator

        cosmo = _fourier_cosmology()
        assert cosmo.get_emulator_cls() is FourierEmulator
        emulator = Emulator(cosmo, _space('omega_cdm', 'omega_b'))
        names = emulator.children_leafnames
        assert 'fourier.pk|of=delta_cb,delta_cb' in names and 'fourier.pk|of=theta_cb,theta_cb' in names
        assert 'primordial.pk' in names and 'input.h' in names and 'input.logA' in names
        more = _fourier_cosmology(extra={'fourier.sigma8_z': [{'of': 'delta_cb', 'z': Z}]})
        wider = Emulator(more, _space('omega_cdm', 'omega_b')).children_leafnames
        assert set(names) < set(wider) and 'fourier.sigma8_z|of=delta_cb,delta_cb' in wider

    def test_what_the_leaves_allow_leaves_the_grid(self):
        """Every candidate is granted on spectra; a sigma8_z leaf keeps h and n_s, whose
        dependence it carries through a window and an integral nothing divides out; a derived
        sigma8 is a sigma8_z leaf like any other."""
        from desilike.emulators import Emulator

        emulator = Emulator(_fourier_cosmology(), _space(*BOUNDS))
        assert emulator.params == ['omega_cdm', 'omega_b']
        assert set(emulator.exact_params) == {'h', 'logA', 'n_s', 'w0_fld', 'wa_fld'}
        cosmo = _fourier_cosmology(extra={'fourier.sigma8_z': [{'of': 'delta_cb', 'z': Z}]})
        emulator = Emulator(cosmo, _space(*BOUNDS))
        assert emulator.params == ['h', 'omega_cdm', 'omega_b', 'n_s']
        # a derived sigma8 is a sigma8_z leaf like any other: routed in the amplitude and the
        # growth, and keeping h and n_s on the grid for the same reason
        emulator = Emulator(_fourier_cosmology(derived=True), _space(*BOUNDS))
        assert emulator.params == ['h', 'omega_cdm', 'omega_b', 'n_s']
        assert emulator._leaf_info()['derived.sigma8_cb']['kind'] == 'sigma'
        emulator = Emulator(_fourier_cosmology(), _space(*BOUNDS), exact=('A_s', 'n_s'))
        assert emulator.params == ['h', 'omega_cdm', 'omega_b', 'w0_fld', 'wa_fld']

    def test_amplitude_and_tilt_are_exact(self):
        """The ratio emulated / exact is the same number at three amplitudes and three tilts:
        not small, identical -- the parameter is not interpolated at all. And the input leaves
        of the parameters that left the grid come back at the sampled value, not the centre."""
        from desilike.emulators import Emulator

        space = _space('omega_cdm', 'omega_b', 'logA', 'n_s')
        emulator = Emulator(_fourier_cosmology(free=()), space).train(budget=1)
        assert emulator.params == ['omega_cdm', 'omega_b']
        centre = dict(space.center)
        for name, deltas in [('logA', (-0.12, 0., 0.12)), ('n_s', (-0.02, 0., 0.02))]:
            ratios = []
            for delta in deltas:
                point = {**centre, name: centre[name] + delta}
                ratios.append(_relative(emulator, point))
            for ratio in ratios[1:]:
                np.testing.assert_allclose(ratio, ratios[0], rtol=0., atol=1e-11)
        point = {**centre, 'logA': 3.15, 'n_s': 0.95}
        predicted = emulator.predict(**point)
        assert np.isclose(float(predicted['input.logA']), 3.15) and np.isclose(float(predicted['input.n_s']), 0.95)
        assert np.allclose(_relative(emulator, point, leaf='primordial.pk'), 1., rtol=1e-10)

    def test_one_emulator_per_sector(self):
        """Spectra, background and sound horizon are three sectors of a `CosmologyEmulator`,
        each expanding only what its own leaves keep: with `rs_drag` in one node set, `h` would
        stay on the power spectrum's grid. The deployed calculator reproduces every requirement
        of the joint cosmology, and the exact parameters reach it at their sampled values."""
        from desilike.base import build
        from desilike.emulators import Emulator
        from desilike.theories.primordial_cosmology import (CosmologyEmulator, FourierEmulator,
                                                             BackgroundEmulator, ThermodynamicsEmulator)

        cosmo = _fourier_cosmology(background=True)
        assert cosmo.get_emulator_cls() is CosmologyEmulator
        emulator = Emulator(cosmo, _space(*BOUNDS))
        sectors = emulator._sectors
        assert [type(sub) for sub in sectors.values()] == [FourierEmulator, BackgroundEmulator, ThermodynamicsEmulator]
        assert sectors['fourier'].params == sectors['thermodynamics'].params == ['omega_cdm', 'omega_b']
        # the background expands in the density fractions, with h a training parameter that
        # nothing depends on.  `w0pwa` is there too: the expansion basis replaces the dark-energy
        # pair by the sum `w0 + wa` (bounded above, so declared with a logit transform -- see
        # `_transforms`), and the distances depend on it through the whole expansion history, so
        # that combination stays on the grid while `w0_fld` itself is applied exactly.
        background = sectors['background']
        assert background.params == ['Omega_cdm', 'Omega_b', 'w0pwa'] and 'h' in background.exact_params
        assert set(background.training.params) - set(background.space.params) == {'Omega_cdm', 'Omega_b', 'w0pwa'}
        assert 'thermodynamics.rs_drag' in sectors['thermodynamics'].children_leafnames
        assert 'background.efunc' in sectors['background'].children_leafnames
        # the age is routed in h (its unit) but not in the dark energy, so it keeps w0 and wa on
        # this sector's grid -- and on this sector's only
        with_age = Emulator(_fourier_cosmology(background=True, age=True), _space(*BOUNDS))._sectors
        assert with_age['background'].params == ['Omega_cdm', 'Omega_b', 'w0_fld', 'w0pwa']
        assert with_age['fourier'].params == ['omega_cdm', 'omega_b']

        space = _space('omega_cdm', 'omega_b', 'logA', 'n_s')
        emulator = Emulator(_fourier_cosmology(free=(), background=True, age=True), space).train(budget=1)
        deployed = emulator.to_calculator()
        exact = _fourier_cosmology(free=(), background=True, age=True)
        point = {'omega_cdm': 0.118, 'omega_b': 0.0222, 'logA': 3.1, 'n_s': 0.96}
        fast, slow = build(deployed), build(exact)
        fast(point), slow(point)
        for key in [('fourier.pk', dict(of='delta_cb', z=Z, k=K)), ('fourier.pk', dict(of='theta_cb', z=Z, k=K)),
                    ('background.efunc', dict(z=Z)), ('background.comoving_transverse_distance', dict(z=Z)),
                    ('thermodynamics.rs_drag', {}), ('background.age', {}), ('primordial.pk', dict(k=K))]:
            np.testing.assert_allclose(deployed.get(key[0], **key[1]), exact.get(key[0], **key[1]), rtol=2e-3)
        assert np.isclose(float(deployed._param_values['logA']), 3.1)


class TestFourierEmulatorAgainstClass:

    @pytest.fixture(scope='class')
    def emulator(self):
        from desilike.emulators import Emulator

        space = _space(*BOUNDS)
        emulator = Emulator(_fourier_cosmology(engine='class', background=True), space)
        assert emulator._sectors['fourier'].params == ['omega_cdm', 'omega_b']
        # `w0pwa` on the background grid: see `TestFourierEmulator.test_one_emulator_per_sector`.
        assert emulator._sectors['background'].params == ['Omega_cdm', 'Omega_b', 'w0pwa']
        return emulator.train(budget=1)

    def test_dilation_and_growth_route_h_w0_wa(self, emulator):
        """h through the dilation and the analytic growth, w0 / wa through the growth alone:
        none is on the grid, so the residual is the routing's own, measured 2026-09-04 against
        CLASS at 1e-4 for h moved 10% and, for w0 / wa, below 1e-5 above k = 0.01 but rising
        towards large scales (5e-4 at k = 3e-3 for w0 moved 0.18, 7e-3 at 1e-3 for wa moved
        0.5): the analytic growth is scale-free, and CLASS's delta_cb is not quite, there.

        Inside the grid: the dilation reads the reference-frame leaf at k s, and above
        kmax / s that is the power-law tail, 1.3e-3 on the last points here. A template's grid
        ends at k = 10, so its tail sits far from any data; this test's ends at 0.5."""
        centre = dict(emulator.space.center)
        for name, delta, tolerance in [('h', 0.04, 3e-4), ('h', -0.04, 3e-4), ('w0_fld', -0.18, 3e-4),
                                       ('wa_fld', 0.45, 5e-4), ('wa_fld', -0.5, 5e-4)]:
            point = {**centre, name: centre[name] + delta}
            for leaf in ('fourier.pk|of=delta_cb,delta_cb', 'fourier.pk|of=theta_cb,theta_cb'):
                ratio = _relative(emulator, point, leaf=leaf, kmin=1e-2, kmax=0.4)
                assert np.max(np.abs(ratio - 1.)) < tolerance, (name, delta, leaf, np.max(np.abs(ratio - 1.)))
            for leaf in ('background.efunc', 'background.comoving_transverse_distance', 'thermodynamics.rs_drag'):
                assert np.max(np.abs(_relative(emulator, point, leaf=leaf) - 1.)) < 1e-3, (name, leaf)

    def test_round_trip_through_hdf5(self, emulator, tmp_path):
        from desilike.emulators import Emulator

        point = {**emulator.space.center, 'h': 0.7, 'wa_fld': 0.2}
        path = emulator.write(str(tmp_path / 'fourier.h5'))
        reloaded = Emulator.read(path)
        before, after = emulator.predict(**point), reloaded.predict(**point)
        for key in before:
            np.testing.assert_allclose(np.asarray(after[key]), np.asarray(before[key]), rtol=1e-12, atol=0.)


class TestCosmologyEmulator:

    def test_one_emulator_per_sector(self, tmp_path):
        """A cosmology serving a CMB likelihood and a full-shape theory dispatches to the
        composite: the harmonic sector keeps tau on the grid in the theta basis, the Fourier
        sector has no tau axis and its amplitude off the grid, and the merged prediction
        matches the joint calculator on both, through a file and through `to_calculator`."""
        from desilike.base import build
        from desilike.emulators import Emulator
        from desilike.theories.primordial_cosmology import (CosmologyEmulator, HarmonicEmulator,
                                                             FourierEmulator)

        cosmo = _fourier_cosmology(engine='class', free=('tau_reio',), harmonic=True, background=True)
        assert cosmo.get_emulator_cls() is CosmologyEmulator
        from desilike.emulators import Space
        space = Space(bounds={'omega_cdm': BOUNDS['omega_cdm'], 'logA': BOUNDS['logA'], 'tau_reio': (0.04, 0.07)})
        emulator = Emulator(cosmo, space)
        harmonic, fourier = emulator._sectors['harmonic'], emulator._sectors['fourier']
        assert isinstance(harmonic, HarmonicEmulator) and isinstance(fourier, FourierEmulator)
        assert set(emulator._sectors) == {'harmonic', 'fourier', 'background', 'thermodynamics'}
        assert 'tau_reio' in harmonic.params and 'logA' in harmonic.params
        for name in ('fourier', 'background', 'thermodynamics'):
            sub = emulator._sectors[name]
            assert sub.params == ['omega_cdm'] and set(sub.exact_params) == {'logA', 'tau_reio'}, name
        nodes = emulator.nodes(budget=1)
        assert len(nodes['fourier']) < len(nodes['harmonic'])
        emulator.train(budget=1)

        point = {'omega_cdm': 0.118, 'logA': 3.1, 'tau_reio': 0.06}
        predicted, exact = emulator.predict(**point), emulator.compute(point)
        assert set(emulator.children_leafnames) <= set(predicted)
        np.testing.assert_allclose(predicted['fourier.pk|of=delta_cb,delta_cb'],
                                   exact['fourier.pk|of=delta_cb,delta_cb'], rtol=2e-3)
        cl = 'harmonic.unlensed_cl|ellmax=60.tt'
        np.testing.assert_allclose(predicted[cl][2:], exact[cl][2:], rtol=5e-3)
        assert np.isclose(float(predicted['input.tau_reio']), 0.06)

        reloaded = Emulator.read(emulator.write(str(tmp_path / 'joint.h5')))
        assert isinstance(reloaded, CosmologyEmulator)
        again = reloaded.predict(**point)
        for key in predicted:
            np.testing.assert_allclose(np.asarray(again[key]), np.asarray(predicted[key]), rtol=1e-12, atol=0.)

        deployed = reloaded.to_calculator(calculator=cosmo)
        build(deployed)(point)
        np.testing.assert_allclose(deployed.get('thermodynamics.rs_drag'), exact['thermodynamics.rs_drag'], rtol=1e-3)
        np.testing.assert_allclose(deployed.get('fourier.pk', of='delta_cb', z=Z, k=K),
                                   np.squeeze(exact['fourier.pk|of=delta_cb,delta_cb']), rtol=2e-3)
        np.testing.assert_allclose(deployed.get('harmonic.unlensed_cl', ellmax=60)['tt'][2:], exact[cl][2:], rtol=5e-3)


def test_a_derived_leaf_keeps_its_own_redshift():
    """A derived sigma8 is one number at one redshift, and the requirement it reads may serve
    several: a template asks `fourier.sigma8_z` at z = 0.8 and the derived parameter at z = 0, so
    the merged grid is two long.

    The routing factor has to follow the getter, not the merged grid, or it broadcasts the scalar
    it divides into a vector. That is not caught by anything local -- the emulator trains, writes
    and predicts -- and surfaces hundreds of steps into a chain as emcee refusing a blob array
    whose width moved under it (measured: a 12-wide derived row against a 14-wide one).
    """
    from desilike.base import build
    from desilike.emulators import Emulator
    from desilike.parameter import Parameter
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology

    params = CosmoprimoCosmology.propose_params(fiducial='DESI')
    for name in ('sigma8_m', 'sigma8_cb'):
        params.set(Parameter(name, value=0., derived=True))
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI', params=params,
                                requirements={'fourier.pk': [{'of': 'delta_cb', 'z': Z, 'k': K}],
                                              'fourier.sigma8_z': [{'of': 'delta_cb', 'z': Z}]})
    build(cosmo)
    emulator = Emulator(cosmo, _space('omega_cdm', 'logA')).train(budget=1)
    point = {'omega_cdm': 0.121, 'logA': 3.05}
    predicted, exact = emulator.predict(**point), emulator.compute(point)
    for name in ('sigma8_m', 'sigma8_cb'):
        for leaf in (f'derived.{name}', f'derived_params.{name}'):
            assert np.ndim(predicted[leaf]) == 0, (leaf, np.shape(predicted[leaf]))
            np.testing.assert_allclose(predicted[leaf], exact[leaf], rtol=1e-3)
    # the requirement itself keeps the merged grid it was registered on
    assert np.shape(predicted['fourier.sigma8_z|of=delta_cb,delta_cb']) == (2,)
