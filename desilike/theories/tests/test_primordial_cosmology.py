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
        from desilike.base import compile, get_params
        from desilike.parameter import Parameter, VariableCollection
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo0 = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
        vc = get_params(cosmo0)
        vc.set(Parameter('Omega_m', value=0.0, derived=True))
        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI', params=vc)

        pipe = compile(cosmo)
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
        from desilike.base import compile
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
        k = np.linspace(0.01, 0.2, 10)
        cosmo.add_requirements({'fourier.pk': [{'of': 'delta_cb', 'z': 1., 'k': k}]})
        compile(cosmo)()
        result = cosmo.get('fourier.pk', of='delta_cb', z=1., k=k)
        assert result is not None

    def test_section_proxy(self):
        """cosmo.get_fourier().pk(...)/get_background().efunc(...)/etc. match the equivalent
        flat cosmo.get(...) calls exactly."""
        from desilike.base import compile
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
        k = np.linspace(0.01, 0.2, 10)
        cosmo.add_requirements({
            'fourier.pk': [{'of': 'delta_cb', 'z': 1., 'k': k}],
            'fourier.sigma8_z': [{'of': 'delta_cb', 'z': 1.}],
            'background.efunc': [{'z': 1.}],
            'background.comoving_transverse_distance': [{'z': 1.}],
            'params.N_eff': None,
            'thermodynamics.rs_drag': None,
        })
        compile(cosmo)()

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
        from desilike.base import compile, get_params
        from desilike.parameter import Parameter
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo0 = CosmoprimoCosmology(engine='camb', fiducial='DESI')
        vc = get_params(cosmo0)
        vc.set(Parameter('Omega_m', value=0.0, derived=True))
        cosmo = CosmoprimoCosmology(engine='camb', fiducial='DESI', params=vc)
        pipe = compile(cosmo)
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
        from desilike.base import compile, get_params
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology

        cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
        k = np.linspace(0.01, 0.2, 5)
        cosmo.add_requirements({'fourier.pk': [{'of': 'delta_cb', 'z': 0.5, 'k': k}]})
        pipe = compile(cosmo)
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
        from desilike.base import compile, get_params
        from desilike.parameter import Parameter, VariableCollection
        from desilike.theories.primordial_cosmology import ACECosmology

        cosmo = ACECosmology(engine='isitgr', base_dir=self.emulator_base_dir, fiducial='DESI')
        params = get_params(cosmo)
        for name in ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']:
            params.set(Parameter(name, value=1.0, ref={'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                                  fixed=True, prior={'dist': 'uniform', 'limits': [-3., 3.]}))
        cosmo.update(params=params)
        k = np.linspace(0.001, 0.1, 20)
        cosmo.add_requirements({'background.comoving_transverse_distance': [{'z': 0.1}]})
        cosmo.add_requirements({'fourier.pk': [{'of': 'delta_cb', 'z': 0.1, 'k': k}]})
        cosmo.add_requirements({'fourier.pk_now': [{'of': 'delta_cb', 'z': 0.1, 'k': k}]})
        cosmo.add_requirements({'fourier.sigma8_z': [{'of': 'delta_cb', 'z': 0.1}]})
        compile(cosmo)()

    def test_section_proxy(self):
        """cosmo.get_fourier().pk(...)/get_background().comoving_transverse_distance(...) match
        the equivalent flat cosmo.get(...) calls exactly, for a second PrimordialCosmology subclass."""
        from desilike.base import compile, get_params
        from desilike.parameter import Parameter
        from desilike.theories.primordial_cosmology import ACECosmology

        cosmo = ACECosmology(engine='isitgr', base_dir=self.emulator_base_dir, fiducial='DESI')
        params = get_params(cosmo)
        for name in ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']:
            params.set(Parameter(name, value=1.0, ref={'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                                  fixed=True, prior={'dist': 'uniform', 'limits': [-3., 3.]}))
        cosmo.update(params=params)
        k = np.linspace(0.001, 0.1, 20)
        cosmo.add_requirements({'fourier.pk': [{'of': 'delta_cb', 'z': 0.1, 'k': k}]})
        cosmo.add_requirements({'background.comoving_transverse_distance': [{'z': 0.1}]})
        compile(cosmo)()

        np.testing.assert_allclose(cosmo.get_fourier().pk(of='delta_cb', z=0.1, k=k),
                                    cosmo.get('fourier.pk', of='delta_cb', z=0.1, k=k))
        np.testing.assert_allclose(cosmo.get_background().comoving_transverse_distance(z=0.1),
                                    cosmo.get('background.comoving_transverse_distance', z=0.1))

    def test_packaged(self):
        """engine='ace' serves DirectSpectrum2Template's and the CMB likelihoods' requirements
        from the packaged jaxace / jaxmapse / jaxcapse trained emulators, matching cosmoprimo
        (class for pk / sigma8_z / rs_drag, camb for the Cl) at the DESI fiducial."""
        from desilike.base import compile
        from desilike.theories.primordial_cosmology import ACECosmology

        ellmax = 2508
        z_test = 1.
        k = np.geomspace(1e-3, 1., 30)
        # engine='ace' includes the derived sigma8_m and rs_drag parameters by default.
        cosmo = ACECosmology(engine='ace', fiducial='DESI')
        cosmo.add_requirements({
            'fourier.pk': [{'of': 'delta_cb', 'z': z_test, 'k': k}, {'of': 'theta_cb', 'z': z_test, 'k': k}],
            'fourier.sigma8_z': [{'of': 'delta_cb', 'z': z_test}, {'of': 'theta_cb', 'z': z_test}],
            'background.efunc': [{'z': z_test}],
            'background.comoving_transverse_distance': [{'z': z_test}],
            'harmonic.lensed_cl': [{'ellmax': ellmax}],
            'harmonic.lens_potential_cl': [{'ellmax': ellmax}],
        })
        compile(cosmo)()

        import cosmoprimo
        fiducial = cosmoprimo.fiducial.DESI(engine='class')
        fo = fiducial.get_fourier()

        # derived params (default with engine='ace'): rs_drag in Mpc/h, sigma8_m at z = 0
        rs_drag = float(cosmo.derived_params['rs_drag'].value)
        assert np.isclose(rs_drag, fiducial.get_thermodynamics().rs_drag, rtol=2e-4)
        sigma8_m = float(cosmo.derived_params['sigma8_m'].value)
        assert np.isclose(sigma8_m, fo.sigma8_z(0., of='delta_m'), rtol=1e-3)

        # fourier: sigma8_z is total-matter (0.5% off delta_cb), fsigma8 = f_z * sigma8_z
        sigma8 = cosmo.get_fourier().sigma8_z(of='delta_cb', z=z_test)
        fsigma8 = cosmo.get_fourier().sigma8_z(of='theta_cb', z=z_test)
        assert np.isclose(float(sigma8), fo.sigma8_z(z_test, of='delta_m'), rtol=1e-3)
        assert np.isclose(float(sigma8), fo.sigma8_z(z_test, of='delta_cb'), rtol=1e-2)
        assert np.isclose(float(fsigma8), fo.sigma8_z(z_test, of='theta_cb'), rtol=1e-2)

        # fourier: linear pk (delta_cb), and pk_tt = f_z^2 pk_dd with f_z = fsigma8 / sigma8
        pk_dd = np.asarray(cosmo.get_fourier().pk(of='delta_cb', z=z_test, k=k))
        pk_tt = np.asarray(cosmo.get_fourier().pk(of='theta_cb', z=z_test, k=k))
        np.testing.assert_allclose(pk_dd, fo.pk_interpolator(of='delta_cb')(k, z=z_test), rtol=5e-3)
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
        from desilike.base import compile
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
            cosmo = ACECosmology(engine=engine, base_dir=base_dir, fiducial='DESI')
            cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': ellmax}],
                                    'harmonic.lens_potential_cl': [{'ellmax': ellmax}]})
            compile(cosmo)()
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
        from desilike.base import compile, get_params
        from desilike.theories.primordial_cosmology import ACECosmology

        # The artifact sits next to the desilike checkout (on NERSC it lives under the
        # default base_dir, Installer().install_dir / 'ace-emulators').
        base_dir = self.emulator_base_dir.parent
        emulator_dir = base_dir / 'capse_mnuw0wacdm_250001'
        if not (emulator_dir / 'TT' / 'nn_setup.json').is_file():
            pytest.skip('capse_mnuw0wacdm_250001 not available')

        ellmax = 2500
        cosmo = ACECosmology(engine={'harmonic': 'capse_mnuw0wacdm_250001', 'background': 'ACE_mnuw0wacdm_ln10As_basis'},
                             base_dir=base_dir, fiducial='DESI')
        cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': ellmax}]})
        pipe = compile(cosmo)
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
        warning flags priors wider than the training range at compile time."""
        import warnings as _warnings
        import jax
        from desilike.base import compile, get_params
        from desilike.theories.primordial_cosmology import ACECosmology

        k = np.geomspace(1e-3, 1., 20)
        z_test = 1.
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            cosmo = ACECosmology(engine='ace', fiducial='DESI')
            cosmo.add_requirements({
                'fourier.pk': [{'of': 'delta_cb', 'z': z_test, 'k': k}],
                'fourier.pk_now': [{'of': 'delta_cb', 'engine': 'peakaverage', 'z': z_test, 'k': k}],
                'harmonic.lensed_cl': [{'ellmax': 100}],
            })
            pipe = compile(cosmo)
        # h prior [0.1, 10] etc. extend beyond the training ranges: warned at compile.
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
        pipe_template = jax.jit(compile(template))
        defaults = {param.name: param._value for param in get_params(template)}
        assert np.all(np.isfinite(np.asarray(pipe_template(defaults))))
        assert np.all(np.isnan(np.asarray(pipe_template({**defaults, 'h': 3.}))))

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
        from desilike.base import compile, get_params
        from desilike.theories.primordial_cosmology import ACECosmology
        from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template

        cosmo = ACECosmology(engine='ace', fiducial='DESI')
        template = DirectSpectrum2Template(z=0.8, fiducial='DESI', cosmo=cosmo)
        pipe = compile(template)
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
        from desilike.base import compile, get_params
        from desilike.parameter import Parameter
        from desilike.theories.primordial_cosmology import ACECosmology, _get_fiducial

        cosmo = ACECosmology(engine='isitgr', base_dir=self.emulator_base_dir, fiducial='DESI')
        params = get_params(cosmo)
        for name in ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']:
            params.set(Parameter(name, value=1.0, ref={'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                                  fixed=True, prior={'dist': 'uniform', 'limits': [-3., 3.]}))
        cosmo.update(params=params)
        cosmo.add_requirements({'background.comoving_transverse_distance': [{'z': 0.1}]})
        compile(cosmo)()

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