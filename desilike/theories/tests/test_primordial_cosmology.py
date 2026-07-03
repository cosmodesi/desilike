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