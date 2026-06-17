"""Tests for primordial cosmology calculators."""

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
