"""Tests for desilike/parameter.py"""

import copy
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from desilike.parameter import ParameterPrior, Parameter, VariableCollection, NAMESPACE_SEP, decode_name, find_names


# ── ParameterPrior ────────────────────────────────────────────────────────────

class TestParameterPrior:

    def test_improper_flat(self):
        p = ParameterPrior()
        assert not p.is_proper()
        assert not p.is_limited()
        assert p.dist == 'uniform'
        assert bool(p.isin(0.))
        assert not bool(p.isin(float('inf')))
        assert float(p.logpdf(0.)) == 0.
        assert float(p.logpdf(1e10)) == 0.
        with pytest.raises(ValueError):
            p.sample(jax.random.key(0))

    def test_uniform_proper(self):
        p = ParameterPrior(dist='uniform', limits=(0., 1.))
        assert p.is_proper()
        assert p.is_limited()
        assert abs(p.center() - 0.5) < 1e-10
        assert abs(p.std() - 1. / np.sqrt(12.)) < 1e-10
        assert float(p.logpdf(0.5)) > float(p.logpdf(1.5))
        s = p.sample(jax.random.key(42), shape=(100,))
        assert s.shape == (100,)
        assert jnp.all((s >= 0.) & (s <= 1.))

    def test_norm(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05)
        assert p.is_proper()
        assert not p.is_limited()
        assert abs(p.center() - 0.3) < 1e-10
        assert abs(p.std() - 0.05) < 1e-10

    def test_truncnorm(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05, limits=(0.2, 0.4))
        assert p.is_proper()
        assert p.is_limited()
        assert float(p.logpdf(0.15)) == -np.inf
        assert float(p.logpdf(0.45)) == -np.inf
        assert float(p.logpdf(0.3)) > -np.inf

    def test_semi_infinite_norm(self):
        p = ParameterPrior(dist='norm', loc=0., scale=1., limits=(0., np.inf))
        assert p.is_proper()
        assert p.is_limited()

    def test_copy(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05)
        q = p.copy()
        assert p == q
        q.attrs['loc'] = 99.
        assert p.attrs['loc'] == 0.3  # independent

    def test_copy_constructor(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05)
        q = ParameterPrior(p)
        assert p == q

    def test_getstate_setstate(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05, limits=(0., 1.))
        state = p.__getstate__()
        q = ParameterPrior(**state)
        assert p == q

    def test_affine_transform(self):
        p = ParameterPrior(dist='norm', loc=0., scale=1.)
        q = p.affine_transform(loc=1., scale=2.)
        assert abs(q.attrs['loc'] - 1.) < 1e-10
        assert abs(q.attrs['scale'] - 2.) < 1e-10

    def test_affine_transform_limits(self):
        p = ParameterPrior(dist='uniform', limits=(0., 2.))
        q = p.affine_transform(loc=1., scale=3.)
        assert abs(q.limits[0] - 1.) < 1e-10
        assert abs(q.limits[1] - 7.) < 1e-10

    def test_invalid_limits(self):
        with pytest.raises(ValueError):
            ParameterPrior(limits=(1., 0.))

    def test_dict_constructor(self):
        p = ParameterPrior({'dist': 'norm', 'loc': 0.3, 'scale': 0.05})
        assert p.dist == 'norm'
        assert p.attrs['loc'] == 0.3

    def test_logpdf_jit(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05)
        fn = jax.jit(p.logpdf)
        assert abs(float(fn(0.3)) - float(p.logpdf(0.3))) < 1e-12

    def test_logpdf_grad(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05)
        g = jax.grad(lambda x: p.logpdf(x))(jnp.array(0.3))
        assert abs(float(g)) < 1e-10  # log-prob is max at loc=0.3, grad=0

    def test_logpdf_truncnorm_grad(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05, limits=(0.2, 0.4))
        g = jax.grad(lambda x: p.logpdf(x))(jnp.array(0.3))
        assert np.isfinite(float(g))

    def test_sample_jax_key(self):
        p = ParameterPrior(dist='norm', loc=0.3, scale=0.05)
        s = p.sample(jax.random.key(0), shape=(50,))
        assert s.shape == (50,)
        assert abs(float(jnp.mean(s)) - 0.3) < 0.1



# ── Parameter ─────────────────────────────────────────────────────────────────

class TestParameter:

    def test_basic(self):
        p = Parameter('omega_m', value=0.3)
        assert p.name == 'omega_m'
        assert p.basename == 'omega_m'
        assert p.namespace == ''
        assert p.value == 0.3
        assert p.fixed  # no prior given

    def test_namespace_in_name(self):
        p = Parameter('galaxy.omega_m', value=0.3)
        assert p.name == 'galaxy.omega_m'
        assert p.basename == 'omega_m'
        assert p.namespace == 'galaxy'

    def test_namespace_kwarg(self):
        p = Parameter('omega_m', value=0.3, namespace='galaxy')
        assert p.name == 'galaxy.omega_m'
        assert p.namespace == 'galaxy'

    def test_namespace_kwarg_plus_embedded(self):
        p = Parameter('sub.omega_m', value=0.3, namespace='galaxy')
        assert p.name == 'galaxy.sub.omega_m'
        assert p.namespace == 'galaxy.sub'

    def test_copy_constructor(self):
        p = Parameter('omega_m', value=0.3)
        q = Parameter(p)
        assert q.name == 'omega_m'
        assert q.value == 0.3

    def test_prior_sets_fixed_false(self):
        p = Parameter('omega_m', prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
        assert not p.fixed
        assert p.varied

    def test_value_from_prior(self):
        p = Parameter('omega_m', prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
        assert abs(p.value - 0.3) < 1e-10

    def test_value_from_uniform_prior(self):
        p = Parameter('a', prior={'dist': 'uniform', 'limits': (0., 2.)})
        assert abs(p.value - 1.0) < 1e-10

    def test_ref_defaults_to_prior(self):
        p = Parameter('omega_m', prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
        assert p.ref == p.prior

    def test_fd_eps_from_ref(self):
        p = Parameter('omega_m', prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
        assert abs(p.fd_eps - p.ref.std()) < 1e-10
        assert abs(p.fd_eps - 0.01) < 1e-10

    def test_fd_eps_explicit(self):
        p = Parameter('omega_m', prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01}, fd_eps=0.05)
        assert abs(p.fd_eps - 0.05) < 1e-10

    def test_derived_bool(self):
        p = Parameter('logL', derived=True)
        assert p.derived is True
        assert not p.solved

    def test_derived_expression(self):
        omega_b = Parameter('omega_b', value=0.02)
        h = Parameter('h', value=0.7)
        p = Parameter('omega_b_h2', derived='omega_b * h**2', depends={'omega_b': omega_b, 'h': h})
        assert isinstance(p.derived, str)
        assert set(p.depends.keys()) == {'omega_b', 'h'}
        assert abs(p() - 0.02 * 0.7 ** 2) < 1e-12

    def test_derived_expression_custom_depends(self):
        omega_b = Parameter('omega_b', value=0.02)
        h = Parameter('h', value=0.7)
        p = Parameter('omega_bh2', derived='ob * h**2', depends={'ob': omega_b, 'h': h})
        assert abs(p() - 0.02 * 0.7 ** 2) < 1e-12

    def test_latex_basic(self):
        p = Parameter('omega_m', latex=r'\Omega_m')
        assert p.latex() == r'\Omega_m'
        assert p.latex(inline=True) == r'$\Omega_m$'
        assert p.latex(namespace=True) == r'\Omega_m'  # no namespace

    def test_latex_with_namespace(self):
        p = Parameter('galaxy.omega_m', latex=r'\Omega_m')
        assert p.latex(namespace=True) == r'\Omega_{m, \mathrm{galaxy}}'

    def test_latex_fallback(self):
        p = Parameter('omega_m', value=0.3)
        assert p.latex() == 'omega_m'

    def test_clone_value(self):
        p = Parameter('omega_m', value=0.3)
        q = p.clone(value=0.35)
        assert q.value == 0.35
        assert p.value == 0.3  # original unchanged

    def test_clone_namespace(self):
        p = Parameter('galaxy.omega_m', value=0.3)
        q = p.clone(namespace='cosmo')
        assert q.name == 'cosmo.omega_m'
        assert q.basename == 'omega_m'

    def test_clone_drop_namespace(self):
        p = Parameter('galaxy.omega_m', value=0.3)
        q = p.clone(namespace='')
        assert q.name == 'omega_m'

    def test_copy(self):
        p = Parameter('omega_m', prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
        q = copy.copy(p)
        assert q == p
        q.prior.attrs['loc'] = 99.
        assert p.prior.attrs['loc'] == 0.3  # independent

    def test_getstate_setstate_roundtrip(self):
        p = Parameter('galaxy.omega_m', value=0.3, prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01},
                      latex=r'\Omega_m', fixed=False, derived=False, fd_eps=0.01)
        state = p.__getstate__()
        q = Parameter.__new__(Parameter)
        q.__setstate__(state)
        assert q.name == p.name
        assert q.value == p.value
        assert q.prior == p.prior
        assert q._latex == p._latex


    def test_eq_and_hash(self):
        p = Parameter('omega_m', value=0.3)
        q = Parameter('omega_m', value=0.5)
        assert p == q  # same name
        assert hash(p) == hash(q)
        r = Parameter('omega_b', value=0.3)
        assert p != r

    def test_repr(self):
        p = Parameter('omega_m', value=0.3)
        assert 'omega_m' in repr(p)
        assert 'fixed' in repr(p)

    def test_str(self):
        p = Parameter('omega_m', value=0.3)
        assert str(p) == 'omega_m'


# ── VariableCollection ───────────────────────────────────────────────────────

class TestVariableCollection:

    def test_empty(self):
        c = VariableCollection()
        assert len(c) == 0
        assert c.names() == []

    def test_from_scalar_dict(self):
        c = VariableCollection({'omega_m': 0.3, 'z': 0.5})
        assert len(c) == 2
        assert set(c.names()) == {'omega_m', 'z'}
        assert c['omega_m'].value == 0.3

    def test_from_dict_of_dicts(self):
        c = VariableCollection({'omega_m': {'value': 0.3, 'prior': {'dist': 'norm', 'loc': 0.3, 'scale': 0.01}}})
        p = c['omega_m']
        assert not p.fixed
        assert p.prior.dist == 'norm'
        assert abs(p.prior.attrs['loc'] - 0.3) < 1e-10

    def test_from_list(self):
        params = [Parameter('omega_m', value=0.3), Parameter('z', value=0.5)]
        c = VariableCollection(params)
        assert len(c) == 2
        assert 'omega_m' in c

    def test_from_collection(self):
        c1 = VariableCollection({'omega_m': 0.3, 'z': 0.5})
        c2 = VariableCollection(c1)
        assert c2.names() == c1.names()
        # Variable objects are shared (shallow copy of the list)
        assert c2['omega_m'] is c1['omega_m']
        # Lists are independent: adding to c2 doesn't affect c1
        c2.set(Parameter('sigma8', value=0.8))
        assert 'sigma8' not in c1.names()

    def test_set_insert_and_replace(self):
        c = VariableCollection()
        c.set(Parameter('omega_m', value=0.3))
        assert len(c) == 1
        c.set(Parameter('omega_m', value=0.35))  # replace
        assert len(c) == 1
        assert c['omega_m'].value == 0.35

    def test_getitem_by_index(self):
        c = VariableCollection({'omega_m': 0.3, 'z': 0.5})
        assert c[0].name == 'omega_m'
        assert c[1].name == 'z'

    def test_getitem_missing(self):
        c = VariableCollection({'omega_m': 0.3})
        with pytest.raises(KeyError):
            _ = c['missing']

    def test_contains(self):
        c = VariableCollection({'omega_m': 0.3})
        assert 'omega_m' in c
        assert 'z' not in c
        p = Parameter('omega_m', value=0.5)
        assert p in c

    def test_iter(self):
        c = VariableCollection({'omega_m': 0.3, 'z': 0.5})
        names = [p.name for p in c]
        assert names == ['omega_m', 'z']

    def test_names_select(self):
        c = VariableCollection({
            'omega_m': {'value': 0.3, 'prior': {'dist': 'norm', 'loc': 0.3, 'scale': 0.01}},
            'z': 0.5,
        })
        assert set(c.names(fixed=False)) == {'omega_m'}
        assert set(c.names(fixed=True)) == {'z'}

    def test_select(self):
        c = VariableCollection({
            'a': {'value': 1.0, 'prior': {'dist': 'norm', 'loc': 1., 'scale': 0.1}},
            'b': 2.0,
        })
        varied = c.select(fixed=False)
        assert len(varied) == 1
        assert varied[0].name == 'a'

    def test_derived_call(self):
        omega_b = Parameter('omega_b', value=0.02)
        h = Parameter('h', value=0.7)
        c = VariableCollection()
        c.set(Parameter('omega_bh2', derived='omega_b * h**2', depends={'omega_b': omega_b, 'h': h}))
        result = c['omega_bh2']()
        assert abs(result - 0.02 * 0.7 ** 2) < 1e-12

    def test_add(self):
        c1 = VariableCollection({'omega_m': 0.3})
        c2 = VariableCollection({'z': 0.5})
        c3 = c1 + c2
        assert set(c3.names()) == {'omega_m', 'z'}
        assert len(c1) == 1  # originals unchanged

    def test_add_override(self):
        c1 = VariableCollection({'omega_m': 0.3})
        c2 = VariableCollection({'omega_m': 0.35})
        c3 = c1 + c2
        assert len(c3) == 1
        assert c3['omega_m'].value == 0.35

    def test_sub(self):
        c1 = VariableCollection({'omega_m': 0.3, 'z': 0.5})
        c2 = VariableCollection({'z': 0.5})
        c3 = c1 - c2
        assert c3.names() == ['omega_m']

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            VariableCollection(42)

    def test_repr(self):
        c = VariableCollection({'omega_m': 0.3})
        assert 'omega_m' in repr(c)


# ── decode_name ───────────────────────────────────────────────────────────────

class TestDecodeName:
    def test_no_brackets(self):
        strings, ranges = decode_name('omega_m')
        assert strings == ['omega_m']
        assert ranges == []

    def test_single_bracket(self):
        strings, ranges = decode_name('a_[0:3]')
        assert ranges == [range(0, 3)]
        assert strings[0] == 'a_'

    def test_multi_bracket(self):
        strings, ranges = decode_name('a_[-4:5:2]_b_[0:2]')
        assert ranges == [range(-4, 5, 2), range(0, 2)]

    def test_wildcard_passthrough(self):
        # '*' is not touched by decode_name; find_names performs the substitution
        strings, ranges = decode_name('omega_*')
        assert ranges == []
        assert strings == ['omega_*']


# ── find_names ────────────────────────────────────────────────────────────────

class TestFindNames:
    NAMES = ['omega_m', 'omega_b', 'sigma8', 'A_0', 'A_1', 'A_2', 'ns']

    def test_exact(self):
        assert find_names(self.NAMES, 'sigma8') == ['sigma8']

    def test_star_wildcard(self):
        assert find_names(self.NAMES, 'omega_*') == ['omega_m', 'omega_b']

    def test_star_matches_all(self):
        assert find_names(self.NAMES, '*') == self.NAMES

    def test_range(self):
        assert find_names(self.NAMES, 'A_[0:2]') == ['A_0', 'A_1']

    def test_range_full(self):
        assert find_names(self.NAMES, 'A_[0:3]') == ['A_0', 'A_1', 'A_2']

    def test_list_of_patterns(self):
        result = find_names(self.NAMES, ['omega_*', 'ns'])
        assert result == ['omega_m', 'omega_b', 'ns']

    def test_no_match_quiet(self):
        assert find_names(self.NAMES, 'z_*') == []

    def test_no_match_not_quiet(self):
        with pytest.raises(ValueError):
            find_names(self.NAMES, 'z_*', quiet=False)

    def test_empty_allnames(self):
        assert find_names([], 'omega_*') == []

    def test_regex_pattern(self):
        import re
        pattern = re.compile(r'A_\d+$')
        assert find_names(self.NAMES, pattern) == ['A_0', 'A_1', 'A_2']


# ── VariableCollection.select with wildcards ──────────────────────────────────

class TestSelectWildcard:
    def _make(self):
        c = VariableCollection()
        for name in ['omega_m', 'omega_b', 'sigma8', 'A_0', 'A_1', 'ns']:
            c.set(Parameter(name, value=1.0))
        return c

    def test_select_exact_name(self):
        c = self._make()
        s = c.select(name='sigma8')
        assert s.names() == ['sigma8']

    def test_select_star_wildcard(self):
        c = self._make()
        s = c.select(name='omega_*')
        assert set(s.names()) == {'omega_m', 'omega_b'}

    def test_select_range(self):
        c = self._make()
        s = c.select(name='A_[0:2]')
        assert s.names() == ['A_0', 'A_1']

    def test_select_combined_fixed_and_name(self):
        c = VariableCollection()
        c.set(Parameter('omega_m', value=0.3, prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01}))
        c.set(Parameter('omega_b', value=0.05, prior={'dist': 'norm', 'loc': 0.05, 'scale': 0.005}))
        c.set(Parameter('z', value=0.5))   # fixed (no prior)
        # wildcard name + attribute filter
        s = c.select(name='omega_*', fixed=False)
        assert set(s.names()) == {'omega_m', 'omega_b'}

    def test_select_no_match_returns_empty(self):
        c = self._make()
        s = c.select(name='h_*')
        assert len(s) == 0

    def test_select_preserves_type(self):
        """select on a VariableCollection returns a VariableCollection, not a base object."""
        c = self._make()
        s = c.select(name='omega_*')
        assert type(s) is VariableCollection

    def test_select_basename_wildcard(self):
        c = VariableCollection()
        c.set(Parameter('ns.omega_m', value=0.3))
        c.set(Parameter('ns.omega_b', value=0.05))
        c.set(Parameter('ns.sigma8', value=0.8))
        s = c.select(basename='omega_*')
        assert set(s.names()) == {'ns.omega_m', 'ns.omega_b'}

    def test_select_namespace_wildcard(self):
        c = VariableCollection()
        c.set(Parameter('survey1.omega_m', value=0.3))
        c.set(Parameter('survey2.omega_m', value=0.3))
        c.set(Parameter('omega_b', value=0.05))
        s = c.select(namespace='survey*')
        assert set(s.names()) == {'survey1.omega_m', 'survey2.omega_m'}

