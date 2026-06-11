"""Tests for desilike/samples.py."""

import os
import tempfile

import numpy as np
import pytest

from desilike.parameter import Variable, Parameter
from desilike.samples import MCSamples

RNG = np.random.default_rng(42)
N = 500   # number of samples used throughout


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scalar_samples():
    """500-sample chain with two scalar params and a log-posterior."""
    s = MCSamples()
    s['omega_m'] = RNG.normal(0.3, 0.01, N)
    s['sigma8']  = RNG.normal(0.8, 0.02, N)
    s.logposterior = -0.5 * RNG.chisquare(2, N)
    return s


@pytest.fixture
def vector_samples():
    """200-sample chain with a scalar param and a 15-element pk vector."""
    s = MCSamples()
    s['A'] = RNG.normal(1.0, 0.1, 200)
    pk_var = Variable('pk', value=np.zeros(15))   # shape=(15,)
    s[pk_var] = RNG.normal(1., 0.05, (200, 15))
    return s


@pytest.fixture
def weighted_samples():
    """300-sample chain with non-trivial aweight."""
    s = MCSamples()
    s['x'] = RNG.normal(0., 1., 300)
    s.aweight = RNG.uniform(0.5, 1.5, 300)
    s.logposterior = -0.5 * s['x']._value ** 2
    return s


# ── shape / size ──────────────────────────────────────────────────────────────

def test_shape_scalar(scalar_samples):
    assert scalar_samples.shape == (N,)
    assert scalar_samples.ndim == 1
    assert scalar_samples.size == N
    assert len(scalar_samples) == N


def test_shape_vector(vector_samples):
    assert vector_samples.shape == (200,)
    assert vector_samples['pk'].shape == (15,)
    assert vector_samples['pk']._value.shape == (200, 15)


def test_shape_empty():
    s = MCSamples()
    assert s.shape == ()
    assert s.size == 0
    assert len(s) == 0


# ── construction ──────────────────────────────────────────────────────────────

def test_construct_from_dict():
    arr = RNG.normal(0., 1., 100)
    s = MCSamples({'x': arr, 'y': arr * 2})
    assert s.shape == (100,)
    assert 'x' in s and 'y' in s
    np.testing.assert_array_equal(s['x']._value, arr)


def test_construct_from_list_of_variables():
    v = Variable('x')
    v._value = np.zeros(50)
    s = MCSamples([v])
    assert s.shape == (50,)


def test_construct_copy():
    s1 = MCSamples({'x': np.ones(10)})
    s2 = MCSamples(s1)
    s2['x'] = np.zeros(10)   # modify copy
    np.testing.assert_array_equal(s1['x']._value, np.ones(10))  # original unchanged


def test_setitem_string_key_infers_shape(scalar_samples):
    """Once shape is known, __setitem__ infers var.shape from value."""
    scalar_samples['vec'] = RNG.normal(0., 1., (N, 5))
    assert scalar_samples['vec'].shape == (5,)
    assert scalar_samples['vec']._value.shape == (N, 5)


def test_setitem_variable_key_scalar(vector_samples):
    """Variable key with shape=() stores correctly."""
    v = Variable('b')          # shape=()
    vector_samples[v] = RNG.normal(0., 1., 200)
    assert vector_samples['b'].shape == ()
    assert vector_samples['b']._value.shape == (200,)


def test_setitem_shape_mismatch_raises():
    s = MCSamples({'x': np.zeros(100)})
    with pytest.raises(ValueError):
        s['y'] = np.zeros(50)   # leading shape (50,) != samples.shape (100,)


# ── slicing ───────────────────────────────────────────────────────────────────

def test_getitem_string_returns_variable(scalar_samples):
    var = scalar_samples['omega_m']
    assert isinstance(var, Variable)
    assert var._value.shape == (N,)


def test_getitem_int_returns_samples(scalar_samples):
    s = scalar_samples[0]
    assert s.shape == (1,)
    assert s['omega_m']._value.shape == (1,)


def test_getitem_slice(scalar_samples):
    s = scalar_samples[10:20]
    assert s.shape == (10,)
    np.testing.assert_array_equal(s['omega_m']._value,
                                  scalar_samples['omega_m']._value[10:20])


def test_getitem_array(scalar_samples):
    idx = np.array([0, 5, 10])
    s = scalar_samples[idx]
    assert s.shape == (3,)


# ── reshape / ravel / concatenate ─────────────────────────────────────────────

def test_reshape(scalar_samples):
    s2 = scalar_samples.reshape(50, 10)
    assert s2.shape == (50, 10)
    assert s2['omega_m']._value.shape == (50, 10)
    # values unchanged
    np.testing.assert_array_equal(s2['omega_m']._value.ravel(),
                                  scalar_samples['omega_m']._value)


def test_ravel_roundtrip(scalar_samples):
    s2 = scalar_samples.reshape(50, 10).ravel()
    assert s2.shape == (N,)


def test_concatenate(scalar_samples):
    s1 = scalar_samples[:N // 2]
    s2 = scalar_samples[N // 2:]
    sc = MCSamples.concatenate(s1, s2)
    assert sc.shape == (N,)
    np.testing.assert_array_equal(sc['omega_m']._value,
                                  scalar_samples['omega_m']._value)


def test_concatenate_list_syntax(scalar_samples):
    halves = [scalar_samples[:N // 2], scalar_samples[N // 2:]]
    sc = MCSamples.concatenate(halves)
    assert sc.shape == (N,)


def test_concatenate_name_mismatch_raises():
    s1 = MCSamples({'x': np.zeros(10)})
    s2 = MCSamples({'y': np.zeros(10)})
    with pytest.raises(ValueError):
        MCSamples.concatenate(s1, s2)


# ── special attributes ────────────────────────────────────────────────────────

def test_logposterior_default_zeros(scalar_samples):
    s = MCSamples({'x': np.zeros(10)})
    np.testing.assert_array_equal(s.logposterior, np.zeros(10))


def test_aweight_default_ones(scalar_samples):
    s = MCSamples({'x': np.zeros(10)})
    np.testing.assert_array_equal(s.aweight, np.ones(10))


def test_fweight_default_ones():
    s = MCSamples({'x': np.zeros(10)})
    np.testing.assert_array_equal(s.fweight, np.ones(10, dtype='i8'))


def test_weight_product(weighted_samples):
    np.testing.assert_array_equal(
        weighted_samples.weight,
        weighted_samples.aweight * weighted_samples.fweight
    )


# ── statistics ────────────────────────────────────────────────────────────────

def test_mean_scalar(scalar_samples):
    w = scalar_samples.weight.ravel()
    expected = np.average(scalar_samples['omega_m']._value, weights=w)
    got = scalar_samples.mean('omega_m')
    assert abs(float(got) - float(expected)) < 1e-12


def test_mean_multi_param(scalar_samples):
    result = scalar_samples.mean(['omega_m', 'sigma8'])
    assert len(result) == 2


def test_mean_vector(vector_samples):
    m = vector_samples.mean('pk')
    assert m.shape == (15,)


def test_mean_all_params(scalar_samples):
    result = scalar_samples.mean()
    assert isinstance(result, list)


def test_var_scalar(scalar_samples):
    v = scalar_samples.var('omega_m')
    assert np.isscalar(v) or v.shape == ()


def test_std_scalar(scalar_samples):
    s = scalar_samples.std('omega_m')
    v = scalar_samples.var('omega_m')
    assert abs(float(s) - float(v) ** 0.5) < 1e-12


def test_std_vector(vector_samples):
    s = vector_samples.std('pk')
    assert s.shape == (15,)


def test_mean_weighted(weighted_samples):
    w = weighted_samples.weight.ravel()
    expected = np.average(weighted_samples['x']._value, weights=w)
    got = float(weighted_samples.mean('x'))
    assert abs(got - float(expected)) < 1e-12


def test_median(scalar_samples):
    med = scalar_samples.median('omega_m')
    # For unweighted uniform samples, median ≈ mean
    assert abs(float(med) - float(scalar_samples.mean('omega_m'))) < 0.05


def test_quantile_scalar(scalar_samples):
    lo, hi = scalar_samples.quantile('omega_m', q=(0.1587, 0.8413))
    # 1-sigma interval should be close to std
    std = float(scalar_samples.std('omega_m'))
    mean = float(scalar_samples.mean('omega_m'))
    assert abs(float(lo) - (mean - std)) < 0.1 * std
    assert abs(float(hi) - (mean + std)) < 0.1 * std


def test_quantile_scalar_single_q(scalar_samples):
    med = scalar_samples.quantile('omega_m', q=0.5)
    assert np.ndim(med) == 0 or med.shape == ()


def test_quantile_vector(vector_samples):
    lo, hi = vector_samples.quantile('pk', q=(0.1587, 0.8413))
    assert lo.shape == (15,) and hi.shape == (15,)


def test_interval(scalar_samples):
    lo, hi = scalar_samples.interval('omega_m', nsigmas=1.)
    assert hi > lo


def test_interval_vector(vector_samples):
    lo, hi = vector_samples.interval('pk', nsigmas=1.)
    assert lo.shape == (15,) and hi.shape == (15,)
    assert np.all(hi > lo)


def test_argmax(scalar_samples):
    val = scalar_samples.argmax('omega_m')
    idx = np.argmax(scalar_samples.logposterior.ravel())
    expected = scalar_samples['omega_m']._value.ravel()[idx]
    assert abs(float(val) - float(expected)) < 1e-12


def test_remove_burnin_int(scalar_samples):
    s = scalar_samples.remove_burnin(100)
    assert len(s) == N - 100


def test_remove_burnin_fraction(scalar_samples):
    s = scalar_samples.remove_burnin(0.2)
    assert len(s) == N - int(0.2 * N + 0.5)


# ── export ────────────────────────────────────────────────────────────────────

def test_to_dict(scalar_samples):
    d = scalar_samples.to_dict(['omega_m', 'sigma8'])
    assert set(d.keys()) == {'omega_m', 'sigma8'}
    np.testing.assert_array_equal(d['omega_m'], scalar_samples['omega_m']._value)


def test_to_array_struct(scalar_samples):
    arr = scalar_samples.to_array(['omega_m', 'sigma8'])
    assert arr.shape == (N,)
    assert set(arr.dtype.names) == {'omega_m', 'sigma8'}
    np.testing.assert_array_equal(arr['omega_m'], scalar_samples['omega_m']._value)


def test_to_array_struct_vector(vector_samples):
    arr = vector_samples.to_array(['A', 'pk'])
    assert arr.shape == (200,)
    assert arr['pk'].shape == (200, 15)


# ── I/O ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_roundtrip(scalar_samples, ext, tmp_path):
    fn = str(tmp_path / f'chain{ext}')
    scalar_samples.write(fn)
    loaded = MCSamples.read(fn)
    assert loaded.shape == scalar_samples.shape
    assert loaded.names() == scalar_samples.names()
    np.testing.assert_allclose(loaded['omega_m']._value,
                               scalar_samples['omega_m']._value, rtol=1e-12)
    np.testing.assert_allclose(loaded.logposterior,
                               scalar_samples.logposterior, rtol=1e-12)


@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_vector_variable(vector_samples, ext, tmp_path):
    fn = str(tmp_path / f'vec{ext}')
    vector_samples.write(fn)
    loaded = MCSamples.read(fn)
    assert loaded['pk'].shape == (15,)           # intrinsic shape preserved
    assert loaded['pk']._value.shape == (200, 15)
    np.testing.assert_allclose(loaded['pk']._value, vector_samples['pk']._value,
                               rtol=1e-12)


@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_attrs(tmp_path, ext):
    s = MCSamples({'x': np.zeros(10)}, attrs={'sampler': 'montecarlo', 'nsteps': 1000})
    fn = str(tmp_path / f'attrs{ext}')
    s.write(fn)
    loaded = MCSamples.read(fn)
    assert loaded.attrs.get('sampler') == 'montecarlo'


def test_write_read_parameter_variable(tmp_path):
    """Parameter (not just Variable) round-trips correctly."""
    s = MCSamples()
    p = Parameter('omega_m', value=0.3, prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
    s[p] = RNG.normal(0.3, 0.01, 100)
    fn = str(tmp_path / 'param.h5')
    s.write(fn)
    loaded = MCSamples.read(fn)
    assert isinstance(loaded['omega_m'], Parameter)
    np.testing.assert_allclose(loaded['omega_m']._value, s['omega_m']._value, rtol=1e-12)


# ── 2-D sample shape ──────────────────────────────────────────────────────────

def test_2d_shape():
    """2-D samples shape: (n_chains, n_steps)."""
    s = MCSamples()
    s['x'] = RNG.normal(0., 1., (4, 250))
    assert s.shape == (4, 250)
    assert s.size == 1000
    assert len(s) == 4


def test_2d_reshape_ravel():
    s = MCSamples()
    s['x'] = RNG.normal(0., 1., (4, 250))
    flat = s.ravel()
    assert flat.shape == (1000,)


def test_2d_mean():
    s = MCSamples()
    s['x'] = RNG.normal(0., 1., (4, 250))
    m = s.mean('x')
    assert m.shape == ()   # scalar variable → scalar mean


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
