"""Tests for desilike.samples.Covariance and Precision."""

import os
import tempfile

import numpy as np
import pytest

from desilike import Covariance, Precision
from desilike.parameter import Parameter, Variable
from desilike.samples.covariance import _index, _flat_size, _param_sizes


# ── fixtures / helpers ────────────────────────────────────────────────────────

def _scalar_cov():
    """2×2 covariance with a 0.5 off-diagonal correlation."""
    params = ['omega_m', 'sigma8']
    arr = np.array([[0.01**2, 0.5 * 0.01 * 0.02],
                    [0.5 * 0.01 * 0.02, 0.02**2]])
    return Covariance(arr, params=params), params


def _param_cov():
    """Same covariance but backed by Parameter objects (with proposal)."""
    a = Parameter('omega_m', value=0.3,  proposal=0.01, fixed=False)
    b = Parameter('sigma8',  value=0.8,  proposal=0.02, fixed=False)
    arr = np.array([[0.01**2, 0.5 * 0.01 * 0.02],
                    [0.5 * 0.01 * 0.02, 0.02**2]])
    return Covariance(arr, params=[a, b]), [a, b]


def _vector_cov():
    """3×3 covariance: v (shape=(2,)) + s (scalar)."""
    v = Parameter('v', value=[0.1, 0.2], shape=(2,), fixed=False)
    s = Parameter('s', value=0.5, fixed=False)
    arr = np.diag([0.01**2, 0.02**2, 0.03**2])
    return Covariance(arr, params=[v, s])


# ── construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_from_strings(self):
        cov, _ = _scalar_cov()
        assert cov.names() == ['omega_m', 'sigma8']
        assert cov.shape == (2, 2)

    def test_from_parameters(self):
        cov, _ = _param_cov()
        assert cov.names() == ['omega_m', 'sigma8']

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match='size'):
            Covariance(np.eye(3), params=['a', 'b'])

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match='square'):
            Covariance(np.ones((2, 3)), params=['a', 'b'])

    def test_attrs(self):
        cov = Covariance(np.eye(1), params=['a'], attrs={'tag': 'test'})
        assert cov.attrs['tag'] == 'test'


# ── statistics ────────────────────────────────────────────────────────────────

class TestStatistics:
    def test_std(self):
        cov, _ = _scalar_cov()
        std = cov.std()
        np.testing.assert_allclose(std, [0.01, 0.02])

    def test_var(self):
        cov, _ = _scalar_cov()
        np.testing.assert_allclose(cov.var(), [0.01**2, 0.02**2])

    def test_corrcoef(self):
        cov, _ = _scalar_cov()
        corr = cov.corrcoef()
        np.testing.assert_allclose(np.diag(corr), [1., 1.])
        np.testing.assert_allclose(corr[0, 1], 0.5, atol=1e-10)

    def test_fom(self):
        cov, _ = _scalar_cov()
        # det = (0.01*0.02)^2 * (1 - 0.25) = 3e-8; fom = det^{-0.5}
        expected = cov.det() ** (-0.5)
        assert np.isfinite(cov.fom())
        np.testing.assert_allclose(cov.fom(), expected)

    def test_det(self):
        cov = Covariance(np.eye(2) * 4., params=['a', 'b'])
        np.testing.assert_allclose(cov.det(), 16.)


# ── view ──────────────────────────────────────────────────────────────────────

class TestView:
    def test_view_all(self):
        cov, _ = _scalar_cov()
        new = cov.view()
        np.testing.assert_array_equal(new._value, cov._value)

    def test_view_subset(self):
        cov, _ = _scalar_cov()
        sub = cov.view(['omega_m'])
        assert sub.names() == ['omega_m']
        assert sub.shape == (1, 1)
        np.testing.assert_allclose(sub._value[0, 0], 0.01**2)

    def test_view_scalar_nparray(self):
        cov, _ = _scalar_cov()
        val = cov.view('omega_m', return_type='nparray')
        assert val.shape == ()
        np.testing.assert_allclose(float(val), 0.01**2)

    def test_view_unknown_gets_nan(self):
        cov, _ = _scalar_cov()
        sub = cov.view(['omega_m', 'h'])
        assert sub.names() == ['omega_m', 'h']
        assert np.isfinite(sub._value[0, 0])
        assert np.isnan(sub._value[1, 1])
        assert sub._value[0, 1] == 0.
        assert sub._value[1, 0] == 0.

    def test_view_reorder(self):
        cov, _ = _scalar_cov()
        sub = cov.view(['sigma8', 'omega_m'])
        # off-diagonal should still be the same
        np.testing.assert_allclose(sub._value[0, 1], cov._value[1, 0])
        np.testing.assert_allclose(sub._value[1, 0], cov._value[0, 1])

    def test_view_fill_proposal(self):
        cov, _ = _param_cov()
        c_param = Parameter('h', value=0.7, proposal=0.1, fixed=False)
        sub = cov.view([cov._params['omega_m'], c_param], fill='proposal')
        np.testing.assert_allclose(sub._value[1, 1], 0.1**2)

    def test_view_fill_proposal_no_proposal(self):
        cov, _ = _scalar_cov()
        # unknown param given as string → no proposal → stays NaN
        sub = cov.view(['omega_m', 'h'], fill='proposal')
        assert np.isnan(sub._value[1, 1])

    def test_select(self):
        cov, _ = _param_cov()
        sub = cov.select(name='sigma8')
        assert sub.names() == ['sigma8']
        np.testing.assert_allclose(sub.std(), [0.02])


# ── vector parameters ─────────────────────────────────────────────────────────

class TestVectorParams:
    def test_flat_size(self):
        cov = _vector_cov()
        assert _flat_size(cov._params) == 3
        assert _param_sizes(cov._params) == [2, 1]

    def test_std(self):
        cov = _vector_cov()
        np.testing.assert_allclose(cov.std(), [0.01, 0.02, 0.03])

    def test_view_vector_param(self):
        cov = _vector_cov()
        sub = cov.view(['v'])
        assert sub.shape == (2, 2)
        np.testing.assert_allclose(np.diag(sub._value), [0.01**2, 0.02**2])

    def test_view_scalar_param(self):
        cov = _vector_cov()
        sub = cov.view(['s'])
        assert sub.shape == (1, 1)
        np.testing.assert_allclose(sub._value[0, 0], 0.03**2)

    def test_index_vector(self):
        cov = _vector_cov()
        np.testing.assert_array_equal(_index(cov._params, ['v']), [0, 1])
        np.testing.assert_array_equal(_index(cov._params, ['s']), [2])


# ── conversion ────────────────────────────────────────────────────────────────

class TestConversion:
    def test_to_precision_and_back(self):
        cov, _ = _scalar_cov()
        prec = cov.to_precision()
        assert isinstance(prec, Precision)
        assert prec.names() == cov.names()
        cov2 = prec.to_covariance()
        np.testing.assert_allclose(cov._value, cov2._value, rtol=1e-12)

    def test_precision_fom(self):
        cov, _ = _scalar_cov()
        prec = cov.to_precision()
        np.testing.assert_allclose(prec.fom(), cov.fom(), rtol=1e-10)


# ── Precision addition ────────────────────────────────────────────────────────

class TestPrecisionAddition:
    def test_add_same_params(self):
        prec1 = Precision(np.diag([100., 25.]), params=['a', 'b'])
        prec2 = Precision(np.diag([50., 75.]),  params=['a', 'b'])
        total = prec1 + prec2
        np.testing.assert_allclose(np.diag(total._value), [150., 100.])

    def test_add_different_params(self):
        prec1 = Precision(np.diag([100., 25.]),  params=['a', 'b'])
        prec2 = Precision(np.diag([50., 400.]),  params=['b', 'c'])
        total = prec1 + prec2
        assert total.names() == ['a', 'b', 'c']
        np.testing.assert_allclose(np.diag(total._value), [100., 75., 400.])

    def test_sum_classmethod(self):
        prec1 = Precision(np.diag([100.]), params=['a'])
        prec2 = Precision(np.diag([200.]), params=['a'])
        total = Precision.sum(prec1, prec2)
        np.testing.assert_allclose(total._value[0, 0], 300.)

    def test_radd_with_zero(self):
        prec = Precision(np.eye(2), params=['a', 'b'])
        total = sum([prec, prec], start=0)
        np.testing.assert_allclose(np.diag(total._value), [2., 2.])

    def test_sum_list(self):
        prec = Precision(np.diag([100., 25.]), params=['a', 'b'])
        total = Precision.sum([prec, prec])
        np.testing.assert_allclose(np.diag(total._value), [200., 50.])


# ── arithmetic ────────────────────────────────────────────────────────────────

class TestArithmetic:
    def test_mul_scalar(self):
        cov, _ = _scalar_cov()
        cov2 = cov * 4.
        np.testing.assert_allclose(cov2._value, cov._value * 4.)

    def test_rmul_scalar(self):
        cov, _ = _scalar_cov()
        cov2 = 4. * cov
        np.testing.assert_allclose(cov2._value, cov._value * 4.)

    def test_div_scalar(self):
        cov, _ = _scalar_cov()
        cov2 = cov / 2.
        np.testing.assert_allclose(cov2._value, cov._value / 2.)

    def test_original_unchanged(self):
        cov, _ = _scalar_cov()
        original = cov._value.copy()
        _ = cov * 2.
        np.testing.assert_array_equal(cov._value, original)


# ── clone / deepcopy ──────────────────────────────────────────────────────────

class TestClone:
    def test_clone_value(self):
        cov, _ = _scalar_cov()
        cov2 = cov.clone(value=np.eye(2))
        np.testing.assert_array_equal(cov2._value, np.eye(2))
        # original unchanged
        np.testing.assert_array_equal(cov._value[0, 0], 0.01**2)

    def test_clone_params(self):
        cov, _ = _scalar_cov()
        sub = cov.clone(params=['omega_m'])
        assert sub.names() == ['omega_m']

    def test_deepcopy(self):
        cov, _ = _scalar_cov()
        cov2 = cov.deepcopy()
        cov2._value[0, 0] = 999.
        assert cov._value[0, 0] != 999.


# ── I/O ───────────────────────────────────────────────────────────────────────

class TestIO:
    def _roundtrip(self, obj):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as file:
            fn = file.name
        try:
            obj.write(fn)
            from desilike.utils import read
            return read(fn)
        finally:
            os.unlink(fn)

    def test_covariance_hdf5(self):
        cov, _ = _scalar_cov()
        cov2 = self._roundtrip(cov)
        assert isinstance(cov2, Covariance)
        assert cov2.names() == cov.names()
        np.testing.assert_allclose(cov2._value, cov._value)

    def test_precision_hdf5(self):
        cov, _ = _scalar_cov()
        prec = cov.to_precision()
        prec2 = self._roundtrip(prec)
        assert isinstance(prec2, Precision)
        np.testing.assert_allclose(prec2._value, prec._value)

    def test_vector_param_hdf5(self):
        cov = _vector_cov()
        cov2 = self._roundtrip(cov)
        assert cov2.names() == cov.names()
        np.testing.assert_allclose(cov2._value, cov._value)


# ── misc ──────────────────────────────────────────────────────────────────────

class TestMisc:
    def test_contains(self):
        cov, _ = _scalar_cov()
        assert 'omega_m' in cov
        assert 'h' not in cov

    def test_len(self):
        cov, _ = _scalar_cov()
        assert len(cov) == 2

    def test_repr(self):
        cov, _ = _scalar_cov()
        assert 'Covariance' in repr(cov)
        assert 'omega_m' in repr(cov)

    def test_eq(self):
        cov1, _ = _scalar_cov()
        cov2, _ = _scalar_cov()
        assert cov1 == cov2

    def test_array_interface(self):
        cov, _ = _scalar_cov()
        arr = np.asarray(cov)
        np.testing.assert_array_equal(arr, cov._value)
