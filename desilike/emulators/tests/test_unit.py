import numpy as np
import pytest

from desilike.base import BaseCalculator
from desilike.emulators import (
    Emulator, EmulatedCalculator,
    PointEmulatorEngine, TaylorEmulatorEngine, MLPEmulatorEngine,
    PCAOperation,
)


# ---------------------------------------------------------------------------
# Lightweight calculators – no heavy cosmology, fast to evaluate
# ---------------------------------------------------------------------------

class LinearModel(BaseCalculator):
    """model = a*x + b  (exact under Taylor order=1)"""

    _params = {
        'a': {'value': 1.0, 'ref': {'limits': [0.5, 2.0]}},
        'b': {'value': 0.5, 'ref': {'limits': [-1.0, 1.0]}},
    }

    def initialize(self):
        self.x = np.linspace(0.0, 1.0, 10)

    def calculate(self, a=1.0, b=0.5):
        self.model = a * self.x + b

    def __getstate__(self):
        return {'x': self.x, 'model': self.model}


class QuadraticModel(BaseCalculator):
    """model = a*x^2 + b*x  (exact under Taylor order=2)"""

    _params = {
        'a': {'value': 1.0, 'ref': {'limits': [0.5, 2.0]}},
        'b': {'value': 0.5, 'ref': {'limits': [-1.0, 1.0]}},
    }

    def initialize(self):
        self.x = np.linspace(0.0, 1.0, 10)

    def calculate(self, a=1.0, b=0.5):
        self.model = a * self.x ** 2 + b * self.x

    def __getstate__(self):
        return {'x': self.x, 'model': self.model}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_params(calc):
    return {str(p): p.value for p in calc.varied_params}


def _build_and_fit(engine, calculator_cls=LinearModel, **set_samples_kwargs):
    calc = calculator_cls()
    emu = Emulator(calc, engine=engine)
    emu.set_samples(**set_samples_kwargs)
    emu.fit()
    return emu


# ---------------------------------------------------------------------------
# PointEmulatorEngine
# ---------------------------------------------------------------------------

class TestPointEmulator:

    def test_predict_at_reference(self):
        """Point emulator reproduces the calculator at the reference point."""
        calc = LinearModel()
        calc(**_ref_params(calc))
        ref = calc.model.copy()

        emu = _build_and_fit(PointEmulatorEngine())
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.allclose(emulated.model, ref)

    def test_output_is_finite(self):
        emu = _build_and_fit(PointEmulatorEngine())
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.isfinite(emulated.model).all()

    def test_save_load_roundtrip(self, tmp_path):
        emu = _build_and_fit(PointEmulatorEngine())
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        pred_before = emulated.model.copy()

        fn = str(tmp_path / 'point_emu.npy')
        emu.save(fn)
        reloaded = EmulatedCalculator.load(fn)
        reloaded(**_ref_params(reloaded))
        assert np.allclose(reloaded.model, pred_before)

    def test_emulated_calculator_save_load(self, tmp_path):
        """EmulatedCalculator.save/load is independent of Emulator.save/load."""
        emu = _build_and_fit(PointEmulatorEngine())
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        pred_before = emulated.model.copy()

        fn = str(tmp_path / 'ec.npy')
        emulated.save(fn)
        reloaded = EmulatedCalculator.load(fn)
        reloaded(**_ref_params(reloaded))
        assert np.allclose(reloaded.model, pred_before)

    def test_deepcopy_is_independent(self):
        # Use TaylorEmulatorEngine: PointEmulatorEngine always returns the
        # reference-point value regardless of params (by design), so we need
        # a parameter-sensitive engine to verify independence.
        calc = LinearModel()
        emu = Emulator(calc, engine=TaylorEmulatorEngine(order=1))
        emu.set_samples()
        emu.fit()
        emulated = emu.to_calculator()
        copy = emulated.deepcopy()

        emulated(a=1.0, b=0.5)
        copy(a=1.0, b=0.5)
        assert np.allclose(emulated.model, copy.model)

        emulated(a=1.8, b=-0.8)
        copy(a=0.6, b=0.9)
        assert not np.allclose(emulated.model, copy.model)


# ---------------------------------------------------------------------------
# TaylorEmulatorEngine
# ---------------------------------------------------------------------------

class TestTaylorEmulator:

    def test_linear_exact_at_order1(self):
        """Taylor order=1 must be exact for a linear model everywhere in its range."""
        emu = _build_and_fit(TaylorEmulatorEngine(order=1))
        calc = LinearModel()
        emulated = emu.to_calculator()

        for a, b in [(0.6, -0.8), (1.0, 0.0), (1.9, 0.9)]:
            calc(a=a, b=b)
            emulated(a=a, b=b)
            assert np.allclose(emulated.model, calc.model, atol=1e-10), \
                f"Mismatch at a={a}, b={b}"

    def test_quadratic_exact_at_order2(self):
        """Taylor order=2 must be exact for a quadratic model."""
        emu = _build_and_fit(TaylorEmulatorEngine(order=2), calculator_cls=QuadraticModel)
        calc = QuadraticModel()
        emulated = emu.to_calculator()

        for a, b in [(0.7, 0.2), (1.5, -0.5)]:
            calc(a=a, b=b)
            emulated(a=a, b=b)
            assert np.allclose(emulated.model, calc.model, atol=1e-6), \
                f"Mismatch at a={a}, b={b}"

    def test_output_changes_with_params(self):
        """Emulated output differs when parameters are changed."""
        emu = _build_and_fit(TaylorEmulatorEngine(order=1))
        emulated = emu.to_calculator()

        emulated(**_ref_params(emulated))
        ref = emulated.model.copy()
        emulated(**{str(p): p.value + 0.3 for p in emulated.varied_params})
        assert not np.allclose(emulated.model, ref)

    def test_check_runs(self):
        """Emulator.check() completes without raising."""
        emu = _build_and_fit(TaylorEmulatorEngine(order=1))
        emu.check()

    def test_save_load_preserves_predictions(self, tmp_path):
        emu = _build_and_fit(TaylorEmulatorEngine(order=1))
        emulated = emu.to_calculator()
        emulated(a=0.8, b=0.3)
        pred_before = emulated.model.copy()

        fn = str(tmp_path / 'taylor_emu.npy')
        emu.save(fn)
        reloaded = EmulatedCalculator.load(fn)
        reloaded(a=0.8, b=0.3)
        assert np.allclose(reloaded.model, pred_before)

    def test_finite_difference_method(self):
        """Finite-difference Taylor engine produces finite, non-NaN output."""
        calc = LinearModel()
        emu = Emulator(calc, engine=TaylorEmulatorEngine(order=1, method='finite'))
        emu.set_samples(method='finite')
        emu.fit()
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.isfinite(emulated.model).all()

    @pytest.mark.parametrize('order', [1, 2, 3])
    def test_orders_are_finite(self, order):
        emu = _build_and_fit(TaylorEmulatorEngine(order=order))
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.isfinite(emulated.model).all()


# ---------------------------------------------------------------------------
# MLPEmulatorEngine
# ---------------------------------------------------------------------------

_MLP_XFAIL = pytest.mark.xfail(
    reason="cosmoprimo bug: numpy.vectorize in MLPEmulatorEngine.fit iterates "
           "element-wise over 2D X, passing a scalar to zip(self.params, x). "
           "Fix needed in cosmoprimo/emulators/tools/base.py:558.",
    strict=True,
)


class TestMLPEmulator:

    @_MLP_XFAIL
    def test_fit_produces_finite_output(self):
        calc = LinearModel()
        emu = Emulator(calc, engine=MLPEmulatorEngine(nhidden=()))
        emu.set_samples(niterations=200)
        emu.fit(epochs=100)
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.isfinite(emulated.model).all()

    @_MLP_XFAIL
    def test_check_runs(self):
        calc = LinearModel()
        emu = Emulator(calc, engine=MLPEmulatorEngine(nhidden=()))
        emu.set_samples(niterations=200)
        emu.fit(epochs=100)
        emu.check(frac=0.5)

    @_MLP_XFAIL
    def test_with_pca_operation(self):
        calc = LinearModel()
        emu = Emulator(calc, engine=MLPEmulatorEngine(nhidden=(), yoperation=PCAOperation(npcs=3)))
        emu.set_samples(niterations=200)
        emu.fit(epochs=100)
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.isfinite(emulated.model).all()

    @_MLP_XFAIL
    def test_save_load_roundtrip(self, tmp_path):
        calc = LinearModel()
        emu = Emulator(calc, engine=MLPEmulatorEngine(nhidden=()))
        emu.set_samples(niterations=200)
        emu.fit(epochs=100)
        emulated = emu.to_calculator()
        params = _ref_params(emulated)
        emulated(**params)
        pred_before = emulated.model.copy()

        fn = str(tmp_path / 'mlp_emu.npy')
        emu.save(fn)
        reloaded = EmulatedCalculator.load(fn)
        reloaded(**params)
        assert np.allclose(reloaded.model, pred_before)


# ---------------------------------------------------------------------------
# EmulatedCalculator generic behaviour
# ---------------------------------------------------------------------------

class TestEmulatedCalculator:

    def test_model_shape_matches_original(self):
        calc = LinearModel()
        emu = _build_and_fit(TaylorEmulatorEngine(order=1))
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert emulated.model.shape == calc.x.shape

    def test_model_is_ndarray(self):
        emu = _build_and_fit(TaylorEmulatorEngine(order=1))
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert isinstance(emulated.model, np.ndarray)

    @pytest.mark.parametrize('engine', [
        PointEmulatorEngine(),
        TaylorEmulatorEngine(order=1),
    ])
    def test_consistent_across_engines(self, engine):
        """All engines give a finite model at the reference point."""
        calc = LinearModel()
        calc(**_ref_params(calc))
        ref = calc.model.copy()

        emu = Emulator(LinearModel(), engine=engine)
        emu.set_samples()
        emu.fit()
        emulated = emu.to_calculator()
        emulated(**_ref_params(emulated))
        assert np.isfinite(emulated.model).all()
        assert np.allclose(emulated.model, ref, atol=1e-8)


if __name__ == '__main__':
    from desilike import setup_logging
    setup_logging()
    pytest.main([__file__, '-v'])
