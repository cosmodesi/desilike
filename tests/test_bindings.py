import numpy as np
import pytest

from desilike.bindings.cobaya.factory import CobayaEngine, BaseExternalEngine


@pytest.fixture
def processed_requires():
    # What CobayaEngine expects AFTER BaseExternalEngine.get_requires(...)
    return {
        "background": {
            "efunc": {"z": np.array([0.0, 0.5, 1.0])},
            "comoving_radial_distance": {"z": np.array([0.2, 1.0])},
            "angular_diameter_distance": {"z": np.array([0.2, 1.0])},
            # These two should be remapped onto angular_diameter_distance
            "comoving_angular_distance": {"z": np.array([0.4])},
            "luminosity_distance": {"z": np.array([0.5])},
        },
        "thermodynamics": {"rs_drag": {}},
        "fourier": {
            # This will be converted into pk_interpolator with non_linear=False
            "sigma8_z": {"z": np.array([0.0, 1.0]), "k": np.array([0.01, 0.1]), "of": [("delta_cb", "theta_cb")]},
            # Already a pk_interpolator block (will be merged)
            "pk_interpolator": {
                "z": np.array([0.0, 1.5]),
                "k": np.linspace(1e-3, 0.5, 6),
                "of": [("delta_m", "delta_m")],
                "non_linear": True,
            },
        },
    }


def test_get_requires_merging(monkeypatch, processed_requires):
    # Patch BaseExternalEngine.get_requires to return our processed mapping
    monkeypatch.setattr(
        BaseExternalEngine,
        "get_requires",
        classmethod(lambda cls, req: processed_requires),
    )

    # Include a 'params' key to force Hubble injection path at the end of get_requires
    out = CobayaEngine.get_requires({"params": {"dummy": 1}})

    # Must have Hubble with z starting at 0
    assert "Hubble" in out
    assert out["Hubble"]["z"][0] == 0.0

    # angular_diameter_distance must be requested (and used to satisfy others)
    assert "angular_diameter_distance" in out

    # Pk_grid must exist and be merged
    pk = out.get("Pk_grid", {})
    assert pk, "Pk_grid was not built"
    assert "z" in pk and "k_max" in pk and "vars_pairs" in pk
    assert pk["k_max"] > 0

    # 'theta_cb' expands into both baryon and cdm velocity fields
    vels = set(tuple(p) for p in pk["vars_pairs"])
    # CB density maps to 'delta_nonu'
    assert ("delta_nonu", "v_newtonian_cdm") in vels
    assert ("delta_nonu", "v_newtonian_baryon") in vels

    # Added coarse z support (0..2) merged in
    assert np.isclose(pk["z"].min(), 0.0)
    assert np.isclose(pk["z"].max(), 2.0)
