import numpy as np
import pytest


from desilike.observables.galaxy_clustering.covariance import BaseFootprint, BoxFootprint, CutskyFootprint


class FakeCosmo(object):
    """
    Minimal cosmology stub with r(z) = z, so shell volumes are easy:
    V ~ (area / sr_per_deg2) / 3 * (zmax^3 - zmin^3)
    """

    def comoving_radial_distance(self, z):
        return np.asarray(z)

    def __getstate__(self):
        return {"name": "fake"}

    @classmethod
    def from_state(cls, state):
        return cls()


@pytest.fixture
def fake_cosmo(monkeypatch):
    """
    Patch get_cosmo inside the module under test so CutskyFootprint(cosmo=...)
    accepts our FakeCosmo instance.
    """
    from desilike.observables.galaxy_clustering import covariance

    monkeypatch.setattr(covariance, "get_cosmo", lambda cosmo: cosmo)
    return FakeCosmo()


def test_base_footprint():
    with pytest.raises(ValueError, match='provide either "size".*or "nbar"'):
        BaseFootprint(volume=1000.0)
    with pytest.raises(ValueError, match="provide volume"):
        BaseFootprint(nbar=1e-4)

    footprint = BaseFootprint(nbar=2e-4, volume=5000.0, attrs={"tag": "x"})
    assert np.isclose(footprint.volume, 5000.0)
    assert np.isclose(footprint.size, 1.0)
    assert np.isclose(footprint.shotnoise, 5000.0)
    assert footprint.attrs == {"tag": "x"}

    footprint = BaseFootprint(size=200.0, volume=1000.0)
    assert np.isclose(footprint.size, 200.0)
    assert np.isclose(footprint._nbar, 0.2)
    assert np.isclose(footprint.shotnoise, 5.0)

    footprint = BaseFootprint(nbar=0.1, volume=1000.0)
    p0 = 2.0
    expected = 1000.0 * (0.1 * p0) / (1.0 + 0.1 * p0)
    assert np.isclose(footprint.volume_eff(p0), expected)

    footprint1 = BaseFootprint(nbar=0.1, volume=1000.0)
    footprint2 = BaseFootprint(nbar=0.2, volume=800.0)

    out = footprint1 & footprint2
    assert isinstance(out, BaseFootprint)
    assert np.isclose(out._nbar, 0.3)
    assert np.isclose(out.volume, 800.0)
    assert np.isclose(out.size, 240.0)

    footprint = BoxFootprint(nbar=0.1, volume=1000.0)
    assert isinstance(footprint, BaseFootprint)
    assert np.isclose(footprint.size, 100.0)

    footprint = BaseFootprint(nbar=0.1, volume=1000.0, attrs={"a": 1})
    state = footprint.__getstate__()
    new = BaseFootprint.from_state(state)
    assert np.allclose(new.size, footprint.size)
    assert np.allclose(new.volume, footprint.volume)
    assert new.attrs == {"a": 1}


def test_cutsky_footprint(fake_cosmo):
    with pytest.raises(ValueError, match='provide either "size".*or "nbar"'):
        CutskyFootprint(area=100.0, zrange=(0.1, 0.2), cosmo=fake_cosmo)
    with pytest.raises(ValueError, match="provide area"):
        CutskyFootprint(nbar=1.0, cosmo=fake_cosmo)
    footprint = CutskyFootprint(nbar=1.0, area=100.0, zrange=(0.1, 0.2), cosmo=None)
    with pytest.raises(ValueError, match="Provide cosmology"):
        _ = footprint.cosmo

    area = 1000.0  # deg^2
    zrange = np.array([1.0, 2.0])
    nbar = 3.0  # interpreted as angular density for scalar case
    footprint = CutskyFootprint(nbar=nbar, area=area, zrange=zrange, cosmo=fake_cosmo)
    sr_factor = area / (180.0 / np.pi) ** 2
    expected_volume = sr_factor / 3.0 * (2.0**3 - 1.0**3)
    assert np.isclose(footprint.area, area)
    assert np.isclose(footprint.volume, expected_volume)
    assert np.isclose(footprint.size, area * nbar)
    assert footprint.zlim == (1.0, 2.0)
    assert np.isclose(footprint.z_eff(), 1.5)

    # Mean fractional coverage = 0.5 over full sky
    area_map = np.array([0.0, 1.0, 0.5, 0.5])
    footprint = CutskyFootprint(nbar=1.0, area=area_map, zrange=(0.1, 0.2), cosmo=fake_cosmo)
    expected_area = np.mean(area_map) * (180.0 / np.pi) ** 2 * (4.0 * np.pi)
    assert np.isclose(footprint.area, expected_area)

    # Binned nbar
    area = 1000.0
    zrange = np.array([1.0, 2.0, 4.0])
    nbar = np.array([10.0, 20.0])  # one value per bin
    footprint = CutskyFootprint(nbar=nbar, area=area, zrange=zrange, cosmo=fake_cosmo)
    dvol = np.diff(zrange**3)  # because r(z)=z
    sr_factor = area / (180.0 / np.pi) ** 2 / 3.0
    expected_size = sr_factor * np.sum(nbar * dvol)
    assert np.isclose(footprint.size, expected_size)
    zmid = np.array([1.5, 3.0])
    expected_zeff1 = np.average(zmid, weights=nbar * dvol)
    expected_zeff2 = np.average(zmid, weights=nbar**2 * dvol)
    assert np.isclose(footprint.z_eff(order=1), expected_zeff1)
    assert np.isclose(footprint.z_eff(order=2), expected_zeff2)

    # nbar.size = zrange.size
    area = 1000.0
    zrange = np.array([1.0, 2.0, 4.0])
    nbar = np.array([10.0, 20.0, 40.0])  # one value per z node
    footprint = CutskyFootprint(nbar=nbar, area=area, zrange=zrange, cosmo=fake_cosmo)
    dvol = np.diff(zrange**3)
    nbar_bin = 0.5 * (nbar[:-1] + nbar[1:])
    sr_factor = area / (180.0 / np.pi) ** 2 / 3.0
    expected_size = sr_factor * np.sum(nbar_bin * dvol)
    assert np.isclose(footprint.size, expected_size)
    zmid = np.array([1.5, 3.0])
    expected_zeff = np.average(zmid, weights=nbar_bin * dvol)
    assert np.isclose(footprint.z_eff(), expected_zeff)

    # Effective volume
    area = 1000.0
    zrange = np.array([1.0, 3.0])
    nbar = 2.0
    p0 = 5.0
    footprint = CutskyFootprint(nbar=nbar, area=area, zrange=zrange, cosmo=fake_cosmo)
    shell_volume = area / (180.0 / np.pi) ** 2 / 3.0 * np.diff(zrange**3)
    expected = np.sum(shell_volume * (nbar * p0) / (1.0 + nbar * p0))
    assert np.isclose(footprint.volume_eff(p0), expected)

    # Float area
    footprint1 = CutskyFootprint(nbar=10.0, area=1000.0, zrange=(0.5, 1.5), cosmo=fake_cosmo)
    footprint2 = CutskyFootprint(nbar=20.0, area=800.0, zrange=(1.0, 2.0), cosmo=fake_cosmo)
    out = footprint1 & footprint2
    assert isinstance(out, CutskyFootprint)
    assert np.isclose(out.area, 800.0)
    assert np.allclose(out._zrange, np.array([1.0, 1.5]))
    assert np.isclose(out._nbar, 30.0)

    # Pixelized area
    area1 = np.array([1.0, 0.5, 0.0])
    area2 = np.array([0.5, 0.5, 1.0])
    footprint1 = CutskyFootprint(nbar=1.0, area=area1, zrange=(0.0, 1.0), cosmo=fake_cosmo)
    footprint2 = CutskyFootprint(nbar=2.0, area=area2, zrange=(0.0, 1.0), cosmo=fake_cosmo)
    out = footprint1 & footprint2
    assert np.allclose(out._area, area1 * area2)
    assert np.isclose(out._nbar, 3.0)

    # Both have non-scalar nbar and no explicit size, so interpolation branch is used.
    footprint1 = CutskyFootprint(
        nbar=np.array([10.0, 20.0]),
        area=1000.0,
        zrange=np.array([0.0, 1.0, 2.0]),
        cosmo=fake_cosmo,
    )
    footprint2 = CutskyFootprint(
        nbar=np.array([1.0, 3.0]),
        area=900.0,
        zrange=np.array([0.5, 1.5, 2.5]),
        cosmo=fake_cosmo,
    )
    out = footprint1 & footprint2
    # Common zrange should be unique merged grid restricted to overlap [0.5, 2.0]
    expected_zrange = np.array([0.5, 1.0, 1.5, 2.0])
    assert np.allclose(out._zrange, expected_zrange)
    # Since both inputs are bin-valued, interpolation is on bin centers
    centers1 = np.array([0.5, 1.5])
    centers2 = np.array([1.0, 2.0])
    out_centers = np.array([0.75, 1.25, 1.75])
    expected_nbar = np.interp(out_centers, centers1, [10.0, 20.0])\
        + np.interp(out_centers, centers2, [1.0, 3.0])
    assert np.allclose(out._nbar, expected_nbar)
    assert np.isclose(out.area, 900.0)


def test_covariance():
    from desilike.observables.galaxy_clustering import TracerSpectrum2PolesObservable, ObservablesCovarianceMatrix
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
    import lsstypes as types

    def get_spectrum2_data(size=10):
        edges = np.linspace(0., 0.2, size + 1)
        edges = np.column_stack([edges[:-1], edges[1:]])
        k = np.mean(edges, axis=-1)
        value = np.zeros_like(k)
        ells = [0, 2, 4]
        data = [types.Mesh2SpectrumPole(k=k, num_raw=value, k_edges=edges, ell=ell) for ell in ells]
        return types.Mesh2SpectrumPoles(data)

    def get_spectrum2_window(observable, size=20):
        edges = np.linspace(0., 0.2, size + 1)
        edges = np.column_stack([edges[:-1], edges[1:]])
        k = np.mean(edges, axis=-1)
        ells = [0, 2, 4]
        theory = [types.Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), k_edges=edges, ell=ell) for ell in ells]
        theory = types.ObservableTree(theory, ells=ells, wa_orders=[0] * len(ells))
        window = np.zeros((observable.size, theory.size))
        return types.WindowMatrix(observable=observable, theory=theory, value=window)

    def get_covariance(observable):
        covariance = np.eye(observable.size)
        return types.CovarianceMatrix(observable=observable, value=covariance)

    data = get_spectrum2_data()
    observable = TracerSpectrum2PolesObservable(data=data, window=get_spectrum2_window(data), theory=KaiserTracerPowerSpectrumMultipoles())
    footprint = CutskyFootprint(nbar=np.array([0.001, 0.002]), area=1000.0, zrange=np.array([0.0, 1.0, 2.0]), cosmo='DESI')
    covariance = ObservablesCovarianceMatrix([observable], footprints=[footprint])
    covariance = covariance()
    assert covariance.value().shape == (data.size,) * 2