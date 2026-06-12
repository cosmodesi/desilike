import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    DensitySplitPowerSpectrumMultipolesObservable,
    density_split_sample_covariance,
    flatten_density_split_power_spectrum_multipoles,
    get_density_split_k,
    load_density_split_mock_matrix,
    load_density_split_power_spectrum_multipoles,
)
from desilike.theories.galaxy_clustering import DensitySplitTracerPowerSpectrumMultipoles, FixedPowerSpectrumTemplate


JAXPT_ROOT = Path('/Users/epaillas/code/jax-pt')
DATA_PATH = JAXPT_ROOT / 'data' / 'data_vector' / 'dsc_pkqg_poles_c000_hod006.h5'
MOCK_DIR = JAXPT_ROOT / 'data' / 'for_covariance'

pytestmark = pytest.mark.skipif(not DATA_PATH.exists() or not MOCK_DIR.exists(), reason='raw jax-pt density-split files are not available')


def test_density_split_raw_data_loading():
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, rebin=13, kmin=0.01, kmax=0.2)
    k = get_density_split_k(data)
    flatdata = flatten_density_split_power_spectrum_multipoles(data)

    assert k.shape == (14,)
    assert np.all(np.diff(k) > 0.)
    assert flatdata.shape == (5 * 3 * k.size,)
    assert np.isfinite(flatdata).all()

    subset = load_density_split_power_spectrum_multipoles(DATA_PATH, quantiles=(1, 3, 5), ells=(0, 4), rebin=13, kmin=0.01, kmax=0.2)
    assert flatten_density_split_power_spectrum_multipoles(subset).shape == (3 * 2 * k.size,)


def test_density_split_raw_covariance_from_mock_subset():
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, rebin=13, kmin=0.01, kmax=0.2)
    k = get_density_split_k(data)

    k_mocks, mock_matrix = load_density_split_mock_matrix(MOCK_DIR, k=k, rebin=13, kmin=0.01, kmax=0.2, max_mocks=6)
    covariance = density_split_sample_covariance(MOCK_DIR, k=k, rebin=13, kmin=0.01, kmax=0.2, max_mocks=6)

    assert np.allclose(k_mocks, k)
    assert mock_matrix.shape == (6, 5 * 3 * k.size)
    assert covariance.shape == (mock_matrix.shape[1], mock_matrix.shape[1])
    assert np.isfinite(covariance).all()
    assert np.allclose(covariance, covariance.T)


def test_density_split_observable_likelihood_runs():
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, rebin=13, kmin=0.01, kmax=0.2)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'))
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    assert np.isfinite(likelihood())
    assert observable.flattheory.shape == observable.flatdata.shape


def test_density_split_observable_plot(tmp_path):
    from matplotlib import pyplot as plt

    data = load_density_split_power_spectrum_multipoles(DATA_PATH, rebin=13, kmin=0.01, kmax=0.2)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'))
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    assert np.isfinite(likelihood())
    fig = observable.plot()
    fn = tmp_path / 'density_split_plot.png'
    fig.savefig(fn)
    plt.close(fig)

    assert fn.exists()


def test_density_split_minuit_script_help():
    script = Path(__file__).resolve().parents[3] / 'scripts' / 'run_density_split_minuit.py'
    result = subprocess.run([sys.executable, str(script), '--help'], check=True, capture_output=True, text=True)
    assert 'Fit tree-level density-split multipoles' in result.stdout
    assert '--plot-output' in result.stdout
