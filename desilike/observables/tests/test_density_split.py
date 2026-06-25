import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from desilike import jax as desilike_jax
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    DensitySplitPowerSpectrumMultipolesObservable,
    density_split_sample_covariance,
    flatten_density_split_power_spectrum_multipoles,
    get_density_split_k,
    load_density_split_mock_matrix,
    load_density_split_power_spectrum_multipoles,
)
from desilike.theories.galaxy_clustering import (
    DensitySplitTracerPowerSpectrumMultipoles,
    FixedPowerSpectrumTemplate,
    ShapeFitPowerSpectrumTemplate,
)


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
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model='tree')
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    assert np.isfinite(likelihood())
    assert observable.flattheory.shape == observable.flatdata.shape


def test_density_split_observable_shapefit_template_params():
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, quantiles=(1,), ells=(0,), rebin=13, kmin=0.01, kmax=0.08)
    template = ShapeFitPowerSpectrumTemplate(z=0.5, fiducial='DESI', apmode='qparqper')
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=template, model='tree')
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=(1,), ells=(0,), rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    assert np.isfinite(likelihood())
    varied = set(likelihood.varied_params.names())
    assert {'qpar', 'qper', 'df', 'dm'} <= varied


def test_density_split_direct_template_params_are_preserved():
    from types import SimpleNamespace

    from scripts.run_density_split_minuit import build_template, select_params

    args = SimpleNamespace(template='direct', template_apmode='qparqper', template_engine='class',
                           template_params=['omega_cdm'], z=0.5, fiducial='DESI')
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, quantiles=(1, 5), ells=(0,),
                                                        rebin=13, kmin=0.01, kmax=0.05)
    template = build_template(args)
    select_params(template, args.template, args.template_params)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=template, model='tree')
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=(1, 5),
                                                               ells=(0,), rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    varied = set(likelihood.varied_params.names())
    assert {'omega_cdm', 'b1p', 'c1q1', 'c1q5'} <= varied
    assert not {'h', 'omega_b', 'logA'} & varied
    assert not {'bq1', 'bq5', 'beta1', 'beta5'} & varied


def test_density_split_one_loop_model_parameters_are_physical():
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, quantiles=(1, 2, 4, 5), ells=(0,),
                                                        rebin=13, kmin=0.01, kmax=0.05)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model='1-loop',
                                                       composite_loop_nq=6, composite_loop_nx=4, composite_loop_nphi=4)
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=(1, 2, 4, 5),
                                                               ells=(0,), rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    assert np.isfinite(likelihood())
    varied = set(likelihood.varied_params.names())
    all_params = {param.basename for param in theory.init.params}
    assert {'c1q1', 'c1q2', 'c1q4', 'c1q5', 'c2q1', 'c2q2', 'c2q4', 'c2q5',
            'e0q1', 'e2q2', 'e4q4', 's0qg1', 's2qg2', 's2muqg4'} <= varied
    assert {'c2q1', 'c2q2', 'c2q4', 'c2q5', 'c3q1', 'c3q2', 'c3q4', 'c3q5'} <= all_params
    assert 'c3q1' not in varied
    assert not any(name.startswith('op') for name in varied)
    assert not {'bq1', 'beta1', 'c01', 'bqnabla1'} & varied


@pytest.mark.parametrize('model', ['tree', '1-loop'])
def test_density_split_linear_parameters_can_be_marginalized(model):
    data = load_density_split_power_spectrum_multipoles(DATA_PATH, quantiles=(1, 2, 4, 5), ells=(0,),
                                                        rebin=13, kmin=0.01, kmax=0.04)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model=model,
                                                       composite_loop_nq=4, composite_loop_nx=4, composite_loop_nphi=4)
    solved_basenames = ['c1q*', 's0qg*'] if model == 'tree' else ['c1q*', 'c2q*', 'e0q*', 'e2q*', 'e4q*',
                                                                  's0qg*', 's2qg*', 's2muqg*']
    for param in theory.init.params.select(basename=solved_basenames):
        param.update(derived='.auto')
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=(1, 2, 4, 5),
                                                               ells=(0,), rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=np.eye(observable.flatdata.size))

    assert np.isfinite(likelihood())
    solved = set(likelihood.all_params.select(solved=True).names())
    varied = set(likelihood.varied_params.names())
    assert {'c1q1', 'c1q2', 'c1q4', 'c1q5'} <= solved
    assert {'s0qg1', 's0qg2', 's0qg4', 's0qg5'} <= solved
    if model == '1-loop':
        assert {'c2q1', 'e0q1', 'e2q1', 'e4q1', 's2qg1', 's2muqg1'} <= solved
    assert not solved & varied
    assert not {'alpha0p', 'alpha2p', 'alpha4p', 'c3q1'} & solved


@pytest.mark.parametrize('model', ['tree', '1-loop'])
def test_density_split_partition_and_subselection(model):
    k = np.linspace(0.01, 0.05, 4)
    full = DensitySplitTracerPowerSpectrumMultipoles(k=k, ells=(0,), quantiles=(1, 2, 3, 4, 5),
                                                     template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'), model=model,
                                                     composite_loop_nq=6, composite_loop_nx=4, composite_loop_nphi=4)
    subset = DensitySplitTracerPowerSpectrumMultipoles(k=k, ells=(0,), quantiles=(1, 2, 4, 5),
                                                       template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model=model,
                                                       composite_loop_nq=6, composite_loop_nx=4, composite_loop_nphi=4)

    full()
    subset()
    full_power = np.asarray(full.power)
    subset_power = np.asarray(subset.power)
    summed = np.mean(full_power, axis=0)
    scale = max(1., np.max(np.abs(full_power)))

    assert np.max(np.abs(summed)) < 1e-10 * scale
    assert np.allclose(subset_power, full_power[[0, 1, 3, 4]], rtol=1e-10, atol=1e-10 * scale)


def test_density_split_tree_matches_strict_kaiser_limit():
    from desilike.theories.galaxy_clustering.density_split import _smoothing_k, _smoothing_window

    k = np.linspace(0.01, 0.05, 4)
    theory = DensitySplitTracerPowerSpectrumMultipoles(k=k, ells=(0, 2), quantiles=(1,),
                                                       template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model='tree')
    power = theory(c1q1=1.3, b1p=1.0)
    b1, _ = theory._folps_pars({'b1p': 1.0})
    f = theory.pt.fsigma8 / theory.pt.sigma8
    mu2 = theory.pt.pt.muap**2
    window = _smoothing_window(_smoothing_k(theory.k, theory.pt.pt.kap, apmode=theory.smoothing_apmode),
                               theory.smoothing_radius, kernel=theory.smoothing_kernel)
    expected_pkmu = 1.3 * window * (b1 + f * mu2)**2 * theory._linear_matter_pk()
    expected = theory.to_poles(theory.pt.pt.jac * expected_pkmu)

    assert np.allclose(np.asarray(power[0]), np.asarray(expected), rtol=1e-12, atol=1e-12)


def test_density_split_tree_does_not_call_one_loop_or_c2():
    def fail(*args, **kwargs):
        raise AssertionError('tree model should not call one-loop helpers')

    theory = DensitySplitTracerPowerSpectrumMultipoles(k=np.linspace(0.01, 0.05, 4), ells=(0,), quantiles=(1,),
                                                       template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model='tree')
    theory._folps_pkmu = fail
    theory._composite_p2_moments = fail
    power = theory()

    assert np.isfinite(np.asarray(power)).all()


@pytest.mark.skipif(desilike_jax.jax is None, reason='jax is not available')
@pytest.mark.parametrize('kmax', [0.12, 0.15, 0.2])
def test_density_split_one_loop_finite_c2_outputs(kmax):
    k = np.linspace(0.01, kmax, 4)
    theory = DensitySplitTracerPowerSpectrumMultipoles(k=k, ells=(0, 2), quantiles=(1, 2, 4, 5),
                                                       template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model='1-loop',
                                                       composite_loop_nq=6, composite_loop_nx=4, composite_loop_nphi=4)
    theory()

    assert np.isfinite(np.asarray(theory.power)).all()
    assert getattr(theory, '_get_composite_p2_moments', None) is not None
    assert not hasattr(theory, '_composite_p2_cache')


@pytest.mark.skipif(desilike_jax.jax is None, reason='jax is not available')
def test_density_split_one_loop_c2_quadrature_stability():
    from desilike.theories.galaxy_clustering.density_split import composite_p2_moments, contract_p2_moments

    kt = np.logspace(-4, 1, 128)
    pklin = 1e4 * kt / (1. + (kt / 0.2)**2)
    k = np.asarray([[0.05, 0.1]])
    mu = np.asarray([[0.2, 0.8]])
    coarse = contract_p2_moments(composite_p2_moments(k, mu, kt, pklin, 0.75, nq=12, nx=8, nphi=8, qmax=1.),
                                 b1=2., b2=0.5, bs=-0.3)
    finer = contract_p2_moments(composite_p2_moments(k, mu, kt, pklin, 0.75, nq=20, nx=12, nphi=12, qmax=1.),
                                b1=2., b2=0.5, bs=-0.3)

    assert np.allclose(np.asarray(coarse), np.asarray(finer), rtol=0.25, atol=1e-8)


def test_density_split_observable_plot(tmp_path):
    from matplotlib import pyplot as plt

    data = load_density_split_power_spectrum_multipoles(DATA_PATH, rebin=13, kmin=0.01, kmax=0.2)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'),
                                                       model='tree')
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
    assert 'Fit density-split multipoles' in result.stdout
    assert '--plot-output' in result.stdout
    assert '--theory-model' in result.stdout
    assert '--template' in result.stdout
    assert '--template-apmode' in result.stdout
    assert '--template-params' in result.stdout
    assert '--emulator' in result.stdout
    assert '--smoothing-apmode' in result.stdout
    assert '--composite-loop-resolution' in result.stdout
    assert '--sample-ds-linear' in result.stdout
    assert '--qg-anisotropic-stochastic' in result.stdout
    assert 'tree' in result.stdout
    assert '1-loop' in result.stdout
    assert '--ds-linear-basis' not in result.stdout
    assert '--ds-relax-prior-scale' not in result.stdout
    assert '--composite-pieces' not in result.stdout
    assert '--fix-composite-stochastic' not in result.stdout
    assert '--operator-prior-scale' not in result.stdout
    assert 'folps_composite_qg_1loop' not in result.stdout


def test_density_split_emcee_script_help():
    from scripts.run_density_split_emcee import chain_filenames

    script = Path(__file__).resolve().parents[3] / 'scripts' / 'run_density_split_emcee.py'
    result = subprocess.run([sys.executable, str(script), '--help'], check=True, capture_output=True, text=True)
    assert 'Sample density-split multipoles' in result.stdout
    assert '--chains' in result.stdout
    assert '--nwalkers' in result.stdout
    assert '--max-eigen-gr' in result.stdout
    assert '--smoothing-apmode' in result.stdout
    assert '--composite-loop-resolution' in result.stdout
    assert '--sample-ds-linear' in result.stdout
    assert '--qg-anisotropic-stochastic' in result.stdout
    assert '--ds-linear-basis' not in result.stdout
    assert '--ds-relax-prior-scale' not in result.stdout
    assert '--operator-prior-scale' not in result.stdout
    assert chain_filenames('/tmp/chain_*.npy', 2) == ['/tmp/chain_0.npy', '/tmp/chain_1.npy']
    assert chain_filenames('/tmp/chain.npy', 2) == ['/tmp/chain_0.npy', '/tmp/chain_1.npy']
