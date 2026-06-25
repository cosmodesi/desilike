from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def test_density_split_script_help_exposes_on_the_fly_measurements():
    scripts_dir = Path(__file__).resolve().parents[3] / 'scripts'
    for script_name, description in [
        ('run_density_split_minuit.py', 'Fit density-split multipoles'),
        ('run_density_split_emcee.py', 'Sample density-split multipoles'),
    ]:
        result = subprocess.run([sys.executable, str(scripts_dir / script_name), '--help'],
                                check=True, capture_output=True, text=True)
        assert description in result.stdout
        assert '--on-the-fly-measurements' in result.stdout
        assert '--measurement-save-dir' in result.stdout
        assert '--measurement-seed' in result.stdout
        assert '--measurement-no-rsd' in result.stdout
        assert '--fiducial-cosmology' in result.stdout
        assert '--sample-ds-linear' in result.stdout
        assert '--qg-anisotropic-stochastic' in result.stdout


def test_resolve_density_split_data_uses_static_data_by_default(tmp_path):
    from scripts.run_density_split_minuit import resolve_density_split_data

    data = tmp_path / 'precomputed.h5'
    args = SimpleNamespace(on_the_fly_measurements=False, data=data)

    assert resolve_density_split_data(args) == data


def test_resolve_density_split_data_uses_on_the_fly_measurement(monkeypatch, tmp_path):
    from scripts import run_density_split_minuit as minuit

    measured = tmp_path / 'fresh_pkqg.h5'
    calls = []

    def fake_measure_mockfactory_acm(**kwargs):
        calls.append(kwargs)
        return {'pkqg': measured}

    monkeypatch.setattr(minuit, 'measure_mockfactory_acm', fake_measure_mockfactory_acm)
    args = SimpleNamespace(
        on_the_fly_measurements=True,
        data=tmp_path / 'ignored.h5',
        seed=11,
        measurement_seed=None,
        measurement_no_rsd=False,
        measurement_bias=2.0,
        measurement_redshift=0.5,
        fiducial_cosmology='abacus-c001',
        measurement_boxsize=500.0,
        measurement_nbar=1e-3,
        measurement_nmesh=128,
        measurement_meshsize=[64, 64, 128],
        measurement_cellsize=3.9,
        smoothing_radius=12.0,
        measurement_los='z',
        ells=[0, 2],
        measurement_k_step=0.002,
        measurement_save_dir=tmp_path,
        measurement_overwrite=True,
    )

    assert minuit.resolve_density_split_data(args) == measured
    assert len(calls) == 1
    call = calls[0]
    assert np.array_equal(call.pop('meshsize'), np.array([64, 64, 128]))
    assert call == {
        'bias': 2.0,
        'redshift': 0.5,
        'boxsize': 500.0,
        'nbar': 1e-3,
        'nmesh': 128,
        'seed': 11,
        'fiducial_cosmology': 'abacus-c001',
        'cellsize': 3.9,
        'smoothing_radius': 12.0,
        'nquantiles': 5,
        'los': 'z',
        'ells': (0, 2),
        'k_step': 0.002,
        'save_dir': tmp_path,
        'overwrite': True,
        'rsd': True,
    }


def test_resolve_density_split_data_rejects_low_on_the_fly_resolution(monkeypatch, tmp_path):
    from scripts import run_density_split_minuit as minuit

    monkeypatch.setattr(minuit, 'measure_mockfactory_acm',
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError('measurement should not run')))
    args = SimpleNamespace(
        on_the_fly_measurements=True,
        data=tmp_path / 'ignored.h5',
        seed=11,
        measurement_seed=None,
        measurement_no_rsd=False,
        measurement_bias=2.0,
        measurement_redshift=0.5,
        fiducial_cosmology='desi',
        measurement_boxsize=2000.0,
        measurement_nbar=1e-3,
        measurement_nmesh=128,
        measurement_meshsize=[128],
        measurement_cellsize=3.9,
        smoothing_radius=12.0,
        measurement_los='z',
        ells=[0, 2],
        kmax=0.2,
        measurement_k_step=0.002,
        measurement_save_dir=tmp_path,
        measurement_overwrite=True,
    )

    with np.testing.assert_raises_regex(ValueError, 'k_Nyquist'):
        minuit.resolve_density_split_data(args)


def test_density_split_emulator_validation_rejects_stale_output_shape():
    from scripts.run_density_split_minuit import validate_observable_prediction

    class DummyObservable:
        quantiles = (1, 2)
        ells = (0,)
        k = np.arange(2, dtype='f8')
        flatdata = np.zeros(4, dtype='f8')

        def __call__(self):
            self.flattheory = np.zeros(5, dtype='f8')

    with np.testing.assert_raises_regex(ValueError, 'different k-grid'):
        validate_observable_prediction(DummyObservable(), context='stale emulator')


def test_density_split_minuit_builds_one_loop_with_jax_backend():
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate
    from scripts.run_density_split_minuit import COMPOSITE_LOOP_RESOLUTIONS, build_theory

    args = SimpleNamespace(
        theory_model='1-loop',
        smoothing_radius=10.,
        smoothing_kernel='gaussian',
        smoothing_apmode='observed',
        composite_loop_resolution='smoke',
        prior_basis='physical_aap',
        sample_ds_linear=False,
        qg_anisotropic_stochastic=False,
    )
    theory = build_theory(args, FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'))

    assert theory.options['backend'] == 'jax'
    assert theory.model == '1-loop'
    assert theory.options['composite_loop_nq'] == COMPOSITE_LOOP_RESOLUTIONS['smoke']['composite_loop_nq']


def test_density_split_minuit_marginalizes_ds_linear_block_by_default():
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate
    from scripts.run_density_split_minuit import build_theory

    base_args = dict(smoothing_radius=10., smoothing_kernel='gaussian', smoothing_apmode='observed',
                     composite_loop_resolution='smoke', prior_basis='physical_aap', sample_ds_linear=False,
                     qg_anisotropic_stochastic=False)
    expected = {
        'tree': {'c1q1', 'c1q2', 'c1q4', 'c1q5',
                 's0qg1', 's0qg2', 's0qg4', 's0qg5'},
        '1-loop': {
            'c1q1', 'c1q2', 'c1q4', 'c1q5',
            'c2q1', 'c2q2', 'c2q4', 'c2q5',
            'e0q1', 'e0q2', 'e0q4', 'e0q5',
            'e2q1', 'e2q2', 'e2q4', 'e2q5',
            'e4q1', 'e4q2', 'e4q4', 'e4q5',
            's0qg1', 's0qg2', 's0qg4', 's0qg5',
            's2qg1', 's2qg2', 's2qg4', 's2qg5',
            's2muqg1', 's2muqg2', 's2muqg4', 's2muqg5',
        },
    }
    for model, expected_solved in expected.items():
        args = SimpleNamespace(theory_model=model, **base_args)
        theory = build_theory(args, FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'))
        theory.runtime_info.initialize()
        solved = set(theory.runtime_info.params.select(solved=True).names())

        assert solved == expected_solved
        assert not any(name.startswith('s0muqg') for name in solved)
        assert not {'alpha0p', 'alpha2p', 'alpha4p', 'c3q1', 'c3q2', 'c3q4', 'c3q5'} & solved

    args = SimpleNamespace(theory_model='1-loop', **(base_args | {'qg_anisotropic_stochastic': True}))
    theory = build_theory(args, FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'))
    theory.runtime_info.initialize()
    solved = set(theory.runtime_info.params.select(solved=True).names())
    assert {'s0muqg1', 's0muqg2', 's0muqg4', 's0muqg5'} <= solved


def test_density_split_minuit_can_sample_ds_linear_block():
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate
    from scripts.run_density_split_minuit import build_theory

    args = SimpleNamespace(
        theory_model='1-loop',
        smoothing_radius=10.,
        smoothing_kernel='gaussian',
        smoothing_apmode='observed',
        composite_loop_resolution='smoke',
        prior_basis='physical_aap',
        sample_ds_linear=True,
        qg_anisotropic_stochastic=True,
    )
    theory = build_theory(args, FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI'))
    theory.runtime_info.initialize()

    solved = set(theory.runtime_info.params.select(solved=True).names())
    varied = set(theory.runtime_info.params.select(varied=True).names())
    assert not solved
    assert {'c1q1', 'c2q1', 'e0q1', 'e2q1', 'e4q1',
            's0qg1', 's0muqg1', 's2qg1', 's2muqg1'} <= varied


def test_emcee_reuses_minuit_density_split_data_resolver():
    from scripts import run_density_split_emcee as emcee
    from scripts import run_density_split_minuit as minuit

    assert emcee.resolve_density_split_data is minuit.resolve_density_split_data


def test_measure_mockfactory_acm_normalizes_fiducial_cosmology_names():
    from scripts import measure_mockfactory_acm as measurement

    assert measurement.normalize_fiducial_cosmology_name('DESI') == 'desi'
    assert measurement.normalize_fiducial_cosmology_name('abacus-c1') == 'abacus-c001'
    assert measurement.normalize_fiducial_cosmology_name('abacus_004') == 'abacus-c004'
    assert measurement.normalize_fiducial_cosmology_name('AbacusSummit-c025') == 'abacus-c025'
    with np.testing.assert_raises_regex(ValueError, 'Unknown fiducial cosmology'):
        measurement.normalize_fiducial_cosmology_name('planck')


def test_measure_mockfactory_acm_fiducial_cosmology_paths_are_distinct(tmp_path):
    from scripts import measure_mockfactory_acm as measurement

    desi_paths = measurement.output_paths(tmp_path, seed=5, fiducial_cosmology='desi')
    abacus_paths = measurement.output_paths(tmp_path, seed=5, fiducial_cosmology='abacus-c001')

    assert desi_paths['spectrum'].name == 'mesh2_spectrum_poles_mockfactory_seed5.h5'
    assert desi_paths['metadata'].name == 'mockfactory_seed5_metadata.json'
    assert abacus_paths['spectrum'].name == 'mesh2_spectrum_poles_mockfactory_seed5_abacus-c001.h5'
    assert abacus_paths['pkqg'].name == 'dsc_pkqg_poles_mockfactory_seed5_abacus-c001.h5'
    assert abacus_paths['metadata'].name == 'mockfactory_seed5_abacus-c001_metadata.json'


def test_measure_mockfactory_acm_reuses_existing_galaxy_spectrum(monkeypatch, tmp_path, caplog):
    from scripts import measure_mockfactory_acm as measurement

    paths = measurement.output_paths(tmp_path, seed=5)
    paths['spectrum'].parent.mkdir(parents=True)
    paths['spectrum'].touch()
    monkeypatch.setattr(measurement, 'generate_mock',
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError('mock generation should not run')))

    with caplog.at_level('INFO'):
        returned = measurement.measure_mockfactory_acm(save_dir=tmp_path, seed=5, measure_density_split=False)

    assert returned['spectrum'] == paths['spectrum']
    assert returned['metadata'] == paths['metadata']
    assert 'Using existing on-the-fly measurements' in caplog.text


def test_measure_mockfactory_acm_reuses_existing_density_split_outputs(monkeypatch, tmp_path, caplog):
    from scripts import measure_mockfactory_acm as measurement

    paths = measurement.output_paths(tmp_path, seed=6)
    for key in ['spectrum', 'pkqg', 'pkqq']:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].touch()
    monkeypatch.setattr(measurement, 'generate_mock',
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError('mock generation should not run')))

    with caplog.at_level('INFO'):
        returned = measurement.measure_mockfactory_acm(save_dir=tmp_path, seed=6, measure_density_split=True)

    assert returned['spectrum'] == paths['spectrum']
    assert returned['pkqg'] == paths['pkqg']
    assert returned['pkqq'] == paths['pkqq']
    assert 'Using existing on-the-fly measurements' in caplog.text


def test_measure_mockfactory_acm_rejects_partial_density_split_cache(monkeypatch, tmp_path):
    from scripts import measure_mockfactory_acm as measurement

    paths = measurement.output_paths(tmp_path, seed=7)
    paths['spectrum'].parent.mkdir(parents=True)
    paths['spectrum'].touch()
    monkeypatch.setattr(measurement, 'generate_mock',
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError('mock generation should not run')))

    with np.testing.assert_raises_regex(FileExistsError, 'Partial on-the-fly measurement cache'):
        measurement.measure_mockfactory_acm(save_dir=tmp_path, seed=7, measure_density_split=True)


def test_measure_mockfactory_acm_runs_when_outputs_are_missing(monkeypatch, tmp_path):
    from scripts import measure_mockfactory_acm as measurement

    calls = []

    def fake_generate_mock(**kwargs):
        calls.append(('generate', kwargs))
        return np.zeros((3, 3), dtype='f8')

    def fake_compute_power_spectrum(*args, output_fn, **kwargs):
        calls.append(('power', kwargs))
        output_fn.touch()

    monkeypatch.setattr(measurement, 'generate_mock', fake_generate_mock)
    monkeypatch.setattr(measurement, 'compute_power_spectrum', fake_compute_power_spectrum)

    returned = measurement.measure_mockfactory_acm(save_dir=tmp_path, seed=8, measure_density_split=False,
                                                   rsd=False, overwrite=False)

    assert returned['spectrum'].exists()
    assert [call[0] for call in calls] == ['generate', 'power']


def test_resolve_galaxy_power_data_uses_static_data_by_default(tmp_path):
    from scripts.run_galaxy_power_minuit import resolve_galaxy_power_data

    data = tmp_path / 'precomputed_power.h5'
    args = SimpleNamespace(on_the_fly_measurements=False, data=data)

    assert resolve_galaxy_power_data(args) == data


def test_resolve_galaxy_power_data_uses_on_the_fly_spectrum(monkeypatch, tmp_path):
    from scripts import run_galaxy_power_minuit as power

    measured = tmp_path / 'fresh_spectrum.h5'
    calls = []

    def fake_measure_mockfactory_acm(**kwargs):
        calls.append(kwargs)
        return {'spectrum': measured, 'pkqg': tmp_path / 'unused_pkqg.h5'}

    monkeypatch.setattr(power, 'measure_mockfactory_acm', fake_measure_mockfactory_acm)
    args = SimpleNamespace(
        on_the_fly_measurements=True,
        data=tmp_path / 'ignored_power.h5',
        seed=13,
        measurement_seed=None,
        measurement_no_rsd=False,
        measurement_bias=2.1,
        measurement_redshift=0.6,
        fiducial_cosmology='abacus-c001',
        measurement_boxsize=2000.0,
        measurement_nbar=5e-4,
        measurement_nmesh=256,
        measurement_meshsize=[512],
        measurement_los='x',
        ells=[0, 2, 4],
        kmax=0.2,
        measurement_k_step=0.001,
        measurement_save_dir=tmp_path,
        measurement_overwrite=True,
    )

    assert power.resolve_galaxy_power_data(args) == measured
    assert len(calls) == 1
    call = calls[0]
    assert call.pop('meshsize') == 512
    assert call == {
        'bias': 2.1,
        'redshift': 0.6,
        'boxsize': 2000.0,
        'nbar': 5e-4,
        'nmesh': 256,
        'seed': 13,
        'fiducial_cosmology': 'abacus-c001',
        'los': 'x',
        'ells': (0, 2, 4),
        'k_step': 0.001,
        'save_dir': tmp_path,
        'overwrite': True,
        'rsd': True,
        'measure_density_split': False,
    }


def test_resolve_galaxy_power_data_rejects_low_on_the_fly_resolution(monkeypatch, tmp_path):
    from scripts import run_galaxy_power_minuit as power

    monkeypatch.setattr(power, 'measure_mockfactory_acm',
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError('measurement should not run')))
    args = SimpleNamespace(
        on_the_fly_measurements=True,
        data=tmp_path / 'ignored_power.h5',
        seed=13,
        measurement_seed=None,
        measurement_no_rsd=False,
        measurement_bias=2.1,
        measurement_redshift=0.6,
        fiducial_cosmology='desi',
        measurement_boxsize=2000.0,
        measurement_nbar=5e-4,
        measurement_nmesh=128,
        measurement_meshsize=[128],
        measurement_los='x',
        ells=[0, 2, 4],
        kmax=0.2,
        measurement_k_step=0.001,
        measurement_save_dir=tmp_path,
        measurement_overwrite=True,
    )

    with np.testing.assert_raises_regex(ValueError, 'k_Nyquist'):
        power.resolve_galaxy_power_data(args)
