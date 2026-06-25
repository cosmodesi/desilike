from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import numpy as np

from desilike import setup_logging
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    DensitySplitPowerSpectrumMultipolesObservable,
    density_split_sample_covariance,
    get_density_split_k,
    load_density_split_power_spectrum_multipoles,
)
from desilike.profilers import MinuitProfiler
from desilike.theories.galaxy_clustering import (
    DensitySplitTracerPowerSpectrumMultipoles,
    DirectPowerSpectrumTemplate,
    FixedPowerSpectrumTemplate,
    ShapeFitPowerSpectrumTemplate,
    StandardPowerSpectrumTemplate,
)
from scripts.measure_mockfactory_acm import measure_mockfactory_acm, parse_meshsize


JAXPT_ROOT = Path('/Users/epaillas/code/jax-pt')
DEFAULT_DATA = JAXPT_ROOT / 'data' / 'data_vector' / 'dsc_pkqg_poles_c000_hod006.h5'
DEFAULT_MOCKS = JAXPT_ROOT / 'data' / 'for_covariance'
FOLPS_DENSITY_SPLIT_MODELS = ('tree', '1-loop')
FOLPS_MODEL = 'folps'
COMPOSITE_LOOP_RESOLUTIONS = {
    'smoke': dict(composite_loop_nq=12, composite_loop_nx=8, composite_loop_nphi=8),
    'default': dict(composite_loop_nq=80, composite_loop_nx=32, composite_loop_nphi=16),
    'high': dict(composite_loop_nq=120, composite_loop_nx=48, composite_loop_nphi=24),
}

TEMPLATE_PARAMETER_BASENAMES = {
    'fixed': (),
    'standard': ('qpar', 'qper', 'qiso', 'qap', 'df'),
    'shapefit': ('qpar', 'qper', 'qiso', 'qap', 'df', 'dm', 'dn'),
    'direct': (
        'h', 'H0', 'theta_MC_100',
        'omega_cdm', 'Omega_cdm', 'omega_b', 'Omega_b', 'Omega_m',
        'logA', 'A_s', 'n_s', 'tau_reio', 'm_ncdm', 'N_eff',
        'w0_fld', 'wa_fld', 'Omega_k',
    ),
}


def build_template(args):
    """Return the desilike power-spectrum template requested by the CLI."""
    if args.template == 'fixed':
        return FixedPowerSpectrumTemplate(z=args.z, fiducial=args.fiducial)
    if args.template == 'standard':
        return StandardPowerSpectrumTemplate(z=args.z, fiducial=args.fiducial, apmode=args.template_apmode)
    if args.template == 'shapefit':
        return ShapeFitPowerSpectrumTemplate(z=args.z, fiducial=args.fiducial, apmode=args.template_apmode)
    if args.template == 'direct':
        return DirectPowerSpectrumTemplate(z=args.z, fiducial=args.fiducial, engine=args.template_engine)
    raise ValueError(f'Unknown template {args.template!r}.')


def select_params(calculator, template, params=None):
    """Restrict varied template/cosmology parameters before building the theory."""
    if not params:
        return

    allowed = set(TEMPLATE_PARAMETER_BASENAMES[template])
    requested = set(params)
    unknown = sorted(requested - allowed)
    if unknown:
        raise ValueError(f'Unknown {template!r} template parameter(s): {", ".join(unknown)}.')

    present = {param.basename for param in calculator.init.params if param.basename in allowed and not param.derived}
    missing = sorted(requested - present)
    if missing:
        available = ', '.join(sorted(present)) or 'none'
        raise ValueError(
            f'Requested template parameter(s) not available for template={template!r}: {", ".join(missing)}. '
            f'Available non-derived template parameters are: {available}.'
        )

    for param in calculator.init.params:
        if param.basename in allowed and not param.derived:
            param.update(fixed=param.basename not in requested)


def _emulator_path(fn):
    fn = Path(fn)
    if fn.exists() or fn.suffix == '.npy':
        return fn
    return Path(str(fn) + '.npy')


def check_emulator(theory, emulator):
    classes = getattr(emulator.emulator, 'calculator__class__', ())
    is_folps_pt = any(str(name).endswith('FOLPSv2PowerSpectrumMultipoles') for name in classes)
    if not is_folps_pt:
        found = ', '.join(str(name) for name in classes) or type(emulator).__name__
        raise ValueError(
            'FOLPS density-split emulation expects a FOLPS PT emulator trained from theory.pt. '
            'The loaded emulator was trained from {}.'.format(found)
        )

    expected = theory.pt.varied_params.names()
    if not expected:
        raise ValueError('FOLPS PT emulation requires at least one varied template/cosmology parameter.')
    found = emulator.varied_params.names()
    if found != expected:
        raise ValueError(
            f'Loaded FOLPS PT emulator varied parameters do not match the current theory: '
            f'found {found}, expected {expected}.'
        )


def maybe_emulate_pt(theory, emulator_fn=None):
    """Load or train a Taylor emulator for the FOLPS PT calculator."""
    if emulator_fn is None:
        return None

    emulator_fn = _emulator_path(emulator_fn)
    if emulator_fn.exists():
        emulated_pt = EmulatedCalculator.load(str(emulator_fn))
        check_emulator(theory, emulated_pt)
        print(f'emulator: loaded {emulator_fn}')
        print('emulator_type: folps_pt')
        print(f'emulator_varied_parameters: {", ".join(emulated_pt.varied_params.names())}')
    else:
        if not theory.pt.varied_params:
            raise ValueError('FOLPS PT emulation requires at least one varied template/cosmology parameter.')
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=3))
        emulator.set_samples()
        emulator.fit()
        emulator_fn.parent.mkdir(parents=True, exist_ok=True)
        emulated_pt = emulator.to_calculator()
        emulated_pt.save(str(emulator_fn))
        check_emulator(theory, emulated_pt)
        print(f'emulator: trained and saved {emulator_fn}')
        print('emulator_type: folps_pt')
        print(f'emulator_varied_parameters: {", ".join(emulated_pt.varied_params.names())}')

    theory_params = theory.init.params.deepcopy()
    theory.init.update(pt=emulated_pt)
    theory.init.params.update(theory_params)
    return emulated_pt


def validate_observable_prediction(observable, context='density-split theory'):
    """Fail fast when a loaded emulator has stale PT output metadata."""
    observable()
    found = int(observable.flattheory.size)
    expected = int(len(observable.quantiles) * len(observable.ells) * observable.k.size)
    data_size = int(observable.flatdata.size)
    if found != expected or found != data_size:
        raise ValueError(
            f'{context} produced a theory vector with length {found}, expected {expected} '
            f'from quantiles={tuple(observable.quantiles)}, ells={tuple(observable.ells)}, '
            f'nk={observable.k.size}, and data length {data_size}. '
            'This usually means the emulator was trained with a different k-grid, ell basis, '
            'or PT output shape; choose a new --emulator path or delete the stale emulator file.'
        )
    return observable.flattheory


def default_ref(prior, value=None):
    if not prior:
        return None
    dist = prior.get('dist', 'uniform')
    limits = prior.get('limits', None)
    if dist == 'norm':
        ref = {'dist': 'norm', 'loc': prior.get('loc', value if value is not None else 0.),
               'scale': prior.get('scale', 1.)}
        if limits is not None:
            ref['limits'] = limits
        return ref
    if dist == 'uniform' and limits is not None:
        lo, hi = limits
        if np.all(np.isfinite([lo, hi])):
            return {'dist': 'norm', 'loc': value if value is not None else 0.5 * (lo + hi),
                    'scale': (hi - lo) / 20., 'limits': [lo, hi]}
    return None


def nuisance_priors(prior_basis='physical_aap', b3_coev=True):
    params = {}
    if prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4.]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0., 'scale': 5.}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2. / 7., 'scale': 5.}}
        params['b3p'] = {'fixed': True} if b3_coev else {'prior': {'dist': 'norm', 'loc': 23. / 42., 'scale': 1.}, 'fixed': False}
        for ell in [0, 2, 4]:
            params[f'alpha{ell:d}p'] = {'prior': {'dist': 'norm', 'loc': 0., 'scale': 12.5}}
        params['ctp'] = {'fixed': True}
        params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0., 10.]}}
    else:
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10.]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50., 50.]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50., 50.]}}
        params['b3'] = {'fixed': True} if b3_coev else {'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'fixed': False}
        for ell in [0, 2, 4]:
            params[f'alpha{ell:d}'] = {'prior': {'dist': 'norm', 'loc': 0., 'scale': 12.5}}
        params['ct'] = {'fixed': True}
        params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0., 10.]}}
    for config in params.values():
        if config.get('fixed', False):
            continue
        ref = default_ref(config.get('prior', None), value=config.get('value', None))
        if ref is not None and 'ref' not in config:
            config['ref'] = ref
    return params


def apply_galaxy_priors(theory, prior_basis='physical_aap', b3_coev=True):
    for name, config in nuisance_priors(prior_basis=prior_basis, b3_coev=b3_coev).items():
        for param in theory.init.params.select(basename=name):
            param.update(**config)


def density_split_linear_parameter_basenames(model, qg_anisotropic_stochastic=False):
    """Return the exact density-split linear block for analytic marginalization."""
    anisotropic = ['s0muqg*'] if qg_anisotropic_stochastic else []
    if model == 'tree':
        return ['c1q*', 's0qg*'] + anisotropic
    if model == '1-loop':
        return ['c1q*', 'c2q*', 'e0q*', 'e2q*', 'e4q*',
                's0qg*'] + anisotropic + ['s2qg*', 's2muqg*']
    raise ValueError(f'Unknown density-split model {model!r}.')


def apply_density_split_linear_marginalization(theory, model, marginalize=True, qg_anisotropic_stochastic=False):
    """Mark exact density-split linear amplitudes as solved by the likelihood."""
    if not marginalize:
        return []
    names = density_split_linear_parameter_basenames(model, qg_anisotropic_stochastic=qg_anisotropic_stochastic)
    solved = []
    for param in theory.init.params.select(basename=names):
        param.update(derived='.auto')
        solved.append(param.basename)
    return solved


def build_theory(args, template):
    loop_resolution = getattr(args, 'composite_loop_resolution', 'default')
    loop_kwargs = dict(COMPOSITE_LOOP_RESOLUTIONS[loop_resolution])
    theory_kwargs = dict(template=template, model=args.theory_model, smoothing_radius=args.smoothing_radius,
                         smoothing_kernel=args.smoothing_kernel, smoothing_apmode=args.smoothing_apmode,
                         qg_anisotropic_stochastic=getattr(args, 'qg_anisotropic_stochastic', False),
                         **loop_kwargs)
    theory = DensitySplitTracerPowerSpectrumMultipoles(prior_basis=args.prior_basis, damping='lor',
                                                       b3_coev=True, backend='jax', **theory_kwargs)
    apply_galaxy_priors(theory, prior_basis=args.prior_basis, b3_coev=True)
    apply_density_split_linear_marginalization(theory, args.theory_model,
                                               marginalize=not getattr(args, 'sample_ds_linear', False),
                                               qg_anisotropic_stochastic=getattr(args, 'qg_anisotropic_stochastic', False))
    return theory


def add_on_the_fly_measurement_args(parser):
    parser.add_argument('--on-the-fly-measurements', action='store_true',
                        help='Generate and measure the density-split data vector on the fly with mockfactory/ACM.')
    parser.add_argument('--measurement-save-dir', type=Path, default=Path('scripts/mockfactory_acm_measurements'),
                        help='Base directory for on-the-fly measurement outputs.')
    parser.add_argument('--measurement-overwrite', action='store_true',
                        help='Overwrite existing on-the-fly measurement outputs.')
    parser.add_argument('--measurement-bias', type=float, default=2.0, help='Eulerian galaxy bias for on-the-fly measurements.')
    parser.add_argument('--measurement-redshift', type=float, default=0.5,
                        help='Redshift for the on-the-fly linear power spectrum.')
    parser.add_argument('--fiducial-cosmology', default='desi',
                        help="Truth cosmology for on-the-fly measurements: 'desi' or an AbacusSummit label such as 'abacus-c001'.")
    parser.add_argument('--measurement-boxsize', type=float, default=2000.0,
                        help='Cubic box size in Mpc/h for on-the-fly measurements.')
    parser.add_argument('--measurement-nbar', type=float, default=5e-4,
                        help='Mean number density in (Mpc/h)^-3 for on-the-fly measurements.')
    parser.add_argument('--measurement-nmesh', type=int, default=512,
                        help='mockfactory density-field mesh size for on-the-fly measurements.')
    parser.add_argument('--measurement-seed', type=int, default=None,
                        help='Base random seed for on-the-fly measurements; defaults to --seed.')
    parser.add_argument('--measurement-no-rsd', action='store_true',
                        help='Disable redshift-space distortions in on-the-fly measurements.')
    parser.add_argument('--measurement-meshsize', type=int, nargs='+', default=[512],
                        help='ACM power-spectrum mesh size for on-the-fly measurements: one integer or three integers.')
    parser.add_argument('--measurement-cellsize', type=float, default=3.9,
                        help='Density-split mesh cell size in Mpc/h for on-the-fly measurements; set <= 0 to reuse --measurement-meshsize.')
    parser.add_argument('--measurement-los', choices=('x', 'y', 'z'), default='z',
                        help='Line-of-sight direction for on-the-fly measurements.')
    parser.add_argument('--measurement-k-step', type=float, default=0.001,
                        help='Power-spectrum k-bin step for on-the-fly measurements.')


def measurement_seed(args):
    return args.seed if args.measurement_seed is None else args.measurement_seed


def mesh_nyquist(meshsize, boxsize):
    meshsize = np.asarray(meshsize, dtype='f8')
    boxsize = np.asarray(boxsize, dtype='f8')
    if boxsize.ndim == 0:
        boxsize = np.full(meshsize.shape, float(boxsize), dtype='f8')
    return float(np.min(np.pi * meshsize / boxsize))


def density_split_meshsize(boxsize, cellsize, measurement_meshsize):
    if float(cellsize) > 0.:
        return int(np.floor(float(boxsize) / float(cellsize)))
    return measurement_meshsize


def validate_on_the_fly_resolution(args, measurement_meshsize):
    kmax = getattr(args, 'kmax', None)
    if kmax is None:
        return

    kmax = float(kmax)
    boxsize = float(args.measurement_boxsize)
    ds_meshsize = density_split_meshsize(boxsize, args.measurement_cellsize, measurement_meshsize)
    checks = [
        ('measurement_nmesh', args.measurement_nmesh, mesh_nyquist(args.measurement_nmesh, boxsize)),
        ('measurement_meshsize', measurement_meshsize, mesh_nyquist(measurement_meshsize, boxsize)),
        ('measurement_cellsize', ds_meshsize, mesh_nyquist(ds_meshsize, boxsize)),
    ]
    for name, value, knyquist in checks:
        if kmax > 0.5 * knyquist:
            minimum = int(np.ceil(2. * kmax * boxsize / np.pi))
            raise ValueError(
                f'On-the-fly density-split measurements require --kmax <= 0.5 * k_Nyquist for {name}; '
                f'got kmax={kmax:g}, {name}={value}, k_Nyquist={knyquist:g}. '
                f'Increase --{name.replace("_", "-")} or lower --kmax. '
                f'A mesh-based guard would require at least {minimum} cells along the limiting axis.'
            )


def resolve_density_split_data(args):
    """Return the density-split data-vector path, measuring it first if requested."""
    if not args.on_the_fly_measurements:
        return args.data

    measurement_meshsize = parse_meshsize(args.measurement_meshsize)
    validate_on_the_fly_resolution(args, measurement_meshsize)
    paths = measure_mockfactory_acm(
        bias=args.measurement_bias,
        redshift=args.measurement_redshift,
        boxsize=args.measurement_boxsize,
        nbar=args.measurement_nbar,
        nmesh=args.measurement_nmesh,
        seed=measurement_seed(args),
        fiducial_cosmology=args.fiducial_cosmology,
        meshsize=measurement_meshsize,
        cellsize=args.measurement_cellsize,
        smoothing_radius=args.smoothing_radius,
        nquantiles=5,
        los=args.measurement_los,
        ells=tuple(args.ells),
        k_step=args.measurement_k_step,
        save_dir=args.measurement_save_dir,
        overwrite=args.measurement_overwrite,
        rsd=not args.measurement_no_rsd,
    )
    return paths['pkqg']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fit density-split multipoles with desilike and Minuit.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Raw density-split mean-measurement HDF5 file.')
    parser.add_argument('--mocks', type=Path, default=DEFAULT_MOCKS, help='Directory with raw density-split mock HDF5 files.')
    parser.add_argument('--mock-pattern', default='dsc_pkqg_poles_ph*.h5', help='Glob pattern for raw density-split mock files.')
    parser.add_argument('--quantiles', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='1-based density-split quantiles to fit.')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2], help='Multipoles to fit.')
    parser.add_argument('--rebin', type=int, default=13, help='Rebinning factor applied to raw HDF5 k bins.')
    parser.add_argument('--kmin', type=float, default=0.01, help='Minimum k included in the fit.')
    parser.add_argument('--kmax', type=float, default=0.2, help='Maximum k included in the fit.')
    parser.add_argument('--z', type=float, default=0.5, help='Template redshift.')
    parser.add_argument('--fiducial', default='DESI', help='Fiducial cosmology passed to the power-spectrum template.')
    parser.add_argument('--template', choices=['fixed', 'standard', 'shapefit', 'direct'], default='fixed',
                        help='Power-spectrum template used by the density-split theory.')
    parser.add_argument('--template-apmode', choices=['qparqper', 'qisoqap', 'qiso', 'qap'], default='qparqper',
                        help='AP parameterization for the standard and shapefit templates.')
    parser.add_argument('--template-engine', choices=['class', 'camb'], default='class',
                        help='Boltzmann engine for the direct template.')
    parser.add_argument('--template-params', nargs='+', default=None,
                        help='Optional allowlist of template/cosmology parameters to vary.')
    parser.add_argument('--emulator', type=Path, default=None,
                        help='Optional path to a saved Taylor FOLPS PT emulator; train and save it if missing.')
    parser.add_argument('--theory-model', choices=list(FOLPS_DENSITY_SPLIT_MODELS),
                        default='1-loop', help='Density-split Pqg theory model to fit.')
    parser.add_argument('--composite-loop-resolution', choices=list(COMPOSITE_LOOP_RESOLUTIONS), default='default',
                        help='Quadrature resolution preset for the composite c2 loop in the 1-loop model.')
    parser.add_argument('--prior-basis', choices=['physical', 'physical_aap', 'standard', 'tcm_chudaykin_aap'],
                        default='physical_aap', help='FOLPS nuisance prior basis for FOLPS density-split models.')
    parser.add_argument('--smoothing-radius', type=float, default=20., help='Gaussian smoothing radius in Mpc/h.')
    parser.add_argument('--smoothing-kernel', choices=['gaussian', 'tophat'], default='gaussian', help='Density-split smoothing kernel.')
    parser.add_argument('--smoothing-apmode', choices=['observed', 'physical'], default='physical',
                        help='Evaluate the density-split smoothing kernel at observed k or AP-remapped physical k.')
    parser.add_argument('--sample-ds-linear', action='store_true',
                        help='Sample density-split linear amplitudes instead of analytically marginalizing them.')
    parser.add_argument('--qg-anisotropic-stochastic', action='store_true',
                        help='Enable optional Pqg anisotropic white stochastic terms s0muqg* mu^2.')
    parser.add_argument('--covariance-rescale', type=float, default=64., help='Divide the raw sample covariance by this factor.')
    parser.add_argument('--covariance-jitter', type=float, default=0., help='Optional diagonal jitter added after covariance rescaling.')
    parser.add_argument('--max-mocks', type=int, default=None, help='Optional cap on the number of raw mocks, useful for smoke tests.')
    add_on_the_fly_measurement_args(parser)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for MinuitProfiler starting points.')
    parser.add_argument('--niterations', type=int, default=5, help='Number of Minuit starting points.')
    parser.add_argument('--output', type=Path, default=Path('density_split_minuit_profiles.npy'), help='Output profiles file.')
    parser.add_argument('--plot-output', type=Path, default=None, help='Optional output path for a best-fit data/theory plot.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()

    quantiles = tuple(args.quantiles)
    ells = tuple(args.ells)
    data_path = resolve_density_split_data(args)
    data = load_density_split_power_spectrum_multipoles(
        data_path,
        quantiles=quantiles,
        ells=ells,
        rebin=args.rebin,
        kmin=args.kmin,
        kmax=args.kmax,
    )
    k = get_density_split_k(data)
    covariance = density_split_sample_covariance(
        args.mocks,
        quantiles=quantiles,
        ells=ells,
        rebin=args.rebin,
        kmin=args.kmin,
        kmax=args.kmax,
        k=k,
        pattern=args.mock_pattern,
        max_mocks=args.max_mocks,
        covariance_rescale=args.covariance_rescale,
    )
    if args.covariance_jitter > 0.:
        covariance = covariance + np.eye(covariance.shape[0], dtype='f8') * float(args.covariance_jitter)

    template = build_template(args)
    select_params(template, args.template, args.template_params)
    theory = build_theory(args, template)
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=quantiles, ells=ells, rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance)
    validate_observable_prediction(observable, context=args.theory_model)
    if args.emulator is not None:
        maybe_emulate_pt(theory, args.emulator)
        validate_observable_prediction(observable, context='emulated {}'.format(args.theory_model))
        likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance)

    print('Density-split Minuit fit')
    print(f'data: {data_path}')
    if args.on_the_fly_measurements:
        print('on_the_fly_measurements: True')
        print(f'measurement_save_dir: {args.measurement_save_dir}')
        print(f'measurement_seed: {measurement_seed(args)}')
        print(f'fiducial_cosmology: {args.fiducial_cosmology}')
        print(f'measurement_rsd: {not args.measurement_no_rsd}')
    print(f'mocks: {args.mocks}')
    print(f'quantiles: {", ".join(str(value) for value in quantiles)}')
    print(f'ells: {", ".join(str(value) for value in ells)}')
    print(f'template: {args.template}')
    if args.template in ['standard', 'shapefit']:
        print(f'template_apmode: {args.template_apmode}')
    if args.template == 'direct':
        print(f'template_engine: {args.template_engine}')
    if args.template_params:
        print(f'template_params: {", ".join(args.template_params)}')
    if args.emulator is not None:
        print(f'emulator_path: {_emulator_path(args.emulator)}')
    print(f'theory_model: {args.theory_model}')
    print(f'folps_model: {FOLPS_MODEL}')
    print(f'prior_basis: {args.prior_basis}')
    print('damping: lor')
    print('b3_coev: True')
    print(f'qg_anisotropic_stochastic: {args.qg_anisotropic_stochastic}')
    ds_solved = likelihood.all_params.select(solved=True).names()
    print(f'ds_linear_marginalization: {not args.sample_ds_linear}')
    print(f'ds_linear_solved_parameters: {", ".join(ds_solved) if ds_solved else "none"}')
    if args.theory_model == '1-loop':
        print('composite_quantile_rule: q3 coefficients derived from q1,q2,q4,q5')
        print(f'composite_loop_resolution: {args.composite_loop_resolution}')
        print('composite_loop_checkpoint: c1 deterministic FOLPS one-loop plus explicit c2 loop; c3 fixed; Pqg stochastic terms active')
    print(f'smoothing_kernel: {args.smoothing_kernel}')
    print(f'smoothing_apmode: {args.smoothing_apmode}')
    print(f'n_k: {len(k)}')
    print(f'data_vector_length: {observable.flatdata.size}')
    print(f'covariance_shape: {covariance.shape}')
    print(f'varied parameters: {", ".join(likelihood.varied_params.names())}')

    profiler = MinuitProfiler(likelihood, seed=args.seed)
    profiles = profiler.maximize(niterations=args.niterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiles.save(args.output)

    if args.plot_output is not None:
        likelihood(**profiles.bestfit.choice(varied=True))
        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        observable.plot(fn=args.plot_output)
        print(f'plot_output: {args.plot_output}')

    print(f'output: {args.output}')
    print(profiles.to_stats(tablefmt='pretty'))


if __name__ == '__main__':
    main()
