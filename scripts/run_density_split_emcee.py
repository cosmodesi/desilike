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
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    DensitySplitPowerSpectrumMultipolesObservable,
    density_split_sample_covariance,
    get_density_split_k,
    load_density_split_power_spectrum_multipoles,
)
from desilike.samplers import EmceeSampler

from scripts.run_density_split_minuit import (
    COMPOSITE_LOOP_RESOLUTIONS,
    DEFAULT_DATA,
    DEFAULT_MOCKS,
    FOLPS_DENSITY_SPLIT_MODELS,
    FOLPS_MODEL,
    _emulator_path,
    add_on_the_fly_measurement_args,
    build_theory,
    build_template,
    maybe_emulate_pt,
    measurement_seed,
    resolve_density_split_data,
    select_params,
)


def chain_filenames(output, nchains):
    output = Path(output)
    pattern = str(output)
    if '*' in pattern:
        return [pattern.replace('*', str(ichain)) for ichain in range(nchains)]
    if nchains == 1:
        return [str(output)]
    return [str(output.with_name(f'{output.stem}_{ichain}{output.suffix}')) for ichain in range(nchains)]


def build_likelihood(args):
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
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=quantiles,
                                                               ells=ells, rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance)
    if args.emulator is not None:
        observable()
        maybe_emulate_pt(theory, args.emulator)
        likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance)
    return likelihood, observable, k, data_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sample density-split multipoles with desilike and emcee.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Raw density-split mean-measurement HDF5 file.')
    parser.add_argument('--mocks', type=Path, default=DEFAULT_MOCKS, help='Directory with raw density-split mock HDF5 files.')
    parser.add_argument('--mock-pattern', default='dsc_pkqg_poles_ph*.h5', help='Glob pattern for raw density-split mock files.')
    parser.add_argument('--quantiles', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='1-based density-split quantiles to fit.')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2, 4], help='Multipoles to fit.')
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

    parser.add_argument('--seed', type=int, default=42, help='Random seed for the emcee sampler.')
    parser.add_argument('--chains', type=int, default=4, help='Number of independent chains.')
    parser.add_argument('--nwalkers', default=None, help='Number of emcee walkers, e.g. 64 or "3 * ndim".')
    parser.add_argument('--max-iterations', type=int, default=100000, help='Maximum number of MCMC iterations.')
    parser.add_argument('--min-iterations', type=int, default=0, help='Minimum number of MCMC iterations before convergence can stop the run.')
    parser.add_argument('--check-every', type=int, default=50, help='Iteration cadence for saving and convergence checks.')
    parser.add_argument('--max-eigen-gr', type=float, default=0.03, help='Stop when the maximum eigenvalue Gelman-Rubin R-1 is below this value.')
    parser.add_argument('--min-ess', type=float, default=None, help='Optional minimum effective sample size convergence criterion.')
    parser.add_argument('--thin-by', type=int, default=1, help='Thin samples by this factor inside emcee.')
    parser.add_argument('--ref-scale', type=float, default=1., help='Scale the initial reference distributions.')
    parser.add_argument('--max-tries', type=int, default=1000, help='Maximum attempts to find finite initial posterior points.')
    parser.add_argument('--no-check', action='store_true', help='Disable convergence checks; useful for short smoke runs.')
    parser.add_argument('--resume', action='store_true', help='Resume from the chain files specified by --output.')
    parser.add_argument('--burnin', type=float, default=0., help='Burn-in fraction or count used only for the printed summary.')
    parser.add_argument('--output', type=Path, default=Path('density_split_emcee_chain_*.npy'), help='Output chain filename or pattern.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()

    likelihood, observable, k, data_path = build_likelihood(args)
    chain_fns = chain_filenames(args.output, args.chains)
    for fn in chain_fns:
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
    chains = chain_fns if args.resume else args.chains

    print('Density-split emcee fit')
    print(f'data: {data_path}')
    if args.on_the_fly_measurements:
        print('on_the_fly_measurements: True')
        print(f'measurement_save_dir: {args.measurement_save_dir}')
        print(f'measurement_seed: {measurement_seed(args)}')
        print(f'fiducial_cosmology: {args.fiducial_cosmology}')
        print(f'measurement_rsd: {not args.measurement_no_rsd}')
    print(f'mocks: {args.mocks}')
    print(f'quantiles: {", ".join(str(value) for value in args.quantiles)}')
    print(f'ells: {", ".join(str(value) for value in args.ells)}')
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
    print(f'covariance_shape: {likelihood.covariance.shape}')
    print(f'varied parameters: {", ".join(likelihood.varied_params.names())}')
    print(f'chains: {args.chains}')
    print(f'chain_files: {", ".join(chain_fns)}')
    print(f'nwalkers: {args.nwalkers if args.nwalkers is not None else "default"}')
    print(f'max_iterations: {args.max_iterations}')
    print(f'check_every: {args.check_every}')

    sampler_kwargs = dict(seed=args.seed, chains=chains, save_fn=chain_fns, ref_scale=args.ref_scale, max_tries=args.max_tries)
    if args.nwalkers is not None:
        sampler_kwargs['nwalkers'] = args.nwalkers
    sampler = EmceeSampler(likelihood, **sampler_kwargs)

    check = False if args.no_check else {'max_eigen_gr': args.max_eigen_gr}
    if isinstance(check, dict) and args.min_ess is not None:
        check['min_ess'] = args.min_ess
    chains = sampler.run(min_iterations=args.min_iterations, max_iterations=args.max_iterations,
                         check_every=args.check_every, check=check, thin_by=args.thin_by)

    if sampler.mpicomm.rank == 0:
        print(f'output: {", ".join(chain_fns)}')
        if chains:
            chain = chains[0].concatenate(chains) if len(chains) > 1 else chains[0]
            if args.burnin:
                chain = chain.remove_burnin(args.burnin)
            print(chain.to_stats(tablefmt='pretty'))


if __name__ == '__main__':
    main()
