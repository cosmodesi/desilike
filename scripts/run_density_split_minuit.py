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
from desilike.profilers import MinuitProfiler
from desilike.theories.galaxy_clustering import DensitySplitTracerPowerSpectrumMultipoles, FixedPowerSpectrumTemplate


JAXPT_ROOT = Path('/Users/epaillas/code/jax-pt')
DEFAULT_DATA = JAXPT_ROOT / 'data' / 'data_vector' / 'dsc_pkqg_poles_c000_hod006.h5'
DEFAULT_MOCKS = JAXPT_ROOT / 'data' / 'for_covariance'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fit tree-level density-split multipoles with desilike and Minuit.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Raw density-split mean-measurement HDF5 file.')
    parser.add_argument('--mocks', type=Path, default=DEFAULT_MOCKS, help='Directory with raw density-split mock HDF5 files.')
    parser.add_argument('--mock-pattern', default='dsc_pkqg_poles_ph*.h5', help='Glob pattern for raw density-split mock files.')
    parser.add_argument('--quantiles', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='1-based density-split quantiles to fit.')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2, 4], help='Multipoles to fit.')
    parser.add_argument('--rebin', type=int, default=13, help='Rebinning factor applied to raw HDF5 k bins.')
    parser.add_argument('--kmin', type=float, default=0.01, help='Minimum k included in the fit.')
    parser.add_argument('--kmax', type=float, default=0.2, help='Maximum k included in the fit.')
    parser.add_argument('--z', type=float, default=0.5, help='Template redshift.')
    parser.add_argument('--fiducial', default='DESI', help='Fiducial cosmology passed to FixedPowerSpectrumTemplate.')
    parser.add_argument('--smoothing-radius', type=float, default=10., help='Gaussian smoothing radius in Mpc/h.')
    parser.add_argument('--covariance-rescale', type=float, default=64., help='Divide the raw sample covariance by this factor.')
    parser.add_argument('--covariance-jitter', type=float, default=0., help='Optional diagonal jitter added after covariance rescaling.')
    parser.add_argument('--max-mocks', type=int, default=None, help='Optional cap on the number of raw mocks, useful for smoke tests.')
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
    data = load_density_split_power_spectrum_multipoles(
        args.data,
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

    template = FixedPowerSpectrumTemplate(z=args.z, fiducial=args.fiducial)
    theory = DensitySplitTracerPowerSpectrumMultipoles(template=template, smoothing_radius=args.smoothing_radius)
    observable = DensitySplitPowerSpectrumMultipolesObservable(data=data, theory=theory, quantiles=quantiles, ells=ells, rebin=None, kmin=None, kmax=None)
    likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance)

    print('Density-split Minuit fit')
    print(f'data: {args.data}')
    print(f'mocks: {args.mocks}')
    print(f'quantiles: {", ".join(str(value) for value in quantiles)}')
    print(f'ells: {", ".join(str(value) for value in ells)}')
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
