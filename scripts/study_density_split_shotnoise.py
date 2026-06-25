from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import numpy as np

from desilike.observables.galaxy_clustering import (
    get_density_split_k,
    load_density_split_power_spectrum_multipoles,
)
from scripts.measure_mockfactory_acm import measure_mockfactory_acm, parse_meshsize


def nbar_label(nbar):
    return 'nbar_{:.1e}'.format(float(nbar)).replace('+', '').replace('.', 'p')


def parse_args():
    parser = argparse.ArgumentParser(description='Study density-split Pqg response to query-selection shot noise.')
    parser.add_argument('--output-dir', type=Path, default=Path('tmp/density_split_shotnoise_study'),
                        help='Directory for measurements, plots, and summary files.')
    parser.add_argument('--nbar-values', type=float, nargs='+', default=[1e-3, 5e-4, 2e-4, 1e-4],
                        help='Mean number densities in (Mpc/h)^-3 to measure.')
    parser.add_argument('--reference-nbar', type=float, default=None,
                        help='Reference nbar for residuals. Defaults to the largest nbar.')
    parser.add_argument('--bias', type=float, default=2.0, help='Eulerian galaxy bias.')
    parser.add_argument('--redshift', type=float, default=0.5, help='Mock redshift.')
    parser.add_argument('--fiducial-cosmology', default='desi',
                        help="Truth cosmology for mock generation: 'desi' or an AbacusSummit label.")
    parser.add_argument('--boxsize', type=float, default=1000.0, help='Cubic box size in Mpc/h.')
    parser.add_argument('--nmesh', type=int, default=256, help='mockfactory density-field mesh size.')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed shared by all nbar values.')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Explicit random seeds to average. Overrides --seed/--nseeds when provided.')
    parser.add_argument('--nseeds', type=int, default=1,
                        help='Number of consecutive seeds to average, starting from --seed.')
    parser.add_argument('--meshsize', type=int, nargs='+', default=[256],
                        help='ACM power-spectrum mesh size: one integer or three integers.')
    parser.add_argument('--cellsize', type=float, default=-1.,
                        help='Density-split cell size in Mpc/h; set <= 0 to reuse --meshsize.')
    parser.add_argument('--smoothing-radius', type=float, default=10.0, help='Density-split smoothing radius in Mpc/h.')
    parser.add_argument('--nquantiles', type=int, default=5, help='Number of density-split quantiles.')
    parser.add_argument('--quantiles', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='1-based quantiles to plot and fit.')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2], help='Multipoles to measure, plot, and fit.')
    parser.add_argument('--los', choices=('x', 'y', 'z'), default='z', help='Line-of-sight direction.')
    parser.add_argument('--k-step', type=float, default=0.001, help='Raw power-spectrum k-bin step.')
    parser.add_argument('--rebin', type=int, default=13, help='Rebinning factor for plots and diagnostics.')
    parser.add_argument('--kmin', type=float, default=0.01, help='Minimum k for plots and diagnostics.')
    parser.add_argument('--kmax', type=float, default=0.2, help='Maximum k for plots and diagnostics.')
    parser.add_argument('--no-rsd', action='store_true', help='Disable redshift-space distortions.')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate existing measurement files.')
    parser.add_argument('--skip-measurements', action='store_true',
                        help='Only read existing outputs and remake plots/diagnostics.')
    parser.add_argument('--shotnoise-scaling', choices=['inverse-nbar', 'fixed'], default='inverse-nbar',
                        help='Shot-noise normalization used for the stochastic-basis diagnostic fits.')
    parser.add_argument('--fixed-shotnoise', type=float, default=1e4,
                        help='Shot-noise normalization when --shotnoise-scaling=fixed.')
    return parser.parse_args()


def seed_values(args):
    if args.seeds is not None:
        if not args.seeds:
            raise ValueError('--seeds must contain at least one seed')
        return [int(seed) for seed in args.seeds]
    if int(args.nseeds) <= 0:
        raise ValueError('--nseeds must be positive')
    return [int(args.seed) + offset for offset in range(int(args.nseeds))]


def measurement_dir(output_dir, nbar, seed, multi_seed=False):
    output_dir = Path(output_dir)
    if multi_seed:
        return output_dir / 'measurements' / 'seed_{:d}'.format(int(seed)) / nbar_label(nbar)
    return output_dir / 'measurements' / nbar_label(nbar)


def run_measurements(args):
    meshsize = parse_meshsize(args.meshsize)
    seeds = seed_values(args)
    multi_seed = len(seeds) > 1 or args.seeds is not None
    paths = {}
    for seed in seeds:
        seed_paths = {}
        for nbar in args.nbar_values:
            save_dir = measurement_dir(args.output_dir, nbar, seed, multi_seed=multi_seed)
            if args.skip_measurements:
                from scripts.measure_mockfactory_acm import output_paths

                seed_paths[float(nbar)] = output_paths(save_dir, seed, fiducial_cosmology=args.fiducial_cosmology)
                continue
            seed_paths[float(nbar)] = measure_mockfactory_acm(
                bias=args.bias,
                redshift=args.redshift,
                boxsize=args.boxsize,
                nbar=float(nbar),
                nmesh=args.nmesh,
                seed=seed,
                fiducial_cosmology=args.fiducial_cosmology,
                meshsize=meshsize,
                cellsize=args.cellsize,
                smoothing_radius=args.smoothing_radius,
                nquantiles=args.nquantiles,
                los=args.los,
                ells=tuple(args.ells),
                k_step=args.k_step,
                save_dir=save_dir,
                overwrite=args.overwrite,
                rsd=not args.no_rsd,
            )
        paths[int(seed)] = seed_paths
    return paths


def load_measurements(paths, args):
    seeds = sorted(paths)
    nbar_values = sorted(next(iter(paths.values())))
    seed_arrays = []
    k_ref = None
    for seed in seeds:
        arrays = []
        seed_nbar_values = sorted(paths[seed])
        if seed_nbar_values != nbar_values:
            raise ValueError('nbar grids differ between seeds')
        for nbar in nbar_values:
            data = load_density_split_power_spectrum_multipoles(
                paths[seed][nbar]['pkqg'],
                quantiles=tuple(args.quantiles),
                ells=tuple(args.ells),
                rebin=args.rebin,
                kmin=args.kmin,
                kmax=args.kmax,
            )
            k = get_density_split_k(data)
            if k_ref is None:
                k_ref = k
            elif not np.allclose(k, k_ref, rtol=1e-10, atol=1e-12):
                raise ValueError('k grids differ between nbar measurements')
            rows = []
            for quantile in args.quantiles:
                branch = data.get(quantiles=int(quantile) - 1)
                rows.append([np.asarray(branch.get(ells=int(ell)).value(), dtype='f8') for ell in args.ells])
            arrays.append(rows)
        seed_arrays.append(arrays)
    seed_power = np.asarray(seed_arrays, dtype='f8')
    power_mean = np.mean(seed_power, axis=0)
    power_std = np.std(seed_power, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(power_mean)
    power_sem = power_std / np.sqrt(len(seeds))
    return (np.asarray(seeds, dtype='i8'), np.asarray(nbar_values, dtype='f8'), np.asarray(k_ref, dtype='f8'),
            seed_power, power_mean, power_std, power_sem)


def stochastic_basis(k, ells, shotnoise, include_mu0=False):
    columns = []
    for ell in ells:
        if int(ell) == 0:
            values = [
                np.full_like(k, float(shotnoise), dtype='f8'),
            ]
            if include_mu0:
                values.append(np.full_like(k, float(shotnoise) / 3., dtype='f8'))
            values += [
                float(shotnoise) * k**2,
                float(shotnoise) * k**2 / 3.,
            ]
            block = np.column_stack(values)
        elif int(ell) == 2:
            values = [
                np.zeros_like(k, dtype='f8'),
            ]
            if include_mu0:
                values.append(np.full_like(k, float(shotnoise) * 2. / 3., dtype='f8'))
            values += [
                np.zeros_like(k, dtype='f8'),
                float(shotnoise) * 2. * k**2 / 3.,
            ]
            block = np.column_stack(values)
        else:
            ncolumns = 4 if include_mu0 else 3
            block = np.zeros((k.size, ncolumns), dtype='f8')
        columns.append(block)
    return np.vstack(columns)


def shotnoise_for_nbar(nbar, args):
    if args.shotnoise_scaling == 'fixed':
        return float(args.fixed_shotnoise)
    return 1. / float(nbar)


def fit_basis_responses(nbar_values, k, power, args, include_amplitude=False, include_mu0=False):
    ref_nbar = float(np.max(nbar_values) if args.reference_nbar is None else args.reference_nbar)
    if not np.any(np.isclose(nbar_values, ref_nbar, rtol=1e-12, atol=0.)):
        raise ValueError('reference nbar {} is not in --nbar-values'.format(ref_nbar))
    iref = int(np.flatnonzero(np.isclose(nbar_values, ref_nbar, rtol=1e-12, atol=0.))[0])

    results = []
    fit_power = np.full_like(power, np.nan)
    for inbar, nbar in enumerate(nbar_values):
        if inbar == iref:
            continue
        stochastic = stochastic_basis(k, args.ells, shotnoise_for_nbar(nbar, args), include_mu0=include_mu0)
        for iq, quantile in enumerate(args.quantiles):
            target = (power[inbar, iq] - power[iref, iq]).reshape(-1)
            columns = []
            names = []
            if include_amplitude:
                columns.append(power[iref, iq].reshape(-1))
                names.append('amplitude')
            columns += [stochastic[:, index] for index in range(stochastic.shape[1])]
            names += ['s0qg', 's0muqg', 's2qg', 's2muqg'] if include_mu0 else ['s0qg', 's2qg', 's2muqg']
            basis = np.column_stack(columns)
            coeffs, _, rank, singular_values = np.linalg.lstsq(basis, target, rcond=None)
            model = basis.dot(coeffs).reshape(len(args.ells), k.size)
            fit_power[inbar, iq] = model
            residual = target - basis.dot(coeffs)
            rms_before = float(np.sqrt(np.mean(target**2)))
            rms_after = float(np.sqrt(np.mean(residual**2)))
            explained = 1. if rms_before == 0. else float(1. - np.sum(residual**2) / np.sum(target**2))
            results.append({
                'nbar': float(nbar),
                'reference_nbar': float(ref_nbar),
                'quantile': int(quantile),
                'basis': '{}{}stochastic'.format('amplitude+' if include_amplitude else '',
                                                  'mu0+' if include_mu0 else ''),
                'shotnoise_normalization': float(shotnoise_for_nbar(nbar, args)),
                'rank': int(rank),
                'singular_values': [float(value) for value in singular_values],
                'rms_before': rms_before,
                'rms_after': rms_after,
                'rms_ratio': float(rms_after / rms_before) if rms_before else 0.,
                'variance_explained': explained,
            })
            results[-1].update({name: float(value) for name, value in zip(names, coeffs)})
    return iref, fit_power, results


def fit_partition_stochastic_responses(nbar_values, k, power, args, include_amplitude=False, include_mu0=False):
    ref_nbar = float(np.max(nbar_values) if args.reference_nbar is None else args.reference_nbar)
    iref = int(np.flatnonzero(np.isclose(nbar_values, ref_nbar, rtol=1e-12, atol=0.))[0])
    independent_quantiles = [quantile for quantile in (1, 2, 4, 5) if quantile in args.quantiles]
    quantile_indices = {int(quantile): index for index, quantile in enumerate(args.quantiles)}
    prefixes = ['s0qg', 's0muqg', 's2qg', 's2muqg'] if include_mu0 else ['s0qg', 's2qg', 's2muqg']

    results = []
    fit_power = np.full_like(power, np.nan)
    for inbar, nbar in enumerate(nbar_values):
        if inbar == iref:
            continue
        one_quantile_basis = stochastic_basis(k, args.ells, shotnoise_for_nbar(nbar, args), include_mu0=include_mu0)
        rows = len(args.quantiles) * len(args.ells) * k.size
        columns = []
        names = []
        for quantile in independent_quantiles:
            if include_amplitude:
                column = np.zeros(rows, dtype='f8')
                qindex = quantile_indices[quantile]
                start = qindex * len(args.ells) * k.size
                stop = start + len(args.ells) * k.size
                column[start:stop] += power[iref, qindex].reshape(-1)
                if 3 in quantile_indices:
                    q3index = quantile_indices[3]
                    start = q3index * len(args.ells) * k.size
                    stop = start + len(args.ells) * k.size
                    column[start:stop] -= power[iref, qindex].reshape(-1)
                columns.append(column)
                names.append('amplitude{}'.format(quantile))
            for iprefix, prefix in enumerate(prefixes):
                column = np.zeros(rows, dtype='f8')
                qindex = quantile_indices[quantile]
                start = qindex * len(args.ells) * k.size
                stop = start + len(args.ells) * k.size
                column[start:stop] += one_quantile_basis[:, iprefix]
                if 3 in quantile_indices:
                    q3index = quantile_indices[3]
                    start = q3index * len(args.ells) * k.size
                    stop = start + len(args.ells) * k.size
                    column[start:stop] -= one_quantile_basis[:, iprefix]
                columns.append(column)
                names.append('{}{}'.format(prefix, quantile))
        basis = np.column_stack(columns)
        target = (power[inbar] - power[iref]).reshape(-1)
        coeffs, _, rank, singular_values = np.linalg.lstsq(basis, target, rcond=None)
        model = basis.dot(coeffs).reshape(len(args.quantiles), len(args.ells), k.size)
        fit_power[inbar] = model
        residual = target - basis.dot(coeffs)
        rms_before = float(np.sqrt(np.mean(target**2)))
        rms_after = float(np.sqrt(np.mean(residual**2)))
        explained = 1. if rms_before == 0. else float(1. - np.sum(residual**2) / np.sum(target**2))
        result = {
            'basis': 'partition-{}{}stochastic'.format('amplitude+' if include_amplitude else '',
                                                       'mu0+' if include_mu0 else ''),
            'nbar': float(nbar),
            'reference_nbar': float(ref_nbar),
            'quantile': 'all',
            'shotnoise_normalization': float(shotnoise_for_nbar(nbar, args)),
            'rank': int(rank),
            'singular_values': [float(value) for value in singular_values],
            'rms_before': rms_before,
            'rms_after': rms_after,
            'rms_ratio': float(rms_after / rms_before) if rms_before else 0.,
            'variance_explained': explained,
        }
        result.update({name: float(value) for name, value in zip(names, coeffs)})
        results.append(result)
    return fit_power, results


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def plot_measurements(output_dir, nbar_values, k, power, args, power_sem=None):
    from matplotlib import pyplot as plt

    colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, len(nbar_values)))
    fig, axes = plt.subplots(len(args.quantiles), len(args.ells), figsize=(4.5 * len(args.ells), 2.4 * len(args.quantiles)),
                             sharex=True, squeeze=False)
    for iq, quantile in enumerate(args.quantiles):
        for iell, ell in enumerate(args.ells):
            ax = axes[iq, iell]
            for inbar, (nbar, color) in enumerate(zip(nbar_values, colors)):
                ax.plot(k, k * power[inbar, iq, iell], color=color, label=r'$\bar{{n}}={:.1e}$'.format(nbar))
                if power_sem is not None and np.any(power_sem[inbar, iq, iell] > 0.):
                    y = k * power[inbar, iq, iell]
                    yerr = k * power_sem[inbar, iq, iell]
                    ax.fill_between(k, y - yerr, y + yerr, color=color, alpha=0.18, linewidth=0.)
            ax.axhline(0., color='0.8', linewidth=0.8)
            ax.set_title('q{} ell {}'.format(quantile, ell))
            ax.set_ylabel(r'$k P_{{qg,\ell}}$')
            if iq == len(args.quantiles) - 1:
                ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fn = Path(output_dir) / 'plots' / 'pqg_multipoles_by_nbar.png'
    fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fn, dpi=180)
    plt.close(fig)
    return fn


def plot_residuals(output_dir, nbar_values, k, power, iref, fit_power, args, filename, title_suffix='', power_sem=None):
    from matplotlib import pyplot as plt

    mask = [index for index in range(len(nbar_values)) if index != iref]
    colors = plt.get_cmap('plasma')(np.linspace(0.1, 0.9, len(mask)))
    fig, axes = plt.subplots(len(args.quantiles), len(args.ells), figsize=(4.5 * len(args.ells), 2.4 * len(args.quantiles)),
                             sharex=True, squeeze=False)
    for iq, quantile in enumerate(args.quantiles):
        for iell, ell in enumerate(args.ells):
            ax = axes[iq, iell]
            for color, inbar in zip(colors, mask):
                delta = power[inbar, iq, iell] - power[iref, iq, iell]
                ax.plot(k, k * delta, color=color, label=r'$\bar{{n}}={:.1e}$'.format(nbar_values[inbar]))
                if power_sem is not None and (np.any(power_sem[inbar, iq, iell] > 0.) or np.any(power_sem[iref, iq, iell] > 0.)):
                    yerr = k * np.sqrt(power_sem[inbar, iq, iell]**2 + power_sem[iref, iq, iell]**2)
                    ax.fill_between(k, k * delta - yerr, k * delta + yerr, color=color, alpha=0.18, linewidth=0.)
                if np.all(np.isfinite(fit_power[inbar, iq])):
                    ax.plot(k, k * fit_power[inbar, iq, iell], color=color, linestyle='--', linewidth=1.2)
            ax.axhline(0., color='0.8', linewidth=0.8)
            ax.set_title('q{} ell {}{}'.format(quantile, ell, title_suffix))
            ax.set_ylabel(r'$k \Delta P_{{qg,\ell}}$')
            if iq == len(args.quantiles) - 1:
                ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fn = Path(output_dir) / 'plots' / filename
    fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fn, dpi=180)
    plt.close(fig)
    return fn


def plot_fit_summary(output_dir, results):
    from matplotlib import pyplot as plt

    if not results:
        return None
    labels = ['{}, n={:.1e}, q{}'.format(item['basis'], item['nbar'], item['quantile']) for item in results]
    ratios = [item['rms_ratio'] for item in results]
    explained = [item['variance_explained'] for item in results]
    y = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(13., max(4., 0.22 * len(results))), sharey=True)
    axes[0].barh(y, ratios, color='C0')
    axes[0].set_xlabel('RMS after / before')
    axes[0].set_yticks(y, labels=labels, fontsize=7)
    axes[0].axvline(1., color='0.4', linewidth=0.8)
    axes[1].barh(y, explained, color='C2')
    axes[1].set_xlabel('variance explained')
    axes[1].set_xlim(min(0., min(explained) - 0.05), 1.02)
    axes[1].axvline(0., color='0.4', linewidth=0.8)
    fig.tight_layout()
    fn = Path(output_dir) / 'plots' / 'stochastic_fit_summary.png'
    fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fn, dpi=180)
    plt.close(fig)
    return fn


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = run_measurements(args)
    seeds, nbar_values, k, seed_power, power, power_std, power_sem = load_measurements(paths, args)
    iref, stochastic_fit_power, stochastic_fit_results = fit_basis_responses(nbar_values, k, power, args, include_amplitude=False)
    _, mu0_stochastic_fit_power, mu0_stochastic_fit_results = fit_basis_responses(
        nbar_values, k, power, args, include_amplitude=False, include_mu0=True)
    _, amplitude_stochastic_fit_power, amplitude_stochastic_fit_results = fit_basis_responses(nbar_values, k, power, args, include_amplitude=True)
    partition_stochastic_fit_power, partition_stochastic_fit_results = fit_partition_stochastic_responses(nbar_values, k, power, args)
    partition_mu0_stochastic_fit_power, partition_mu0_stochastic_fit_results = fit_partition_stochastic_responses(
        nbar_values, k, power, args, include_mu0=True)
    partition_amplitude_stochastic_fit_power, partition_amplitude_stochastic_fit_results = fit_partition_stochastic_responses(
        nbar_values, k, power, args, include_amplitude=True)
    partition_amplitude_mu0_stochastic_fit_power, partition_amplitude_mu0_stochastic_fit_results = fit_partition_stochastic_responses(
        nbar_values, k, power, args, include_amplitude=True, include_mu0=True)
    fit_results = (stochastic_fit_results + mu0_stochastic_fit_results + amplitude_stochastic_fit_results
                   + partition_stochastic_fit_results + partition_mu0_stochastic_fit_results
                   + partition_amplitude_stochastic_fit_results + partition_amplitude_mu0_stochastic_fit_results)

    np.savez(
        args.output_dir / 'pqg_shotnoise_measurements.npz',
        seeds=seeds,
        nbar=nbar_values,
        k=k,
        seed_power=seed_power,
        power=power,
        power_std=power_std,
        power_sem=power_sem,
        stochastic_fit_power=stochastic_fit_power,
        mu0_stochastic_fit_power=mu0_stochastic_fit_power,
        amplitude_stochastic_fit_power=amplitude_stochastic_fit_power,
        partition_stochastic_fit_power=partition_stochastic_fit_power,
        partition_mu0_stochastic_fit_power=partition_mu0_stochastic_fit_power,
        partition_amplitude_stochastic_fit_power=partition_amplitude_stochastic_fit_power,
        partition_amplitude_mu0_stochastic_fit_power=partition_amplitude_mu0_stochastic_fit_power,
        quantiles=np.asarray(args.quantiles, dtype='i4'),
        ells=np.asarray(args.ells, dtype='i4'),
        reference_nbar=nbar_values[iref],
    )
    plot_files = [
        plot_measurements(args.output_dir, nbar_values, k, power, args, power_sem=power_sem),
        plot_residuals(args.output_dir, nbar_values, k, power, iref, stochastic_fit_power, args,
                       filename='pqg_delta_and_stochastic_fit.png', title_suffix='', power_sem=power_sem),
        plot_residuals(args.output_dir, nbar_values, k, power, iref, mu0_stochastic_fit_power, args,
                       filename='pqg_delta_and_mu0_stochastic_fit.png', title_suffix=' mu0', power_sem=power_sem),
        plot_residuals(args.output_dir, nbar_values, k, power, iref, amplitude_stochastic_fit_power, args,
                       filename='pqg_delta_and_amplitude_stochastic_fit.png', title_suffix=' amp', power_sem=power_sem),
        plot_residuals(args.output_dir, nbar_values, k, power, iref, partition_amplitude_stochastic_fit_power, args,
                       filename='pqg_delta_and_partition_amplitude_stochastic_fit.png', title_suffix=' part amp',
                       power_sem=power_sem),
        plot_residuals(args.output_dir, nbar_values, k, power, iref, partition_amplitude_mu0_stochastic_fit_power, args,
                       filename='pqg_delta_and_partition_amplitude_mu0_stochastic_fit.png',
                       title_suffix=' part amp mu0', power_sem=power_sem),
        plot_fit_summary(args.output_dir, fit_results),
    ]
    metadata = {
        'settings': vars(args) | {'output_dir': str(args.output_dir)},
        'seeds': [int(seed) for seed in seeds],
        'nbar_values': [float(value) for value in nbar_values],
        'reference_nbar': float(nbar_values[iref]),
        'paths': {
            str(seed): {
                str(nbar): {key: str(value) for key, value in nbar_paths.items()}
                for nbar, nbar_paths in seed_paths.items()
            }
            for seed, seed_paths in paths.items()
        },
        'fit_results': fit_results,
        'plot_files': [str(path) for path in plot_files if path is not None],
    }
    save_json(args.output_dir / 'summary.json', metadata)

    print('Density-split shot-noise study complete')
    print('output_dir: {}'.format(args.output_dir))
    print('seeds: {}'.format(', '.join(str(seed) for seed in seeds)))
    print('reference_nbar: {:.6g}'.format(nbar_values[iref]))
    for path in plot_files:
        if path is not None:
            print('plot: {}'.format(path))
    print('summary: {}'.format(args.output_dir / 'summary.json'))


if __name__ == '__main__':
    main()
