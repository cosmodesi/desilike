"""Generate a mockfactory box mock and measure ACM power spectra.

This script uses local checkouts of mockfactory and acm, generates a
LagrangianLinearMock on the fly, and writes galaxy and density-split power
spectra to HDF5 files.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


MOCKFACTORY_DIR = Path("/Users/epaillas/code/mockfactory")
ACM_DIR = Path("/Users/epaillas/code/acm")
ABACUS_COSMOLOGY_RE = re.compile(r"^(?:abacus(?:summit)?)[-_]?c?(\d+)$")


def prepend_local_sources() -> None:
    """Prefer the local mockfactory and acm source trees."""
    for path in (ACM_DIR, MOCKFACTORY_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_meshsize(values: list[int]) -> int | np.ndarray:
    """Parse one or three mesh sizes from the command line."""
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return np.asarray(values, dtype=int)
    raise argparse.ArgumentTypeError("--meshsize expects one integer or three integers")


def normalize_fiducial_cosmology_name(name: str) -> str:
    """Normalize supported on-the-fly mock truth cosmology names."""
    cleaned = str(name).strip().lower().replace(" ", "")
    if cleaned == "desi":
        return "desi"
    match = ABACUS_COSMOLOGY_RE.match(cleaned)
    if match:
        return f"abacus-c{int(match.group(1)):03d}"
    raise ValueError(
        f"Unknown fiducial cosmology {name!r}. Use 'desi' or an AbacusSummit label "
        "such as 'abacus-c001'."
    )


def get_fiducial_cosmology(name: str):
    """Return the normalized cosmology label and cosmoprimo fiducial instance."""
    normalized = normalize_fiducial_cosmology_name(name)
    if normalized == "desi":
        from cosmoprimo.fiducial import DESI

        return normalized, DESI()

    from cosmoprimo.fiducial import AbacusSummit

    index = int(normalized.rsplit("c", 1)[1])
    return normalized, AbacusSummit(index)


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a mockfactory Lagrangian mock and measure ACM spectra.",
    )
    parser.add_argument("--bias", type=float, default=2.0, help="Eulerian galaxy bias.")
    parser.add_argument("--redshift", type=float, default=0.5, help="Redshift for the linear power spectrum.")
    parser.add_argument(
        "--fiducial-cosmology",
        default="desi",
        help="Truth cosmology for mock generation: 'desi' or an AbacusSummit label such as 'abacus-c001'.",
    )
    parser.add_argument("--boxsize", type=float, default=500.0, help="Cubic box size in Mpc/h.")
    parser.add_argument("--nbar", type=float, default=1e-3, help="Mean number density in (Mpc/h)^-3.")
    parser.add_argument("--nmesh", type=int, default=128, help="mockfactory density-field mesh size.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--no-rsd", action="store_true", help="Disable redshift-space distortions.")
    parser.add_argument(
        "--meshsize",
        type=int,
        nargs="+",
        default=[128],
        help="ACM power-spectrum mesh size: one integer or three integers.",
    )
    parser.add_argument(
        "--cellsize",
        type=float,
        default=3.9,
        help="Density-split mesh cell size in Mpc/h; set <= 0 to reuse --meshsize.",
    )
    parser.add_argument("--smoothing_radius", type=float, default=10.0, help="Density-split smoothing radius in Mpc/h.")
    parser.add_argument("--nquantiles", type=int, default=5, help="Number of density-split quantiles.")
    parser.add_argument("--los", choices=("x", "y", "z"), default="z", help="Line-of-sight direction.")
    parser.add_argument("--ells", type=int, nargs="+", default=[0, 2, 4], help="Multipoles to measure.")
    parser.add_argument("--k-step", type=float, default=0.001, help="Power-spectrum k-bin step.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("scripts/mockfactory_acm_measurements"),
        help="Base directory for measurement outputs.",
    )
    parser.add_argument("--no-density-split", action="store_true", help="Only measure the galaxy power spectrum.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def as_list(value: Any) -> Any:
    """Convert numpy scalars and arrays to JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: as_list(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_list(val) for val in value]
    return value


def cosmology_metadata(cosmo) -> dict[str, Any]:
    """Extract a compact JSON-safe summary of useful cosmological parameters."""
    metadata = {}
    for name in ["h", "H0", "omega_cdm", "omega_b", "Omega_m", "A_s", "logA", "n_s"]:
        try:
            metadata[name] = as_list(cosmo[name])
        except Exception:
            continue
    return metadata


def normalize_centered_positions(positions: np.ndarray, boxsize: float) -> np.ndarray:
    """Wrap positions into the centered interval [-L/2, L/2)."""
    return np.mod(positions + boxsize / 2.0, boxsize) - boxsize / 2.0


def generate_mock(
    *,
    bias: float,
    redshift: float,
    boxsize: float,
    nbar: float,
    nmesh: int,
    seed: int,
    los: str,
    rsd: bool = True,
    fiducial_cosmology: str = "desi",
) -> np.ndarray:
    from mockfactory import LagrangianLinearMock

    _, cosmo = get_fiducial_cosmology(fiducial_cosmology)
    fourier = cosmo.get_fourier()
    power = fourier.pk_interpolator().to_1d(z=redshift)
    mock = LagrangianLinearMock(
        power,
        nmesh=nmesh,
        boxsize=boxsize,
        boxcenter=0.0,
        seed=seed,
        unitary_amplitude=False,
    )
    mock.set_real_delta_field(bias=bias - 1.0)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed + 1)
    if rsd:
        growth_rate = fourier.sigma8_z(z=redshift, of="theta_cb") / fourier.sigma8_z(z=redshift, of="delta_cb")
        mock.set_rsd(f=growth_rate, los=los)
    catalog = mock.to_catalog()
    positions = np.asarray(catalog["Position"], dtype="f8")
    return normalize_centered_positions(positions, boxsize)


def growth_rate(redshift: float, fiducial_cosmology: str = "desi") -> float:
    _, cosmo = get_fiducial_cosmology(fiducial_cosmology)
    fourier = cosmo.get_fourier()
    return float(fourier.sigma8_z(z=redshift, of="theta_cb") / fourier.sigma8_z(z=redshift, of="delta_cb"))


def output_paths(save_dir: Path, seed: int, fiducial_cosmology: str = "desi") -> dict[str, Path]:
    save_dir = Path(save_dir)
    fiducial_cosmology = normalize_fiducial_cosmology_name(fiducial_cosmology)
    suffix = "" if fiducial_cosmology == "desi" else f"_{fiducial_cosmology}"
    spectrum_dir = save_dir / "spectrum"
    density_split_dir = save_dir / "density_split_power"
    return {
        "spectrum": spectrum_dir / f"mesh2_spectrum_poles_mockfactory_seed{seed}{suffix}.h5",
        "pkqg": density_split_dir / f"dsc_pkqg_poles_mockfactory_seed{seed}{suffix}.h5",
        "pkqq": density_split_dir / f"dsc_pkqq_poles_mockfactory_seed{seed}{suffix}.h5",
        "metadata": save_dir / f"mockfactory_seed{seed}{suffix}_metadata.json",
    }


def prepare_outputs(paths: dict[str, Path], overwrite: bool, product_keys: tuple[str, ...]) -> bool:
    """Prepare output directories; return True when all requested products already exist."""
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return False

    products = {key: paths[key] for key in product_keys}
    existing = {key: path for key, path in products.items() if path.exists()}
    missing = {key: path for key, path in products.items() if not path.exists()}
    if not existing:
        return False
    if not missing:
        message = "Using existing on-the-fly measurements: " + ", ".join(
            f"{key}={path}" for key, path in products.items()
        )
        logging.getLogger(__name__).info(message)
        return True

    existing_msg = ", ".join(f"{key}={path}" for key, path in existing.items())
    missing_msg = ", ".join(f"{key}={path}" for key, path in missing.items())
    raise FileExistsError(
        "Partial on-the-fly measurement cache found. "
        f"Existing files: {existing_msg}. Missing files: {missing_msg}. "
        "Pass --overwrite, or --measurement-overwrite in the fitting scripts, to regenerate the full set."
    )


def density_split_meshsize(boxsize: float, cellsize: float, fallback: int | np.ndarray) -> int | np.ndarray:
    if cellsize <= 0:
        return fallback
    return np.maximum(np.floor(boxsize / cellsize).astype(int), 1)


def compute_power_spectrum(
    positions: np.ndarray,
    *,
    boxsize: float,
    meshsize: int | np.ndarray,
    ells: tuple[int, ...],
    los: str,
    k_step: float,
    output_fn: Path,
) -> None:
    from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles

    ps = PowerSpectrumMultipoles(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
    )
    ps.set_density_contrast(resampler="tsc", interlacing=3, compensate=True)
    ps.compute_spectrum(edges={"step": k_step}, ells=ells, los=los, save_fn=output_fn)


def compute_density_split_power(
    positions: np.ndarray,
    *,
    boxsize: float,
    meshsize: int | np.ndarray,
    smoothing_radius: float,
    nquantiles: int,
    ells: tuple[int, ...],
    los: str,
    k_step: float,
    pkqg_fn: Path,
    pkqq_fn: Path,
) -> None:
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    ds = DensitySplit(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
    )
    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=nquantiles, query_method="randoms")
    ds.quantile_data_power(
        positions,
        edges={"step": k_step},
        ells=ells,
        los=los,
        save_fn=pkqg_fn,
    )
    ds.quantile_power(
        edges={"step": k_step},
        ells=ells,
        los=los,
        save_fn=pkqq_fn,
    )


def write_metadata(filename: Path, metadata: dict[str, Any], overwrite: bool) -> None:
    if filename.exists() and not overwrite:
        logging.info("Skipping existing metadata file %s", filename)
        return
    tmp_filename = filename.with_name(filename.stem + ".tmp" + filename.suffix)
    tmp_filename.write_text(json.dumps(as_list(metadata), indent=2, sort_keys=True) + "\n")
    tmp_filename.replace(filename)


def measure_mockfactory_acm(
    *,
    bias: float = 2.0,
    redshift: float = 0.5,
    boxsize: float = 500.0,
    nbar: float = 1e-3,
    nmesh: int = 128,
    seed: int = 42,
    meshsize: int | np.ndarray = 128,
    cellsize: float = 3.9,
    smoothing_radius: float = 10.0,
    nquantiles: int = 5,
    los: str = "z",
    ells: tuple[int, ...] = (0, 2, 4),
    k_step: float = 0.001,
    save_dir: Path = Path("scripts/mockfactory_acm_measurements"),
    overwrite: bool = False,
    rsd: bool = True,
    measure_density_split: bool = True,
    fiducial_cosmology: str = "desi",
) -> dict[str, Path]:
    """Generate a mockfactory catalog, measure ACM spectra, and return output paths."""
    prepend_local_sources()

    logger = logging.getLogger(__name__)

    from jax import config

    config.update("jax_enable_x64", True)

    save_dir = Path(save_dir)
    ells = tuple(ells)
    fiducial_cosmology, cosmo = get_fiducial_cosmology(fiducial_cosmology)
    paths = output_paths(save_dir, seed, fiducial_cosmology=fiducial_cosmology)
    active_paths = dict(paths) if measure_density_split else {key: paths[key] for key in ["spectrum", "metadata"]}
    product_keys = ("spectrum", "pkqg", "pkqq") if measure_density_split else ("spectrum",)
    if prepare_outputs(active_paths, overwrite=overwrite, product_keys=product_keys):
        return active_paths

    t0 = time.time()
    logger.info("Generating mockfactory LagrangianLinearMock with fiducial cosmology %s.", fiducial_cosmology)
    positions = generate_mock(
        bias=bias,
        redshift=redshift,
        boxsize=boxsize,
        nbar=nbar,
        nmesh=nmesh,
        seed=seed,
        los=los,
        rsd=rsd,
        fiducial_cosmology=fiducial_cosmology,
    )
    logger.info("Generated %d galaxies in %.2f s.", len(positions), time.time() - t0)

    logger.info("Measuring galaxy power spectrum.")
    compute_power_spectrum(
        positions,
        boxsize=boxsize,
        meshsize=meshsize,
        ells=ells,
        los=los,
        k_step=k_step,
        output_fn=paths["spectrum"],
    )

    ds_meshsize = None
    if measure_density_split:
        ds_meshsize = density_split_meshsize(boxsize, cellsize, meshsize)
        logger.info("Measuring density-split power spectra.")
        compute_density_split_power(
            positions,
            boxsize=boxsize,
            meshsize=ds_meshsize,
            smoothing_radius=smoothing_radius,
            nquantiles=nquantiles,
            ells=ells,
            los=los,
            k_step=k_step,
            pkqg_fn=paths["pkqg"],
            pkqq_fn=paths["pkqq"],
        )

    metadata = {
        "bias": bias,
        "redshift": redshift,
        "boxsize": boxsize,
        "nbar": nbar,
        "nmesh": nmesh,
        "seed": seed,
        "fiducial_cosmology": fiducial_cosmology,
        "cosmology": cosmology_metadata(cosmo),
        "rsd": rsd,
        "rsd_growth_rate": growth_rate(redshift, fiducial_cosmology=fiducial_cosmology) if rsd else None,
        "measure_density_split": measure_density_split,
        "galaxy_count": len(positions),
        "meshsize": meshsize,
        "density_split_meshsize": ds_meshsize,
        "cellsize": cellsize,
        "smoothing_radius": smoothing_radius,
        "nquantiles": nquantiles,
        "ells": ells,
        "los": los,
        "k_step": k_step,
        "output_files": active_paths,
    }
    write_metadata(paths["metadata"], metadata, overwrite=overwrite)
    logger.info("Done. Outputs written under %s", save_dir)
    return active_paths


def main() -> None:
    from acm.utils.logging import setup_logging

    setup_logging()
    args = get_cli_args()
    measurement_meshsize = parse_meshsize(args.meshsize)
    measure_mockfactory_acm(
        bias=args.bias,
        redshift=args.redshift,
        boxsize=args.boxsize,
        nbar=args.nbar,
        nmesh=args.nmesh,
        seed=args.seed,
        fiducial_cosmology=args.fiducial_cosmology,
        meshsize=measurement_meshsize,
        cellsize=args.cellsize,
        smoothing_radius=args.smoothing_radius,
        nquantiles=args.nquantiles,
        los=args.los,
        ells=tuple(args.ells),
        k_step=args.k_step,
        save_dir=args.save_dir,
        overwrite=args.overwrite,
        rsd=not args.no_rsd,
        measure_density_split=not args.no_density_split,
    )


if __name__ == "__main__":
    main()
