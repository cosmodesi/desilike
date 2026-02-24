# mike_data_tools.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Base path to the measurements (your local copy on holylogin05)
# You can override with environment variable MIKE_DATA_PATH.
# -----------------------------------------------------------------------------
DEFAULT_PATH = Path("/n/home12/cgarciaquintero/DESI/GQC_mocks/cubic_boxes")
path_mike = Path(os.environ.get("MIKE_DATA_PATH", str(DEFAULT_PATH))).expanduser().resolve()

# -----------------------------------------------------------------------------
# Defaults used by ExtractData (kept for backwards compatibility)
# -----------------------------------------------------------------------------
pole_selection = [True, True, True, True, True]
range1 = [0.00, 100.0]
ranges = [range1, range1, range1, range1, range1]


# -----------------------------------------------------------------------------
# Helper: build file paths
# -----------------------------------------------------------------------------
def _mock_dirs(mocks: str, tracer: str, z_string: str) -> tuple[Path, Path]:
    """
    Return (path_pk, path_bisp) for a given mocks type.
    Layout expected:
      {path_mike}/{mocks}/{folder_box}/{tracer}/{z}/diag/powspec/
      {path_mike}/{mocks}/{folder_box}/{tracer}/{z}/diag/bispec/
    """
    if mocks not in ("EZmock", "AbacusSummit"):
        raise ValueError(f"mocks must be 'EZmock' or 'AbacusSummit', got {mocks!r}")

    folder_box = "CubicBox_6Gpc" if mocks == "EZmock" else "CubicBox"
    base = path_mike / mocks / folder_box / tracer / z_string / "diag"
    return base / "powspec", base / "bispec"


def _load_first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate files exist:\n  " + "\n  ".join(str(p) for p in candidates))


# -----------------------------------------------------------------------------
# Public API used by your runner script
#   - ExtractDataAbacusSummit
#   - ExtractDataEZmock
#   - covariance
#   - pole_k_selection
# -----------------------------------------------------------------------------
def ExtractDataEZmock(tracer: str, z_string: str, path_mike_override: str | Path | None = None):
    """
    Load EZmock measurements:
      - 2000 mocks
      - returns (k_eff_all, pkl0, pkl2, pkl4, B000, B202)
    """
    global path_mike
    if path_mike_override is not None:
        path_mike = Path(path_mike_override).expanduser().resolve()

    mocks = "EZmock"
    path_pk, path_bisp = _mock_dirs(mocks, tracer, z_string)

    nummocks = 2000

    # Determine binning from first file
    seed1 = f"{1:04d}"
    fB000 = _load_first_existing(
        path_bisp / f"bk000_diag_{tracer}_{z_string}_seed{seed1}",
        path_bisp / f"bk000_diag_{tracer}_{z_string}_seed{seed1}.txt",
    )
    fileB000 = np.loadtxt(fB000, unpack=True)
    k_eff_all = np.asarray(fileB000[1])
    nbins = k_eff_all.size

    # Allocate
    pkl0 = np.zeros((nummocks, nbins))
    pkl2 = np.zeros((nummocks, nbins))
    pkl4 = np.zeros((nummocks, nbins))
    B000 = np.zeros((nummocks, nbins))
    B202 = np.zeros((nummocks, nbins))

    for i in range(nummocks):
        j = i + 1
        seed = f"{j:04d}"

        fB000 = _load_first_existing(
            path_bisp / f"bk000_diag_{tracer}_{z_string}_seed{seed}",
            path_bisp / f"bk000_diag_{tracer}_{z_string}_seed{seed}.txt",
        )
        fB202 = _load_first_existing(
            path_bisp / f"bk202_diag_{tracer}_{z_string}_seed{seed}",
            path_bisp / f"bk202_diag_{tracer}_{z_string}_seed{seed}.txt",
        )
        fp0 = _load_first_existing(
            path_pk / f"pk0_{tracer}_{z_string}_seed{seed}",
            path_pk / f"pk0_{tracer}_{z_string}_seed{seed}.txt",
        )
        fp2 = _load_first_existing(
            path_pk / f"pk2_{tracer}_{z_string}_seed{seed}",
            path_pk / f"pk2_{tracer}_{z_string}_seed{seed}.txt",
        )
        fp4 = _load_first_existing(
            path_pk / f"pk4_{tracer}_{z_string}_seed{seed}",
            path_pk / f"pk4_{tracer}_{z_string}_seed{seed}.txt",
        )

        fileB000 = np.loadtxt(fB000, unpack=True)
        fileB202 = np.loadtxt(fB202, unpack=True)
        filep0 = np.loadtxt(fp0, unpack=True)
        filep2 = np.loadtxt(fp2, unpack=True)
        filep4 = np.loadtxt(fp4, unpack=True)

        # Conventions in your original code:
        #   - k on row 1
        #   - B multipoles on row 6
        #   - P multipoles on row 3
        B000[i] = fileB000[6]
        B202[i] = fileB202[6]
        pkl0[i] = filep0[3]
        pkl2[i] = filep2[3]
        pkl4[i] = filep4[3]

    return k_eff_all, pkl0, pkl2, pkl4, B000, B202


def ExtractDataAbacusSummit(tracer: str, z_string: str, path_mike_override: str | Path | None = None, subtract_shot: bool = False):
    """
    Load AbacusSummit measurements:
      - 25 phases (ph000..ph024)
      - returns (k_eff_all, pkl0, pkl2, pkl4, B000, B202)
    """
    global path_mike
    if path_mike_override is not None:
        path_mike = Path(path_mike_override).expanduser().resolve()

    mocks = "AbacusSummit"
    path_pk, path_bisp = _mock_dirs(mocks, tracer, z_string)

    nummocks = 25

    # Determine binning from first file
    seed0 = f"{0:03d}"
    fB0000 = _load_first_existing(
        path_bisp / f"bk000_diag_{tracer}_{z_string}_ph{seed0}",
        path_bisp / f"bk000_diag_{tracer}_{z_string}_ph{seed0}.txt",
    )
    fileB0000 = np.loadtxt(fB0000, unpack=True)
    k_eff_all = np.asarray(fileB0000[1])
    nbins = k_eff_all.size

    # Allocate
    pkl0 = np.zeros((nummocks, nbins))
    pkl2 = np.zeros((nummocks, nbins))
    pkl4 = np.zeros((nummocks, nbins))
    B000 = np.zeros((nummocks, nbins))
    B202 = np.zeros((nummocks, nbins))

    for j in range(nummocks):
        seed = f"{j:03d}"

        fB000 = _load_first_existing(
            path_bisp / f"bk000_diag_{tracer}_{z_string}_ph{seed}",
            path_bisp / f"bk000_diag_{tracer}_{z_string}_ph{seed}.txt",
        )
        fB202 = _load_first_existing(
            path_bisp / f"bk202_diag_{tracer}_{z_string}_ph{seed}",
            path_bisp / f"bk202_diag_{tracer}_{z_string}_ph{seed}.txt",
        )
        fp0 = _load_first_existing(
            path_pk / f"pk0_{tracer}_{z_string}_ph{seed}",
            path_pk / f"pk0_{tracer}_{z_string}_ph{seed}.txt",
        )
        fp2 = _load_first_existing(
            path_pk / f"pk2_{tracer}_{z_string}_ph{seed}",
            path_pk / f"pk2_{tracer}_{z_string}_ph{seed}.txt",
        )
        fp4 = _load_first_existing(
            path_pk / f"pk4_{tracer}_{z_string}_ph{seed}",
            path_pk / f"pk4_{tracer}_{z_string}_ph{seed}.txt",
        )

        fileB000 = np.loadtxt(fB000, unpack=True)
        fileB202 = np.loadtxt(fB202, unpack=True)
        filep0 = np.loadtxt(fp0, unpack=True)
        filep2 = np.loadtxt(fp2, unpack=True)
        filep4 = np.loadtxt(fp4, unpack=True)

        # Optional shot-noise subtraction (as in your original code)
        noiseB000 = 0.0
        noiseB202 = 0.0
        noisel0 = 0.0
        noisel2 = 0.0
        noisel4 = 0.0
        if subtract_shot:
            # Your original indices: pk noise at row 5, B noise at row 8
            noisel0 = filep0[5]
            noisel2 = filep2[5]
            noisel4 = filep4[5]
            noiseB000 = fileB000[8]
            noiseB202 = fileB202[8]

        B000[j] = fileB000[6] - noiseB000
        B202[j] = fileB202[6] - noiseB202
        pkl0[j] = filep0[3] - noisel0
        pkl2[j] = filep2[3] - noisel2
        pkl4[j] = filep4[3] - noisel4

    return k_eff_all, pkl0, pkl2, pkl4, B000, B202


def covariance(k, pkl0, pkl2, pkl4, B000, B202, Nscaling: float = 1.0):
    """
    Build covariance matrix from mocks arrays.

    Inputs:
      pkl0, pkl2, pkl4, B000, B202 : arrays shaped (Nmocks, Nk)
      k : array shaped (Nk,)

    Returns:
      k_cov : concatenated k for all poles (length 5*Nk)
      mean_data : mean vector (length 5*Nk)
      cov : covariance matrix (5*Nk, 5*Nk)

    Notes:
      - Multiplies by 27 to scale EZmock 6Gpc^3 -> Abacus 2Gpc^3 volume convention,
        matching your original comment.
      - Divides by Nscaling (e.g. volume scaling knob you used in the runner).
    """
    data = np.concatenate((pkl0.T, pkl2.T, pkl4.T, B000.T, B202.T)).T  # (Nmocks, 5*Nk)
    k_cov = np.concatenate((k, k, k, k, k))

    cov = np.cov(data, rowvar=False)
    cov = cov * 27.0
    cov = cov / float(Nscaling)

    mean_data = np.mean(data, axis=0)
    return k_cov, mean_data, cov


def pole_k_selection(ks_vector, pole_selection, ranges):
    """
    Select entries in a stacked vector ordered as:
      [P0(k), P2(k), P4(k), B000(k), B202(k)]
    where each block has the same k-grid (ks_vector[0:Nk]).

    ks_vector: array length 5*Nk (usually k repeated 5 times)
    pole_selection: list[bool] length 5
    ranges: list[[kmin,kmax]] length 5
    """
    ks_vector = np.asarray(ks_vector)
    if len(pole_selection) != 5 or len(ranges) != 5:
        raise ValueError("pole_selection and ranges must have length 5.")

    if ks_vector.size % 5 != 0:
        raise ValueError(f"ks_vector length must be multiple of 5, got {ks_vector.size}")

    nk = ks_vector.size // 5
    fit_selection = np.repeat(np.asarray(pole_selection, dtype=bool), nk)

    krange = ks_vector[:nk]
    p0_range = (ranges[0][0] < krange) & (krange < ranges[0][1])
    p2_range = (ranges[1][0] < krange) & (krange < ranges[1][1])
    p4_range = (ranges[2][0] < krange) & (krange < ranges[2][1])
    b0_range = (ranges[3][0] < krange) & (krange < ranges[3][1])
    b2_range = (ranges[4][0] < krange) & (krange < ranges[4][1])

    range_selection = np.concatenate((p0_range, p2_range, p4_range, b0_range, b2_range))
    return fit_selection & range_selection


# -----------------------------------------------------------------------------
# Optional convenience wrapper (not required by your main runner, but fixed)
# -----------------------------------------------------------------------------
def ExtractData(tracer, z_string, ranges=ranges, pole_selection=pole_selection, subtract_shot=False):
    """
    Convenience wrapper: mean Abacus data + EZmock covariance, applying selection.

    Returns a dict with:
      mask, data_vector, cov_array, k_eff_Abacus_all, k_eff_EZmocks_all
    """
    k_all, pl0_, pl2_, pl4_, B000_, B202_ = ExtractDataAbacusSummit(tracer, z_string, subtract_shot=subtract_shot)
    data_ = np.concatenate((pl0_.T, pl2_.T, pl4_.T, B000_.T, B202_.T)).T
    data_vector_all = np.mean(data_, axis=0)

    k_cov, pl0_cov, pl2_cov, pl4_cov, B000_cov, B202_cov = ExtractDataEZmock(tracer, z_string)
    k_cov_all, mean_ezmocks_all, cov_array_all = covariance(k_cov, pl0_cov, pl2_cov, pl4_cov, B000_cov, B202_cov)

    mask = pole_k_selection(k_cov_all, pole_selection, ranges)

    data_vector = data_vector_all[mask]
    cov_array = cov_array_all[np.ix_(mask, mask)]

    return {
        "mask": mask,
        "data_vector": data_vector,
        "cov_array": cov_array,
        "k_eff_Abacus_all": k_all,
        "k_eff_EZmocks_all": k_cov_all,
    }
