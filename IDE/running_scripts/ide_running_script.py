#!/usr/bin/env python3

# Adapted from MG/running_scripts/run_desilike_validations.py

"""
Run DESI-like Pk(+Bk) chains with fkptjax + IDE parameters.

Supported mock types (match full_shape.py):
  - cubic     : Table 2 settings + Mike cubic-box measurements/cov (mike_data_tools.py)
  - cutsky    : Table 1 settings (data loader NOT wired here yet)
  - synthetic : file-based noiseless vectors/cov (no windowing, no EZ cov)

FIX (THIS VERSION):
- Emulator is trained ONLY on cosmology + MG parameters.
- Emulator is built from the PT calculator object (ps_theory.pt), so the emulator learns PT internals
  (sigma8, fsigma8, AP quantities, fk/tables, etc.) and NOT nuisance-dependent final 'power'.
- For cubic ONLY, enforce PT k-grid = kin before training/using emu.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from glob import glob

import numpy as np
from mpi4py import MPI

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
os.environ["FOLPS_BACKEND"] = "jax"

# mike_data_tools must define:
#   ExtractDataAbacusSummit, ExtractDataEZmock, covariance, pole_k_selection
from mike_data_tools import *  # noqa: F401,F403

from desilike import setup_logging, parameter
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI

from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.samplers import MCMCSampler

from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine
from desilike.observables import ObservableCovariance

from desilike.profilers import MinuitProfiler
import json

# These must exist in your branch
from desilike.theories.galaxy_clustering import (
    fkptjaxTracerPowerSpectrumMultipoles,
    fkptjaxTracerBispectrumMultipoles,
)

# -----------------------------------------------------------------------------
# Redshifts
# -----------------------------------------------------------------------------
CUTSKY_ZEFF = {  # Table 1
    "BGS": 0.295,
    "LRG1": 0.510,
    "LRG2": 0.706,
    "LRG3": 0.922,
    "ELG1": 0.955,
    "ELG2": 1.321,
    "QSO": 1.484,
}

CUBIC_ZEFF = {  # Table 2
    "ABACUS_MC1_LRG": 0.800,
    "ABACUS_MC1_ELG": 0.950,
    "ABACUS_MC1_QSO": 1.400,
    "ABACUS_MC2_LRG": 0.725,
    "MC4_LRG_0P5": 0.500,
    "MC4_LRG_0P7": 0.700,
    "MC4_LRG_0P95": 0.950,
}

SYNTHETIC_ZEFF = {  # consistent with fkptTracerPowerSpectrumMultipoles._SYNTHETIC
    "BGS": 0.295,
    "LRG1": 0.510,
    "LRG2": 0.706,
    "LRG3": 0.934,
    "ELG": 1.321,
    "QSO": 1.484,
}

# NEW: WeiLiu (formatted EZmock cubic-box, rescaled to 2 Gpc/h)
WEILIU_ZEFF = {
    "BGS": 0.200,
    "LRG1": 0.500,
    "LRG2": 0.800,
    "ELG1": 0.950,
    "ELG2": 1.325,
    "QSO": 1.400,
}

WEILIU_ONLY_TRACERS = {"BGS", "LRG1", "ELG1", "ELG2"}  # the ones you want from WeiLiu

def apply_alpha_analytic_marginalization(theory):
    """Mark alphas as analytically marginalized (derived='.marg')."""
    is_phys = bool(getattr(theory, "is_physical_prior", False))
    suffix = "p" if is_phys else ""

    def pname(base: str) -> str:
        return f"{base}{suffix}"

    alpha_bases = ["alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot"]
    for par in theory.params.select(basename=[pname(b) for b in alpha_bases]):
        if par.varied:
            par.update(derived=".marg")
    return suffix


def allowed_tracers_for(mock_type: str) -> set[str]:
    mt = str(mock_type).lower()
    if mt == "cutsky":
        return set(CUTSKY_ZEFF.keys())
    if mt == "cubic":
        return set(CUBIC_ZEFF.keys())
    if mt == "synthetic":
        return set(SYNTHETIC_ZEFF.keys())
    if mt == "weiliu":
        return set(WEILIU_ZEFF.keys())
    if mt in ("cubic+weiliu", "mixed"):
        # allow both families; we'll decide per tracer
        return set(CUBIC_ZEFF.keys()) | set(WEILIU_ZEFF.keys())
    raise ValueError(f"Unknown mock_type={mock_type!r} (expected cubic/cutsky/synthetic).")


def z_eff_for(tracer: str, mock_type: str) -> float:
    mt = str(mock_type).lower()
    tr = str(tracer).upper()
    if mt == "cutsky":
        return float(CUTSKY_ZEFF[tr])
    if mt == "cubic":
        return float(CUBIC_ZEFF[tr])
    if mt == "synthetic":
        return float(SYNTHETIC_ZEFF[tr])
    if mt == "weiliu":
        return float(WEILIU_ZEFF[tr])
    if mt in ("cubic+weiliu", "mixed"):
        if tr in WEILIU_ZEFF:
            return float(WEILIU_ZEFF[tr])
        if tr in CUBIC_ZEFF:
            return float(CUBIC_ZEFF[tr])
    raise ValueError(f"Unknown mock_type={mock_type!r} (expected cubic/cutsky/synthetic).")


def mike_tracer_name(tracer: str, mock_type: str) -> str:
    """Map cubic tracer key to Mike family name (LRG/ELG/QSO)."""
    mt = str(mock_type).lower()
    tr = str(tracer).upper()
    if mt != "cubic":
        return tr

    if tr.endswith("_LRG") or "LRG" in tr:
        return "LRG"
    if tr.endswith("_ELG") or "ELG" in tr:
        return "ELG"
    if tr.endswith("_QSO") or "QSO" in tr:
        return "QSO"
    raise ValueError(f"[cubic] Could not map tracer={tr!r} to Mike tracer family (LRG/ELG/QSO).")

# -----------------------------------------------------------------------------
# WeiLiu helpers (file-based, like synthetic but coming from full_shape.py loader)
# -----------------------------------------------------------------------------
def load_weiliu_vector_and_covariance(
    tracer: str,
    *,
    kmin_cut: float = 0.02,
    kmax_cut: float = 0.30,
    requested_ells: tuple[int, ...] = (0, 2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load WeiLiu formatted Pk multipoles for tracer:
      returns data (stacked by ell blocks), cov (matching data), and k-vector.

    The underlying files contain (k, P0, P2, P4) and a (3Nk x 3Nk) covariance.
    We apply k cuts and then select the requested ell blocks.
    """
    tr = str(tracer).upper()

    # Uses your classmethod added in full_shape.py
    k_all, data_all, cov_all = fkptjaxTracerPowerSpectrumMultipoles._load_weiliu_pk(tr)  # noqa

    k_all = np.asarray(k_all, dtype=float).reshape(-1)
    data_all = np.asarray(data_all, dtype=float).reshape(-1)
    cov_all = np.asarray(cov_all, dtype=float)

    present_ells = (0, 2, 4)
    Nk = k_all.size
    if data_all.size != 3 * Nk:
        raise ValueError(f"[weiliu] data size mismatch: got {data_all.size}, expected {3*Nk} (Nk={Nk}).")
    if cov_all.shape != (3 * Nk, 3 * Nk):
        raise ValueError(f"[weiliu] cov shape mismatch: got {cov_all.shape}, expected {(3*Nk, 3*Nk)}.")

    # k mask
    m = (k_all >= float(kmin_cut)) & (k_all <= float(kmax_cut))
    if not np.any(m):
        raise ValueError(f"[weiliu] Empty k-range after cuts: [{kmin_cut},{kmax_cut}] on tracer={tr}.")

    # slice each ell block by k-mask
    blocks = {}
    for i, ell in enumerate(present_ells):
        blocks[ell] = data_all[i * Nk : (i + 1) * Nk][m]

    # build index list for covariance selection (matching the flattened ordering)
    idx = []
    for i, ell in enumerate(present_ells):
        ii = np.where(m)[0]
        idx.append(i * Nk + ii)
    idx = np.concatenate(idx).astype(int)

    cov_cut = cov_all[np.ix_(idx, idx)]

    # now select only requested ell blocks (still flattened)
    req = tuple(int(e) for e in requested_ells)
    for e in req:
        if e not in present_ells:
            raise ValueError(f"[weiliu] requested ell={e} not in present_ells={present_ells}.")

    # build final data and covariance by ell-block selection
    # order is requested_ells blocks, each of length Nk_cut
    k_cut = k_all[m]
    Nk_cut = k_cut.size

    # mapping ell -> block index in (0,2,4)
    order_map = {ell: j for j, ell in enumerate(present_ells)}

    data_out = np.concatenate([blocks[e] for e in req], axis=0)

    # covariance block extraction
    # cov_cut is ordered as [P0(kcut), P2(kcut), P4(kcut)] blocks
    sel = []
    for e in req:
        j = order_map[e]
        sel.extend(range(j * Nk_cut, (j + 1) * Nk_cut))
    sel = np.array(sel, dtype=int)

    cov_out = cov_cut[np.ix_(sel, sel)]
    return data_out, cov_out, k_cut

def build_weiliu_observables_for_tracer(
    tracer: str,
    ps_theory,
    *,
    ells: tuple[int, ...],
    kmin_cut: float = 0.02,
    kmax_cut: float = 0.30,
):
    """
    WeiLiu provides only P(k) multipoles (P0,P2,P4).
    Treat like unwindowed (no wmatrix), similar to synthetic.
    """
    if tuple(ells) not in ((0, 2), (0, 2, 4)):
        raise ValueError(f"[weiliu] Supported --ells are only '0,2' or '0,2,4'. Got {ells!r}.")

    data, cov, k = load_weiliu_vector_and_covariance(
        tracer,
        kmin_cut=kmin_cut,
        kmax_cut=kmax_cut,
        requested_ells=ells,
    )

    obs = TracerPowerSpectrumMultipolesObservable(
        data=data,
        covariance=cov,
        theory=ps_theory,
        ells=list(ells),
        k=k,
        # IMPORTANT: no windowing for WeiLiu formatted files
    )
    return [obs], cov


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--tracers",
        nargs="+",
        default=["BGS", "LRG1", "LRG2", "LRG3", "ELG", "QSO"],
        help="List of tracers. For cubic: ABACUS_MC2_LRG, MC4_LRG_0P7, ... "
             "For synthetic: LRG2, QSO, ... "
             "For cutsky: BGS, LRG1, LRG2, LRG3, ELG1, ELG2, QSO "
             "For weiliu: BGS, LRG1, LRG2, ELG1, ELG2, QSO",
    )

    p.add_argument(
        "--add-bispectrum",
        action="store_true",
        help="Include bispectrum multipoles B000,B202 (Pk+Bk). (cubic only)",
    )

    p.add_argument(
        "--prior_basis",
        default="standard",
        choices=["standard", "physical", "physical_velocileptors", "APscaling", "standard_folps", "physical_folps"],
        help="fkpt* prior basis aliasing is handled in the class.",
    )

    p.add_argument("--freedom", default="max", choices=["none", "min", "max"], help="Priors freedom knob (min/max)")
    p.add_argument("--model", default="IDE", help="IDE model label used by fkptjax)")
    
    # Adding options for IDE variant
    p.add_argument(
        "--ide-variant",
        default="IDEModel1",
        choices=["IDEModel1", "IDEModel2", "DS"],
        help="IDE variant",
    )

    p.add_argument(
        "--mock-type",
        default="cubic",
        choices=["cubic", "cutsky", "synthetic", "weiliu", "cubic+weiliu"],
        help="cubic: Mike cubic-box data+EZ cov; cutsky: not wired here; "
             "synthetic: file-based vectors/cov; "
             "weiliu: formatted EZmock vectors/cov (P0/P2/P4) from WEILIU_FORMATTED_DIR.",
    )

    p.add_argument("--ells", type=str, default="0,2", help="Comma-separated subset of 0,2,4.")

    # ---- synthetic-only inputs (file-based) ----
    p.add_argument("--fid-model", type=str, default="LCDM", help="Synthetic filename tag, e.g. LCDM, HS_F4, ...")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/global/homes/n/nishavk/desilike/IDE/synthetic_data/"),
        help="Directory containing {TRACER}_{FIDMODEL}_k.txt, _P0P2P4.txt, _cov.txt",
    )
    p.add_argument("--kmin-cut", type=float, default=0.02, help="Synthetic: kmin cut.")
    p.add_argument("--kmax-cut", type=float, default=0.30, help="Synthetic: kmax cut.")
    p.add_argument("--use-rescaled-cov", action="store_true", help="Synthetic: use *_cov_x0p04.txt instead of *_cov.txt")
    p.add_argument("--cov-scale", type=float, default=1.0, help="Synthetic: multiply covariance by this factor.")

    # Model switches
    p.add_argument("--beyond_eds", action="store_true", help="Use beyond-EdS kernels")
    p.add_argument("--rescale_PS", action="store_true", help="Rescale P(k) with growth ratio (HDKI only)")

    # k cuts for cubic-windowed pipeline
    p.add_argument("--kr_max", type=float, default=0.20)
    p.add_argument("--kr_b0_max", type=float, default=0.12)
    p.add_argument("--kr_b2_max", type=float, default=0.08)

    # OLD-script cosmology prior toggles
    p.add_argument("--skip-ns-prior", action="store_true", help="Do not apply a prior on n_s.")
    p.add_argument("--skip-bbn-prior", action="store_true", help="Do not apply BBN prior on omega_b.")

    # adding flag to force LCDM
    p.add_argument("--force-LCDM", action="store_true", help="Force LCDM: fix IDE params to LCDM values regardless of variant.")

    # Output / running
    p.add_argument("--chain_name", type=str, default="./chains/chain_IDE.npy")
    p.add_argument("--restart", action="store_true", help="Resume chains if matching files exist.")
    p.add_argument("--nchains", type=int, default=4, help="Number of chains (MPI-friendly) when not restarting.")
    p.add_argument("--ref-scale", type=float, default=1.2)
    p.add_argument("--check-every", type=int, default=3000)
    p.add_argument("--max-iter", type=int, default=50000)

    p.add_argument("--run_chains", action="store_true")
    p.add_argument("--test", action="store_true")

    # Test plotting
    p.add_argument("--plot-test", action="store_true", help="With --test, also save data-vs-theory plots.")
    p.add_argument("--plot-dir", type=Path, default=Path("./test_plots"), help="Output directory for --plot-test.")
    p.add_argument("--no-jit-test", action="store_true", help="With --test, do not JIT (useful for debugging).")

    # Emulator knobs
    g = p.add_mutually_exclusive_group()
    g.add_argument("--create-emu", action="store_true", help="Build per-tracer emulator(s) then exit.")
    g.add_argument("--use-emu", action="store_true", help="Load and use per-tracer emulator(s).")
    p.add_argument("--emu-dir", type=Path, default=Path("./emulators"))
    p.add_argument("--emu-order-lcdm", type=int, default=4, help="Taylor order for non-IDE params.")
    p.add_argument("--emu-order-ide", type=int, default=8, help="Taylor order for IDE params.")

    # Best-fit (MAP/profile) run
    p.add_argument("--run-bestfit", action="store_true", help="Run MinuitProfiler (MAP / best-fit) instead of MCMC.")
    p.add_argument("--profiles-out", type=Path, default=None, help="Output .npy for profiler (default next to chain_name).")
    p.add_argument("--bestfit-out", type=Path, default=None, help="Output JSON with best-fit parameter values.")
    p.add_argument("--max-calls", type=int, default=20000, help="Max calls/iterations for Minuit.")
    p.add_argument("--gradient", action="store_true", help="Use gradients if available.")
    p.add_argument("--rescale", action="store_true", help="Rescale parameters in profiler.")
    p.add_argument("--covariance", type=str, default=None, help="Covariance option passed to MinuitProfiler (string).")
    p.add_argument("--niterations", type=int, default=4, help="Number of independent Minuit starts (best-of-N).")

    return p.parse_args()


def parse_ells_str(s: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    allowed = {0, 2, 4}
    if not set(vals).issubset(allowed) or len(vals) == 0:
        raise ValueError(f"--ells must be a comma-separated subset of 0,2,4; got {s!r}")
    return tuple(sorted(vals))


def parse_prior(spec: str):
    parts = [x.strip() for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Bad prior spec {spec!r}. Use 'uniform,lo,hi' or 'norm,mean,sigma'.")
    dist = parts[0]
    a = float(parts[1])
    b = float(parts[2])
    if dist == "uniform":
        return {"dist": "uniform", "limits": [a, b]}
    if dist in ("norm", "normal"):
        return {"dist": "norm", "loc": a, "scale": b}
    raise ValueError(f"Unknown dist {dist!r} in {spec!r}.")


# -----------------------------------------------------------------------------
# OLD-style prior plumbing
# -----------------------------------------------------------------------------
def add_or_update_param(
    cosmo: Cosmoprimo,
    name: str,
    value: float,
    fixed: bool,
    prior: dict | None = None,
    ref: dict | None = None,
    delta: float | None = None,
):
    if name not in cosmo.init.params:
        cosmo.init.params.data.append(parameter.Parameter(basename=name, value=float(value), fixed=bool(fixed)))
    upd = dict(value=float(value), fixed=bool(fixed))
    if prior is not None:
        upd["prior"] = prior
    if ref is not None:
        upd["ref"] = ref
    if delta is not None:
        upd["delta"] = float(delta)
    cosmo.init.params[name].update(**upd)


def apply_oldstyle_cosmo_and_ide_priors(cosmo: Cosmoprimo, args):
    # Baseline fixes
    if "tau_reio" in cosmo.init.params:
        cosmo.init.params["tau_reio"].update(fixed=True)

    if "N_eff" in cosmo.init.params:
        cosmo.init.params["N_eff"].update(fixed=True, value=3.046)
    if "m_ncdm" in cosmo.init.params:
        cosmo.init.params["m_ncdm"].update(fixed=True, value=0.06)

    ns_prior = None if args.skip_ns_prior else {"dist": "norm", "loc": 0.9649, "scale": 0.02}
    bbn_prior = None if args.skip_bbn_prior else {"dist": "norm", "loc": 0.02237, "scale": 0.00055}

    if ns_prior is not None and "n_s" in cosmo.init.params:
        cosmo.init.params["n_s"].update(
            fixed=False,
            prior=ns_prior,
            ref={"dist": "norm", "loc": 0.9649, "scale": 0.004},
            delta=0.01,
        )

    if bbn_prior is not None and "omega_b" in cosmo.init.params:
        cosmo.init.params["omega_b"].update(
            fixed=False,
            prior=bbn_prior,
            delta=0.0015,
        )

    prior_limits = {"h": (0.4, 1.0), "omega_cdm": (0.001, 0.99), "logA": (1.61, 3.91)}
    for name, scale, delta in [("h", 0.001, 0.03), ("omega_cdm", 0.001, 0.007), ("logA", 0.001, 0.05)]:
        if name in cosmo.init.params:
            lo, hi = prior_limits[name]
            par = cosmo.init.params[name]
            par.update(
                fixed=False,
                prior={"dist": "uniform", "limits": (lo, hi)},
                ref={"dist": "norm", "loc": par.value, "scale": scale},
                delta=delta,
            )

    if "sigma8_m" in cosmo.init.params:
        cosmo.init.params["sigma8_m"].update(derived=True, latex=r"\sigma_8")

    force_lcdm = bool(args.force_LCDM)
    
    if args.ide_variant == "IDEModel1":
        add_or_update_param(
            cosmo,
            "beta",
            value=0.0,
            fixed=force_lcdm,
            prior=None if force_lcdm else {"dist": "uniform", "limits": (-1.0, 1.0)},
            ref=None if force_lcdm else {"dist": "norm", "loc": 0.0, "scale": 0.01},
            delta=None if force_lcdm else 0.25,
        )
        
    # for thawing models, need to add w, wp etc as well + DS parameters for DS model
    
# -----------------------------------------------------------------------------
# Emulator helpers
# -----------------------------------------------------------------------------
def emulator_engine_for(ide_variant: str, order_lcdm: int = 4, order_ide: int = 8) -> TaylorEmulatorEngine:
    order = {"*": int(order_lcdm)}

    if ide_variant == "IDEModel1":
        order["beta"] = int(order_ide)
 
    return TaylorEmulatorEngine(method="finite", order=order)


def emu_filename(
    tracer: str,
    kr_max: float,
    ide_variant: str,
    beyond_eds: bool,
    add_bispectrum: bool,
    mock_type: str,
    ells: tuple[int, ...],
    *,
    nkin: int | None = None,
) -> str:
    mode = ide_variant
    if not beyond_eds:
        mode = "eds_" + mode
    spec = "pkbk" if add_bispectrum else "pk"
    elltag = "l" + "".join(str(e) for e in ells)  # l02 or l024
    nk = f"_nkin{int(nkin)}" if (nkin is not None and str(mock_type).lower() == "cubic") else ""
    return f"emu-{mock_type}_fkptjax_{mode}_{tracer}_{elltag}_kmax{kr_max:.3f}{nk}_{spec}.npy"


def chain_pattern_from(chain_name: str) -> str:
    if "*" in chain_name:
        return chain_name
    p = Path(chain_name)
    if p.suffix == ".npy":
        return str(p.with_suffix("")) + "_*.npy"
    return chain_name + "_*.npy"


def _flat_size(x) -> int:
    return int(np.asarray(x).reshape(-1).size)


def _try_init_update(obj, **kwargs) -> bool:
    init = getattr(obj, "init", None)
    if init is None:
        return False
    if hasattr(init, "update"):
        try:
            init.update(**kwargs)
            return True
        except Exception:
            return False
    return False


def _set_pt_kgrid_for_cubic(pt_calc, kin: np.ndarray):
    """
    Force the PT evaluation grid to be `kin` for cubic windowed emulation.
    Tries pt.<k-like> = kin and pt.init.update(k-like=kin).
    """
    kin = np.asarray(kin).reshape(-1)
    candidate_names = ["kin", "k", "kgrid", "k_eval", "kpt", "k_array"]

    candidates = [pt_calc]
    for obj in list(candidates):
        for attr in ["power", "template", "engine", "calculator", "theory", "model"]:
            sub = getattr(obj, attr, None)
            if sub is not None:
                candidates.append(sub)

    for obj in candidates:
        for nm in candidate_names:
            if hasattr(obj, nm):
                try:
                    setattr(obj, nm, kin)
                    print(f"[cubic emu] Set attribute {obj.__class__.__name__}.{nm} (len={kin.size})")
                    return
                except Exception:
                    pass

    for obj in candidates:
        for nm in candidate_names:
            if _try_init_update(obj, **{nm: kin}):
                print(f"[cubic emu] Set init.{nm} for {obj.__class__.__name__} (len={kin.size})")
                return

    raise RuntimeError(
        "[cubic emu] Could not locate/set PT k-grid using attrs or init.update. "
        "Check where your PT stores its evaluation grid."
    )


def _allowed_emu_param_basenames(ide_variant: str) -> set[str]:
    allowed = {"h", "omega_cdm", "omega_b", "logA", "n_s"}
    if ide_variant == "IDEModel1":
        allowed.add("beta")
    return allowed


def _freeze_all_params_except(calc, allowed_basenames: set[str]) -> dict[str, bool]:
    """
    Freeze all parameters in calc.params except those in allowed_basenames.
    Returns {basename: old_fixed} for restoring.
    """
    saved: dict[str, bool] = {}
    for par in calc.params:
        bn = getattr(par, "basename", None)
        if bn is None:
            continue
        saved[bn] = bool(getattr(par, "fixed", False))
        if bn not in allowed_basenames:
            par.update(fixed=True)
    return saved


def _restore_fixed_flags(calc, saved: dict[str, bool]):
    for bn, was_fixed in saved.items():
        if bn in calc.params:
            calc.params[bn].update(fixed=bool(was_fixed))


# -----------------------------------------------------------------------------
# ObservableCovariance metadata helper
# -----------------------------------------------------------------------------
def covmeta_from_observables(obs_list):
    metas = []
    for obs in obs_list:
        ells = list(getattr(obs, "ells", []))
        k = getattr(obs, "k", None)

        if isinstance(k, (list, tuple)):
            x = [np.asarray(kk).reshape(-1) for kk in k]
        else:
            x = [np.asarray(k).reshape(-1) for _ in ells]

        # detect bispectrum-style projections like (l1,l2,l3)
        is_bk = any(isinstance(e, (tuple, list)) and len(e) == 3 for e in ells)
        name = "BispectrumMultipoles" if is_bk else "PowerSpectrumMultipoles"

        metas.append({"name": name, "x": x, "projs": ells})
    return metas

# -----------------------------------------------------------------------------
# Synthetic helpers (file-based)
# -----------------------------------------------------------------------------
def select_data_and_cov(
    Pvec: np.ndarray,
    cov: np.ndarray,
    present_ells=(0, 2, 4),
    requested_ells=(0, 2),
    start=0,
    Ncut=0,
) -> tuple[np.ndarray, np.ndarray]:
    Pvec = np.asarray(Pvec).reshape(-1)
    cov = np.asarray(cov)
    Nraw = Pvec.size // len(present_ells)
    order_map = {ell: i for i, ell in enumerate(present_ells)}
    idx: list[int] = []
    for ell in requested_ells:
        base = order_map[ell] * Nraw
        idx.extend(range(base + start, base + start + Ncut))
    idx = np.array(idx, dtype=int)
    return Pvec[idx], cov[np.ix_(idx, idx)]


# -----------------------------------------------------------------------------
# cubic data / covariance (Mike tools)
# -----------------------------------------------------------------------------
def load_cubic_vector_and_covariance(
    tracer: str,
    k_max: float,
    k_max_b0: float | None,
    k_max_b2: float | None,
    P4: bool = False,
):
    k_min = 0.02

    isP0, isP2, isP4 = True, True, bool(P4)
    isB000, isB202 = True, True

    if k_max_b0 is None and k_max_b2 is None:
        isB000, isB202 = False, False
        k_max_b0 = 0.08
        k_max_b2 = 0.08

    Vol = 1.0
    totsim = 2000

    z_ev = z_eff_for(tracer, "cubic")
    z_str = f"z{z_ev:.3f}"
    tracer_mike = mike_tracer_name(tracer, "cubic")

    k_eff, p0_ez, p2_ez, p4_ez, b000_ez, b202_ez = ExtractDataEZmock(tracer_mike, z_str)  # noqa: F405
    k_cov_all, mean_all, cov_all = covariance(  # noqa: F405
        k_eff, p0_ez, p2_ez, p4_ez, b000_ez, b202_ez, Nscaling=Vol
    )

    k_data_ab, p0_all, p2_all, p4_all, b000_all, b202_all = ExtractDataAbacusSummit(  # noqa: F405
        tracer_mike, z_str, subtract_shot=True
    )
    k_data = k_data_ab

    Pk0 = np.mean(p0_all, axis=0)
    Pk2 = np.mean(p2_all, axis=0)
    Pk4 = np.mean(p4_all, axis=0)
    B000 = np.mean(b000_all, axis=0)
    B202 = np.mean(b202_all, axis=0)

    pole_selection = [isP0, isP2, isP4, isB000, isB202]
    ranges = [
        [k_min, k_max],
        [k_min, k_max],
        [k_min, k_max],
        [k_min, float(k_max_b0)],
        [k_min, float(k_max_b2)],
    ]

    mask = pole_k_selection(k_cov_all, pole_selection, ranges)  # noqa: F405
    cov = cov_all[np.ix_(mask, mask)]

    k_points_pk = np.where((k_min < k_data) & (k_data < k_max) & isP0)
    k_points_b0 = np.where((k_min < k_data) & (k_data < float(k_max_b0)) & isB000)
    k_points_b2 = np.where((k_min < k_data) & (k_data < float(k_max_b2)) & isB202)

    kr_pk = k_data[k_points_pk]
    kr_b0 = k_data[k_points_b0]
    kr_b2 = k_data[k_points_b2]

    blocks = [Pk0[k_points_pk], Pk2[k_points_pk]]
    if isP4:
        blocks.append(Pk4[k_points_pk])
    if isB000:
        blocks.append(B000[k_points_b0])
    if isB202:
        blocks.append(B202[k_points_b2])  # <-- FIXED bracket
    data = np.concatenate(tuple(blocks))

    n_data = len(data)
    hartlap = (totsim - 1.0) / (totsim - n_data - 2.0)
    cov = cov * hartlap

    # Windowing machinery
    N_ck = max(int(k_max * 100) + 2, 25)
    k_thy = np.linspace(0.0, 0.01 * N_ck, 2 * N_ck * 5, endpoint=False) + 0.0025 + 0.0005
    ko = k_data[0 : 2 * N_ck]

    m_bin = np.zeros((len(ko), len(k_thy)))
    for i, _ki in enumerate(ko):
        norm = (1.0 / 3.0) * ((k_thy[5 * i + 4]) ** 3 - (k_thy[5 * i]) ** 3)
        for j in range(5):
            ff = (5 - 1) / 5
            m_bin[i, 5 * i + j] = (k_thy[5 * i + j] ** 2) * 0.001 / norm * ff

    return data, cov, kr_pk, kr_b0, kr_b2, k_thy, m_bin, k_points_pk, k_points_b0, k_points_b2


def build_cubic_observables_for_tracer(
    tracer: str,
    ps_theory,
    bs_theory,
    kr_max: float,
    kr_b0_max: float | None,
    kr_b2_max: float | None,
    include_bispectrum: bool,
    ells: tuple[int, ...],
):
    if tuple(ells) not in ((0, 2), (0, 2, 4)):
        raise ValueError(f"[cubic] Supported --ells are only '0,2' or '0,2,4'. Got {ells!r}.")

    want_p4 = (4 in ells)

    data, cov, kr_pk, kr_b0, kr_b2, k_thy, m_bin, k_points_pk, k_points_b0, k_points_b2 = (
        load_cubic_vector_and_covariance(tracer, kr_max, kr_b0_max, kr_b2_max, P4=want_p4)
    )

    from scipy.linalg import block_diag

    w_pk = m_bin[np.asarray(k_points_pk).ravel(), :]
    wmatrix_pk = block_diag(*([w_pk] * len(ells)))
    n_pk = len(ells) * len(kr_pk)

    ps_obs = TracerPowerSpectrumMultipolesObservable(
        data=data[:n_pk],
        covariance=cov[:n_pk, :n_pk],
        theory=ps_theory,
        kin=k_thy,
        ells=list(ells),
        k=kr_pk,
        wmatrix=wmatrix_pk,
    )

    if not include_bispectrum:
        return [ps_obs], cov, k_thy

    start = len(data) - (len(kr_b0) + len(kr_b2))
    w_b0 = m_bin[np.asarray(k_points_b0).ravel(), :]
    w_b2 = m_bin[np.asarray(k_points_b2).ravel(), :]
    wmatrix_bk = block_diag(w_b0, w_b2)

    bs_obs = TracerPowerSpectrumMultipolesObservable(
        data=data[start:],
        covariance=cov[start:, start:],
        theory=bs_theory,
        kin=k_thy,
        ells=[(0, 0, 0), (2, 0, 2)],
        k=[kr_b0, kr_b2],
        wmatrix=wmatrix_bk,
    )

    return [ps_obs, bs_obs], cov, k_thy

def plot_observables(obs_for_plot: dict[str, list], outdir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    def _to_1d(x):
        return np.asarray(x, dtype=float).reshape(-1)

    def _extract_pred_vector(obs, out):
        """
        Try hard to get the *flat* prediction vector matching obs.data layout.
        Handles ndarray / ObservableArray-like (with .power) / dict outputs.
        """
        # 1) already a numeric array
        if isinstance(out, (np.ndarray, list, tuple)):
            try:
                return _to_1d(out)
            except Exception:
                pass

        # 2) ObservableArray-like: often has .power
        if hasattr(out, "power"):
            try:
                return _to_1d(getattr(out, "power"))
            except Exception:
                pass

        # 3) dict outputs: common keys
        if isinstance(out, dict):
            for key in ("power", "theory", "model", "vector"):
                if key in out:
                    return _to_1d(out[key])
            # if it's a single-entry dict, use the only value
            if len(out) == 1:
                return _to_1d(next(iter(out.values())))
            raise RuntimeError(f"[plot] Don't know how to extract theory from dict keys={list(out.keys())}")

        # 4) fallback: some observables expose flattheory
        if hasattr(obs, "flattheory"):
            return _to_1d(getattr(obs, "flattheory"))

        # 5) give up with a helpful message
        raise RuntimeError(f"[plot] Could not extract prediction vector. out type={type(out)}")

    outdir.mkdir(parents=True, exist_ok=True)

    for tracer, obs_list in obs_for_plot.items():
        for iobs, obs in enumerate(obs_list):
            data = _to_1d(getattr(obs, "data"))

            out = obs()  # may be dict / ObservableArray / ndarray
            pred = _extract_pred_vector(obs, out)

            # sanity: must match for plotting in the same layout
            if pred.size != data.size:
                raise ValueError(
                    f"[plot] data/pred size mismatch for {tracer} obs#{iobs}: "
                    f"data={data.size}, pred={pred.size} (out type={type(out)})"
                )

            ells = list(getattr(obs, "ells", []))
            k = getattr(obs, "k", None)

            if isinstance(k, (list, tuple)):
                k_list = [np.asarray(kk).reshape(-1) for kk in k]
            else:
                k1 = np.asarray(k).reshape(-1)
                k_list = [k1 for _ in ells]

            th = getattr(obs, "theory", None)
            kind = "bk" if (th is not None and "Bispectrum" in th.__class__.__name__) else "pk"

            plt.figure()
            offset = 0
            for ell, kk in zip(ells, k_list):
                n = kk.size
                d = data[offset:offset + n]
                m = pred[offset:offset + n]
                offset += n

                plt.plot(kk, kk*d, marker="o", linestyle="none", label=f"data ℓ={ell}")
                plt.plot(kk, kk*m, linestyle="-", label=f"model ℓ={ell}")

            plt.xlabel("k [h/Mpc]")
            plt.ylabel("observable")
            plt.title(f"{tracer} {kind} (obs #{iobs})")
            plt.legend()

            fname = outdir / f"{tracer}_{kind}_obs{iobs}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[plot] saved {fname}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nmodes = int(args.test) + int(args.run_chains) + int(args.run_bestfit) + int(args.create_emu)
    if nmodes == 0:
        pass
    elif nmodes > 1:
        raise ValueError("Choose only one of: --test, --run_chains, --run-bestfit, --create-emu.")
    
    mt = str(args.mock_type).lower()
    
    freedom = None if args.freedom == "none" else str(args.freedom)

    if mt == "synthetic" and args.add_bispectrum:
        raise ValueError("mock_type='synthetic' currently supports P(k) only (no bispectrum).")
    if mt == "cutsky":
        raise NotImplementedError("mock_type='cutsky' not wired here yet.")

    req_ells = parse_ells_str(args.ells)
    if mt == "cubic" and tuple(req_ells) not in ((0, 2), (0, 2, 4)):
        raise ValueError(f"[cubic] Supported --ells only '0,2' or '0,2,4'. Got {req_ells!r}.")

    if rank == 0:
        try:
            import jax
            print("[JAX] backend:", jax.default_backend())
            print("[JAX] devices:", jax.devices())
            print("[JAX] x64:", jax.config.read("jax_enable_x64"))
        except Exception:
            pass

    tracers = [t.strip().upper() for t in args.tracers]
    allowed = allowed_tracers_for(mt)
    for t in tracers:
        if t not in allowed:
            raise ValueError(f"Unknown tracer {t!r} for mock_type={mt!r}. Allowed: {sorted(allowed)}")

    args.emu_dir.mkdir(parents=True, exist_ok=True)

    fid = DESI()

    # -------------------------
    # Cosmoprimo / ISiTIDE engine flags by IDE variant
    # -------------------------
    
    # default 
    dark_energy_model = "IDEModel1"
        
    if str(args.ide_variant) == "IDEModel1":
        dark_energy_model = str(args.ide_variant)
        
    try:
        cosmo = Cosmoprimo(
            engine="isitide",
            dark_energy_model=dark_energy_model,
        )
    except TypeError:
        raise SystemExit(
            "Cosmoprimo(...) did not accept ISiTIDE calculation flags "
            "(MG_parameterization/use_BZ_form/use_growth_index and/or binning flags). "
            "Abort as requested.\n"
        )

    for nm, latex in [("H0", None), ("Omega_m", None), ("sigma8_m", r"\sigma_8")]:
        if nm in cosmo.init.params:
            if latex is None:
                cosmo.init.params[nm].update(derived=True)
            else:
                cosmo.init.params[nm].update(derived=True, latex=latex)

    apply_oldstyle_cosmo_and_ide_priors(cosmo, args)

    likes = []
    obs_for_plot: dict[str, list] = {}

    for tracer in tracers:
        # Decide dataset source per tracer if mixed
        if mt in ("cubic+weiliu", "mixed"):
            mt_tr = "weiliu" if tracer in WEILIU_ONLY_TRACERS else "cubic"
        else:
            mt_tr = mt
        include_bk = bool(args.add_bispectrum) and (mt_tr == "cubic")
            
        z = z_eff_for(tracer, mt_tr)
        template = DirectPowerSpectrumTemplate(fiducial=fid, cosmo=cosmo, z=float(z))
        theory_variant = str(args.ide_variant)

        nkin_for_name: int | None = None
        kin_expected = None
        bs_theory = None

        if mt_tr == "synthetic":
            fid_model = str(args.fid_model)
            k_path = args.data_dir / f"{tracer}_{fid_model}_k.txt"
            p_path = args.data_dir / f"{tracer}_{fid_model}_P0P2P4.txt"
            c_suffix = "_cov_x0p04.txt" if args.use_rescaled_cov else "_cov.txt"
            c_path = args.data_dir / f"{tracer}_{fid_model}{c_suffix}"

            if not (k_path.exists() and p_path.exists() and c_path.exists()):
                raise FileNotFoundError(f"[{tracer}] Missing synthetic inputs:\n  {k_path}\n  {p_path}\n  {c_path}")

            k_all = np.loadtxt(k_path)
            P_all = np.loadtxt(p_path)
            cov_all = np.loadtxt(c_path)
            if args.cov_scale != 1.0:
                cov_all = float(args.cov_scale) * cov_all

            start = int(np.searchsorted(k_all, float(args.kmin_cut), side="left"))
            stop = int(np.searchsorted(k_all, float(args.kmax_cut), side="right"))
            Ncut = stop - start
            if Ncut <= 0:
                raise RuntimeError(f"[{tracer}] k cuts removed all bins.")
            k_out = k_all[start:stop]

            data_vec, cov_mat = select_data_and_cov(
                P_all, cov_all, present_ells=(0, 2, 4), requested_ells=tuple(req_ells), start=start, Ncut=Ncut
            )

            ps_theory = fkptjaxTracerPowerSpectrumMultipoles()
            ps_theory.init.update(
                freedom=freedom,
                mock_type="synthetic",
                prior_basis=args.prior_basis,
                tracer=tracer,
                template=template,
                k=k_out,
                ells=list(req_ells),
                model=args.model,
                ide_variant=theory_variant,
                beyond_eds=bool(args.beyond_eds),
                rescale_PS=bool(args.rescale_PS),
                h_fid=getattr(fid, "h", None),
                shotnoise=1e4,
                b3_coev=True,
            )
            _ = apply_alpha_analytic_marginalization(ps_theory)

            ps_obs = TracerPowerSpectrumMultipolesObservable(
                data=data_vec,
                theory=ps_theory,
                k=[k_out for _ in req_ells],
                ells=list(req_ells),
            )

            cov_like = ObservableCovariance(cov_mat, observables=covmeta_from_observables([ps_obs]))
            obs_list = [ps_obs]

        elif mt_tr == "cubic":
            ps_theory = fkptjaxTracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=args.prior_basis,
                freedom=freedom,
                mock_type="cubic",
                tracer=tracer,
                model=args.model,
                ide_variant=theory_variant,
                beyond_eds=bool(args.beyond_eds),
                rescale_PS=bool(args.rescale_PS),
                h_fid=getattr(fid, "h", None),
                shotnoise=1e4,
                ells=req_ells,
            )

            # Analytic marginalization over alphas (same as synthetic/weiliu)
            _ = apply_alpha_analytic_marginalization(ps_theory)

            if include_bk:
                bs_theory = fkptjaxTracerBispectrumMultipoles(
                    pt=ps_theory.pt,
                    template=template,
                    prior_basis=args.prior_basis,
                    freedom=freedom,
                    mock_type="cubic",
                    tracer=tracer,
                    model=args.model,
                    ide_variant=theory_variant,
                    beyond_eds=bool(args.beyond_eds),
                    rescale_PS=bool(args.rescale_PS),
                    ells=((0, 0, 0), (2, 0, 2)),
                    precision=[10, 8, 8],
                    basis="sugiyama",
                    renormalized=True,
                    interpolation_method="linear",
                    bias_scheme="folps",
                    h_fid=getattr(fid, "h", None),
                )

            obs_list, cov_mat, kin_expected = build_cubic_observables_for_tracer(
                tracer=tracer,
                ps_theory=ps_theory,
                bs_theory=bs_theory,
                kr_max=args.kr_max,
                kr_b0_max=(args.kr_b0_max if include_bk else None),
                kr_b2_max=(args.kr_b2_max if include_bk else None),
                include_bispectrum=include_bk,
                ells=tuple(req_ells),
            )

            kin_expected = np.asarray(kin_expected).reshape(-1)
            nkin_for_name = int(kin_expected.size)

            cov_like = ObservableCovariance(cov_mat, observables=covmeta_from_observables(obs_list))

        elif mt_tr == "weiliu":
            # Load WeiLiu vector/cov + k grid first
            data_vec, cov_mat, k_out = load_weiliu_vector_and_covariance(
                tracer,
                kmin_cut=args.kmin_cut,
                kmax_cut=args.kmax_cut,
                requested_ells=tuple(req_ells),
            )

            # Build theory on the same k grid
            ps_theory = fkptjaxTracerPowerSpectrumMultipoles()
            ps_theory.init.update(
                freedom=freedom,
                mock_type="weiliu",
                prior_basis=args.prior_basis,
                tracer=tracer,
                template=template,
                k=k_out,
                ells=list(req_ells),
                model=args.model,
                ide_variant=theory_variant,
                beyond_eds=bool(args.beyond_eds),
                rescale_PS=bool(args.rescale_PS),
                h_fid=getattr(fid, "h", None),
                shotnoise=1e4,
                b3_coev=True,
            )
            _ = apply_alpha_analytic_marginalization(ps_theory)

            ps_obs = TracerPowerSpectrumMultipolesObservable(
                data=data_vec,
                covariance=cov_mat,
                theory=ps_theory,
                ells=list(req_ells),
                k=k_out,
                # IMPORTANT: no windowing for WeiLiu formatted files
            )

            cov_like = ObservableCovariance(cov_mat, observables=covmeta_from_observables([ps_obs]))
            obs_list = [ps_obs]
            kin_expected = None
            nkin_for_name = None
            
        else:
            raise ValueError(f"Unhandled mock_type={mt_tr!r}")

        emu_path = args.emu_dir / emu_filename(
            tracer=tracer,
            kr_max=float(args.kr_max),
            ide_variant=str(args.ide_variant),
            beyond_eds=bool(args.beyond_eds),
            add_bispectrum=bool(include_bk),
            mock_type=mt_tr,
            ells=tuple(req_ells),
            nkin=nkin_for_name,
        )

        # ------------------------------
        # Create emulator (COSMO+IDE ONLY) from PT
        # ------------------------------
        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emu] ({tracer}) exists -> {emu_path.name}")
            else:
                print(
                    f"[Emu] ({tracer}) fitting Taylor emulator: "
                    f"order_lcdm={args.emu_order_lcdm}, order_ide={args.emu_order_ide} ..."
                )

                pt_for_emu = ps_theory.pt

                # cubic: force PT k-grid = kin
                if mt_tr == "cubic":
                    print(f"[Emu] ({tracer}) cubic: enforcing PT k-grid = kin (len={kin_expected.size})")
                    _set_pt_kgrid_for_cubic(pt_for_emu, kin_expected)

                # Freeze everything except cosmology+IDE in PT params
                allowed_basenames = _allowed_emu_param_basenames(str(args.ide_variant))
                saved_fixed = _freeze_all_params_except(pt_for_emu, allowed_basenames)

                try:
                    emu_engine = emulator_engine_for(args.ide_variant, args.emu_order_lcdm, args.emu_order_ide)
                    emu = Emulator(pt_for_emu, engine=emu_engine)

                    emu.set_samples()
                    emu.fit()
                    emu.save(str(emu_path))
                    print(f"[Emu] ({tracer}) saved -> {emu_path}")

                finally:
                    # Restore fixed flags so later likelihood works normally
                    _restore_fixed_flags(pt_for_emu, saved_fixed)

        # ------------------------------
        # Use emulator: replace PT inside ps_theory (and bs_theory)
        # ------------------------------
        if args.use_emu:
            if rank == 0 and not emu_path.exists():
                raise FileNotFoundError(f"[Emu] Missing {emu_path} for {tracer}. Run with --create-emu first.")
            comm.Barrier()

            emu_loaded = EmulatedCalculator.load(str(emu_path))

            # Ensure cosmology+IDE params exist on the emulator init
            for p in cosmo.init.params:
                if p in emu_loaded.init.params:
                    emu_loaded.init.params.set(p)

            if mt_tr == "cubic":
                # Make sure emulator uses the same k-grid (kin) as window expects
                _set_pt_kgrid_for_cubic(emu_loaded, kin_expected)

            # Plug emulator as PT in theory calculators
            ps_theory.init.update(pt=emu_loaded)
            if include_bk and bs_theory is not None:
                bs_theory.init.update(pt=emu_loaded)

            if rank == 0:
                print(f"[Emu] ({tracer}) using {emu_path.name}")

        obs_for_plot[tracer] = obs_list

        # Namespace nuisance params per tracer
        is_phys = bool(getattr(ps_theory, "is_physical_prior", False))
        suffix = "p" if is_phys else ""

        def pname(base: str) -> str:
            return f"{base}{suffix}"

        nuis_to_namespace = [
            "b1", "b2", "bs2", "b3nl",
            "alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot",
            "ctilde", "PshotP",
        ]
        if str(args.prior_basis) == "APscaling":
            nuis_to_namespace += ["bK2", "btd"]

        for par in ps_theory.params.select(basename=[pname(nm) for nm in nuis_to_namespace]):
            par.update(namespace=tracer.lower())
        if include_bk and bs_theory is not None:
            for par in bs_theory.params.select(basename=[pname(nm) for nm in nuis_to_namespace]):
                par.update(namespace=tracer.lower())

        like = ObservablesGaussianLikelihood(observables=obs_list, covariance=cov_like, name=tracer.lower())
        likes.append(like)

    if args.create_emu:
        comm.Barrier()
        if rank == 0:
            print("[Emu] Done. Exiting because --create-emu was set.")
        return

    likelihood = SumLikelihood(likes)

    # -------------------------------------------------------------------------
    # Single likelihood evaluation (for testing purposes)
    # -------------------------------------------------------------------------
    if args.test:
        if rank == 0:
            print("[test] Evaluating likelihood at current default parameter point...")

        if args.no_jit_test:
            lval = float(likelihood())
            if rank == 0:
                print("[test] Likelihood (no-jit):", lval)
        else:
            import jax
            jlike = jax.jit(likelihood)
            lval = float(jlike())
            if rank == 0:
                print("[test] Likelihood (jit):", lval)

        if args.plot_test and rank == 0:
            plot_observables(obs_for_plot, args.plot_dir)

        if rank == 0:
            print("[test] OK.")
        return

    # -------------------------------------------------------------------------
    # MCMC (Metropolis-Hasting)
    # -------------------------------------------------------------------------
    if args.run_chains:
        save_pattern = chain_pattern_from(str(Path(args.chain_name)))
        out_parent = Path(save_pattern.replace("*", "chain")).parent
        out_parent.mkdir(parents=True, exist_ok=True)

        if args.restart:
            existing = sorted(glob(save_pattern))
            existing = [f for f in existing if "profiles" not in Path(f).stem]
            chains_arg = existing if existing else int(args.nchains)
            if rank == 0:
                if existing:
                    print(f"[resume] Resuming {len(existing)} chains")
                else:
                    print(f"[resume] No existing files found; starting {args.nchains} chains")
        else:
            chains_arg = int(args.nchains)

        sampler = MCMCSampler(
            likelihood,
            chains=chains_arg,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            seed=42,
            ref_scale=float(args.ref_scale),
        )
        sampler.run(check={"max_eigen_gr": 0.01}, check_every=int(args.check_every), max_iterations=int(args.max_iter))
        return

    # -------------------------------------------------------------------------
    # Best-fit / MAP (MinuitProfiler)
    # -------------------------------------------------------------------------
    if args.run_bestfit:
        # Decide default outputs
        prefix = Path(args.chain_name).with_suffix("").name
        profiles_out = args.profiles_out or (Path(args.chain_name).parent / f"{prefix}_profiles.npy")
        bestfit_out = args.bestfit_out or (Path(args.chain_name).parent / f"{prefix}_bestfit.json")

        profiles_out.parent.mkdir(parents=True, exist_ok=True)
        bestfit_out.parent.mkdir(parents=True, exist_ok=True)

        if rank == 0:
            print(f"[MAP] Saving profiles to {profiles_out}")
            print(f"[MAP] Saving best-fit params to {bestfit_out}")

        save_fn = str(profiles_out) if rank == 0 else None
        profiler = MinuitProfiler(
            likelihood,
            gradient=bool(args.gradient),
            rescale=bool(args.rescale),
            covariance=args.covariance,
            save_fn=save_fn,
            ref_scale=float(args.ref_scale),
        )

        profiler.maximize(
            niterations=int(args.niterations),        # e.g. 4
            max_iterations=int(args.max_calls),
        )

        if rank == 0:
            # After maximize, likelihood.params should sit at the best-fit point.
            bestfit = {}
            for p in likelihood.params:
                try:
                    bestfit[p.name] = float(p.value)
                except Exception:
                    pass

            with open(bestfit_out, "w") as f:
                json.dump(bestfit, f, indent=2, sort_keys=True)

            print(f"[MAP] done. profiles -> {profiles_out}")
            print(f"[MAP] done. bestfit  -> {bestfit_out}")

        comm.Barrier()
        return

    if rank == 0:
        print("Nothing to do. Use --test, --run_chains, --create-emu, and/or --use-emu.")


if __name__ == "__main__":
    main()