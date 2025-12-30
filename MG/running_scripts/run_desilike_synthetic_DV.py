#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from pathlib import Path
from glob import glob

import numpy as np
from mpi4py import MPI

# -------------------------------------------------------------------
# Environment knobs (set before importing heavy JAX stacks downstream)
# -------------------------------------------------------------------
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ["FOLPS_BACKEND"] = "jax"
#import jax
#jax.config.update("jax_enable_x64", True)



#os.environ.setdefault("JAX_DISABLE_JIT", "1")  # set "0" for speed once validated

from desilike import setup_logging, parameter
from desilike.theories import Cosmoprimo
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from cosmoprimo.fiducial import DESI

from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.observables import ObservableCovariance
from desilike.likelihoods import ObservablesGaussianLikelihood

from desilike.samplers import MCMCSampler
from desilike.profilers import MinuitProfiler

try:
    from desilike.samplers import EmceeSampler
except Exception:
    EmceeSampler = None

from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine

try:
    from desilike.theories.galaxy_clustering import fkptjaxTracerPowerSpectrumMultipoles
except Exception as exc:
    raise ImportError(
        "Could not import fkptjaxTracerPowerSpectrumMultipoles from desilike.theories.galaxy_clustering.\n"
        "Make sure your branch/module exposes it there."
    ) from exc


# ============================================================================
# Helpers
# ============================================================================
def parse_ells_str(s: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    allowed = {0, 2, 4}
    if not set(vals).issubset(allowed) or len(vals) == 0:
        raise ValueError(f"--ells must be a comma-separated subset of 0,2,4; got {s!r}")
    return tuple(sorted(vals))


def select_data_and_cov(
    Pvec: np.ndarray,
    cov: np.ndarray,
    present_ells: tuple[int, ...],
    requested_ells: tuple[int, ...],
    start: int,
    Ncut: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assumes stacked-by-ell blocks:
      Pvec = [P0(k_0..k_{Nk-1}), P2(...), P4(...)] (or subset)
    with covariance in the same ordering (block by ell).
    """
    if not set(requested_ells).issubset(set(present_ells)):
        raise ValueError(f"Requested ells {requested_ells} not subset of present {present_ells}")

    Pvec = np.asarray(Pvec).reshape(-1)
    if Pvec.size % len(present_ells) != 0:
        raise ValueError(f"Pvec size {Pvec.size} not divisible by n_ells={len(present_ells)}")

    Nraw = Pvec.size // len(present_ells)
    cov = np.asarray(cov)
    if cov.shape != (Pvec.size, Pvec.size):
        raise ValueError(f"cov shape {cov.shape} not compatible with Pvec size {Pvec.size}")

    order_map = {ell: i for i, ell in enumerate(present_ells)}
    idx: list[int] = []
    for ell in requested_ells:
        base = order_map[ell] * Nraw
        idx.extend(range(base + start, base + start + Ncut))
    idx = np.array(idx, dtype=int)
    return Pvec[idx], cov[np.ix_(idx, idx)]


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


def emu_filename(
    tag: str,
    k: np.ndarray,
    ells: tuple[int, ...],
    beyond_eds: bool,
    redshift_bins: bool,
    scale_bins: bool,
    mg_variant: str,
    prior_basis: str,
    kc: float | None = None,
    scale_bins_method: str | None = None,
) -> str:
    """Match the old-script emulator naming logic (no cosmology/nuisance state embedded)."""
    dk = float(np.median(np.diff(k))) if k.size > 1 else 0.0
    kmin_edge = float(k.min() - 0.5 * dk)
    kmax_edge = float(k.max() + 0.5 * dk)

    if redshift_bins and scale_bins:
        mode = "binning_zk"
        if scale_bins_method:
            mode += f"-{scale_bins_method}"
        if kc is not None:
            mode += f"_kc{kc:g}"
    elif redshift_bins:
        mode = "binning_z"
    else:
        if mg_variant == "mu_OmDE":
            mode = "mu0"
        else:
            mode = str(mg_variant)

    if not beyond_eds:
        mode = "eds_" + mode

    return (
        f"emu-fs_folps_isitgr_{mode}_{tag}"
        f"_k{kmin_edge:.3f}-{kmax_edge:.3f}"
        f"_l{''.join(map(str, ells))}.npy"
    )


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="DESI FS runner: isitgr cosmo + fkptjax tables + folps multipoles (HDKI plumbing), "
        "mirroring your OLD script's cosmology + nuisance settings."
    )

    p.add_argument("--mode", choices=["mcmc", "emcee", "map"], default="mcmc")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--create-emu", action="store_true", help="Build per-tracer emulators then exit.")
    g.add_argument("--use-emu", action="store_true", help="Load and use per-tracer emulators.")
    p.add_argument("--emu-dir", type=Path, default=Path("./emulators"), help="Directory for emulator files.")
    p.add_argument("--emu-order", type=int, default=6, help="Taylor emulator order (finite).")

    p.add_argument("--chains-dir", type=Path, default=Path("./chains"))
    p.add_argument("--chain-prefix", type=str, default="chain_fs_folps_isitgr_fkptjax")

    # synthetic inputs
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/n/home12/cgarciaquintero/DESI/MG_validation/synthetic_noiseless/data_vectors/"),
    )
    p.add_argument("--data-tag", type=str, default=None, help="Required (e.g. LCDM, HS_F4, ...).")
    p.add_argument("--use-cov-x10", action="store_true")
    p.add_argument("--cov-scale", type=float, default=1.0)

    # cuts + multipoles
    p.add_argument("--kmin-cut", type=float, default=0.02)
    p.add_argument("--kmax-cut", type=float, default=0.30)
    p.add_argument("--ells", type=str, default="0,2")
    p.add_argument("--freedom", choices=["max", "min"], default="max")
    p.add_argument("--fid-model", type=str, default="LCDM")

    # IMPORTANT: match OLD script basis behavior
    p.add_argument(
        "--prior-basis",
        choices=["standard", "physical", "physical_velocileptors", "APscaling"],
        default="physical",
    )
    p.add_argument("--h-fid", type=float, default=None, help="Needed for --prior-basis APscaling (synthetic fid h).")

    p.add_argument("--beyond-eds", action="store_true")

    # cosmology external priors toggles (as in OLD script)
    p.add_argument("--skip-ns-prior", action="store_true", help="Do not apply a prior on n_s.")
    p.add_argument("--skip-bbn-prior", action="store_true", help="Do not apply BBN prior on omega_b.")

    # MG toggles
    p.add_argument("--force-GR", action="store_true")
    p.add_argument("--mg-variant", choices=["mu_OmDE", "BZ", "BZ_fR", "binning"], default="mu_OmDE")

    # binning toggles
    p.add_argument("--redshift-bins", action="store_true")
    p.add_argument("--scale-bins", action="store_true")
    p.add_argument("--scale-bins-method", type=str, default="traditional")
    p.add_argument("--kc", type=float, default=0.1)

    # running
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ref-scale", type=float, default=1.2)

    p.add_argument("--nchains", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=50000)
    p.add_argument("--check-every", type=int, default=3000)

    p.add_argument("--nwalkers", type=str, default=None)
    p.add_argument("--emcee-max-iter", type=int, default=20000)

    # MAP
    p.add_argument("--max-calls", type=int, default=int(1e5))
    p.add_argument("--gradient", action="store_true")
    p.add_argument("--rescale", action="store_true")
    p.add_argument("--covariance", type=str, default=None)
    p.add_argument("--profiles-out", type=Path, default=None)

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    setup_logging("info")

    if args.data_tag is None:
        raise ValueError("--data-tag is required (e.g. LCDM, HS_F4).")

    # IMPORTANT: your fkptjax wrapper expects this to be False for now
    RESCALE_PS = False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if MPI.COMM_WORLD.Get_rank() == 0:
        import jax
        print("[JAX] backend:", jax.default_backend())
        print("[JAX] devices:", jax.devices())
        print("[JAX] x64:", jax.config.read("jax_enable_x64"))

    args.chains_dir.mkdir(parents=True, exist_ok=True)
    args.emu_dir.mkdir(parents=True, exist_ok=True)

    ells = parse_ells_str(args.ells)
    fid_model = str(args.fid_model)
    prior_basis = str(args.prior_basis)

    # enforce OLD script binning logic
    if args.scale_bins and not args.redshift_bins:
        raise ValueError("--scale-bins requires --redshift-bins")
    if (args.redshift_bins or args.scale_bins) and args.mg_variant != "binning":
        raise ValueError("--redshift-bins/--scale-bins only for --mg-variant binning.")
    if args.mg_variant == "binning" and not (args.redshift_bins or args.scale_bins):
        raise ValueError("--mg-variant binning requires --redshift-bins and/or --scale-bins")

    # APscaling requires h_fid (OLD behavior)
    h_fid_global: float | None = None
    if prior_basis == "APscaling":
        if args.h_fid is not None:
            h_fid_global = float(args.h_fid)
        else:
            fid = DESI()
            try:
                h_fid_global = float(getattr(fid, "h", fid["h"]))
            except Exception:
                raise RuntimeError("Could not infer h_fid from DESI(); pass --h-fid explicitly.")

    # Tracers (keep the b1/b2 refs you were using in the newer folps runner)
    # (file_tag, tracer_tag, b1_fid, z_eff, b2_ref)
    tracer_table = [
        ("BGS",  "BGS",  1.5, 0.295, -0.52),
        ("LRG1", "LRG1", 2.0, 0.510, -0.42),
        ("LRG2", "LRG2", 2.1, 0.706, -0.36),
        ("LRG3", "LRG3", 2.2, 0.934, -0.30),
        ("ELG",  "ELG",  1.3, 1.321, -0.62),
        ("QSO",  "QSO",  2.5, 1.484, -0.25),
    ]

    # ---------- Build prefix (similar spirit to OLD script) ----------
    prefix = args.chain_prefix
    prefix += "_beds" if args.beyond_eds else "_eds"
    if args.use_emu:
        prefix += "_emu"
    prefix += f"_{args.freedom}_{prior_basis}_{fid_model}"
    if args.mg_variant == "binning":
        prefix += "_binning"
        if args.redshift_bins:
            prefix += "_z"
        if args.scale_bins:
            prefix += f"_k_kc{args.kc:g}"
    else:
        prefix += "_GR" if args.force_GR else (f"_mu0" if args.mg_variant == "mu_OmDE" else f"_{args.mg_variant}")
    prefix += f"_l{''.join(map(str, ells))}"
    prefix += f"_data-{args.data_tag}"
    if args.use_cov_x10:
        prefix += "_covx10"
    if args.cov_scale != 1.0:
        prefix += f"_covx{args.cov_scale:g}"

    if rank == 0:
        print("Job prefix:", prefix)

    # -----------------------------
    # Cosmology engine (MATCH OLD SCRIPT SETTINGS)
    # -----------------------------
    cosmo = Cosmoprimo(
        engine="isitgr",
        redshift_bins=args.redshift_bins,
        scale_bins=args.scale_bins,
        scale_bins_method=args.scale_bins_method,
    )

    # Fix / set values like OLD
    if "tau_reio" in cosmo.init.params:
        cosmo.init.params["tau_reio"].update(fixed=True)

    if "N_eff" in cosmo.init.params:
        cosmo.init.params["N_eff"].update(fixed=True, value=3.046)
    if "m_ncdm" in cosmo.init.params:
        cosmo.init.params["m_ncdm"].update(fixed=True, value=0.06)

    # Optional external priors (OLD)
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

    # Wide uniforms + refs (OLD)
    prior_limits = {"h": (0.4, 1.0), "omega_cdm": (0.001, 0.99), "logA": (1.61, 3.91)}
    for name, scale, delta in [("h", 0.001, 0.03), ("omega_cdm", 0.001, 0.007), ("logA", 0.001, 0.05)]:
        if name in cosmo.init.params:
            par = cosmo.init.params[name]
            lo, hi = prior_limits[name]
            par.update(
                fixed=False,
                prior={"dist": "uniform", "limits": (lo, hi)},
                ref={"dist": "norm", "loc": par.value, "scale": scale},
                delta=delta,
            )

    if "sigma8_m" in cosmo.init.params:
        cosmo.init.params["sigma8_m"].update(derived=True, latex=r"\sigma_8")

    # -----------------------------
    # MG params (as in OLD logic; plus your BZ_fR convenience)
    # -----------------------------
    if args.mg_variant == "mu_OmDE":
        add_or_update_param(
            cosmo,
            "mu0",
            value=0.0,
            fixed=args.force_GR,
            prior=None if args.force_GR else {"dist": "uniform", "limits": (-3.0, 3.0)},
            ref=None if args.force_GR else {"dist": "norm", "loc": 0.0, "scale": 0.01},
            delta=None if args.force_GR else 0.25,
        )

    elif args.mg_variant in ["BZ", "BZ_fR"]:
        # ensure parameters exist
        for nm, val in [("beta_1", 0.0), ("lambda_1", 0.0), ("exp_s", 0.0)]:
            add_or_update_param(cosmo, nm, value=val, fixed=True)

        if args.force_GR:
            cosmo.init.params["beta_1"].update(fixed=True, value=0.0)
            cosmo.init.params["lambda_1"].update(fixed=True, value=0.0)
            cosmo.init.params["exp_s"].update(fixed=True, value=0.0)
        else:
            if args.mg_variant == "BZ_fR":
                # "f(R)-like" BZ choice
                cosmo.init.params["beta_1"].update(fixed=True, value=4.0 / 3.0)
                cosmo.init.params["exp_s"].update(fixed=True, value=4.0)
                cosmo.init.params["lambda_1"].update(
                    fixed=False,
                    prior={"dist": "uniform", "limits": (0.0, 1e6)},
                    ref={"dist": "norm", "loc": 30.0, "scale": 10.0},
                    delta=10.0,
                )
            else:
                cosmo.init.params["beta_1"].update(
                    fixed=False,
                    prior={"dist": "uniform", "limits": (-5.0, 5.0)},
                    ref={"dist": "norm", "loc": 1.0, "scale": 0.1},
                    delta=0.3,
                )
                cosmo.init.params["lambda_1"].update(
                    fixed=False,
                    prior={"dist": "uniform", "limits": (0.0, 1e6)},
                    ref={"dist": "norm", "loc": 100.0, "scale": 100.0},
                    delta=100.0,
                )
                cosmo.init.params["exp_s"].update(
                    fixed=False,
                    prior={"dist": "uniform", "limits": (0.0, 5.0)},
                    ref={"dist": "norm", "loc": 2.0, "scale": 0.3},
                    delta=0.5,
                )

    elif args.mg_variant == "binning":
        # mu bins varied unless force_GR
        for nm in ["mu1", "mu2", "mu3", "mu4"]:
            add_or_update_param(
                cosmo,
                nm,
                value=1.0,
                fixed=args.force_GR,
                prior=None if args.force_GR else {"dist": "uniform", "limits": (-2.0, 4.0)},
                ref=None if args.force_GR else {"dist": "norm", "loc": 1.0, "scale": 0.05},
                delta=None if args.force_GR else 0.5,
            )
        # transitions/fixed knobs
        for nm, val in [
            ("z_div", 1.0),
            ("z_TGR", 2.0),
            ("z_tw", 0.05),
            ("k_tw", 0.01),
            ("k_c", args.kc),
            ("Sigma1", 1.0),
            ("Sigma2", 1.0),
            ("Sigma3", 1.0),
            ("Sigma4", 1.0),
        ]:
            add_or_update_param(cosmo, nm, value=float(val), fixed=True)
    else:
        raise ValueError(f"Unknown mg-variant {args.mg_variant!r}")

    likelihoods = []

    # -----------------------------
    # Loop over tracers
    # -----------------------------
    for file_tag, tracer_tag, b1_fid, z_eff, b2_ref in tracer_table:
        namespace = file_tag.lower()

        data_tag = str(args.data_tag)
        k_path = args.data_dir / f"{file_tag}_{data_tag}_k.txt"
        p_path = args.data_dir / f"{file_tag}_{data_tag}_P0P2P4.txt"
        c_suffix = "_cov_x10.txt" if args.use_cov_x10 else "_cov.txt"
        c_path = args.data_dir / f"{file_tag}_{data_tag}{c_suffix}"

        if not k_path.exists() or not p_path.exists() or not c_path.exists():
            raise FileNotFoundError(
                f"[{file_tag}] Missing synthetic inputs:\n"
                f"  k:   {k_path}\n"
                f"  P:   {p_path}\n"
                f"  cov: {c_path}\n"
            )

        k_all = np.loadtxt(k_path)
        P_all = np.loadtxt(p_path)
        cov_all = np.loadtxt(c_path)

        Nk_file = k_all.size
        if P_all.size != 3 * Nk_file:
            raise ValueError(
                f"[{file_tag}] Expected P size = 3*Nk = {3*Nk_file}, got {P_all.size}. "
                "This script assumes P file is stacked by ell blocks: P0 then P2 then P4."
            )
        if cov_all.shape != (P_all.size, P_all.size):
            raise ValueError(f"[{file_tag}] cov shape {cov_all.shape} incompatible with P size {P_all.size}")

        if args.cov_scale != 1.0:
            cov_all = float(args.cov_scale) * cov_all

        present_ells_data = (0, 2, 4)

        start = int(np.searchsorted(k_all, float(args.kmin_cut), side="left"))
        stop = int(np.searchsorted(k_all, float(args.kmax_cut), side="right"))
        Ncut = stop - start
        if Ncut <= 0:
            raise RuntimeError(f"[{file_tag}] k cuts removed all bins.")

        k_out = k_all[start:stop]
        data_vec, cov_mat = select_data_and_cov(P_all, cov_all, present_ells_data, ells, start, Ncut)

        # -----------------------------
        # Build template + theory
        # -----------------------------
        template = DirectPowerSpectrumTemplate(z=float(z_eff), fiducial=DESI(), cosmo=cosmo)

        # for BZ_fR we still pass theory BZ (like your folps runner)
        theory_variant = "BZ" if args.mg_variant == "BZ_fR" else args.mg_variant

        theory = fkptjaxTracerPowerSpectrumMultipoles()

        init_kwargs = dict(
            freedom=args.freedom,
            prior_basis=prior_basis,
            tracer=tracer_tag,
            template=template,
            k=k_out,
            ells=list(ells),
            b3_coev=True,
            model="HDKI",
            mg_variant=theory_variant,
            beyond_eds=bool(args.beyond_eds),
            rescale_PS=RESCALE_PS,
            shotnoise=1e4,
        )
        # Match OLD: APscaling passes h_fid, and nuisance mapping handled internally
        if prior_basis == "APscaling":
            init_kwargs["h_fid"] = float(h_fid_global) if h_fid_global is not None else float(args.h_fid)
        # Keep b1_fid as a stable ref for your physical/folps usage (harmless for standard too)
        init_kwargs["b1_fid"] = float(b1_fid)

        theory.init.update(**init_kwargs)

        # -----------------------------
        # Emulator (optional)
        # -----------------------------
        emu_path = args.emu_dir / emu_filename(
            tag=file_tag,
            k=k_out,
            ells=ells,
            beyond_eds=bool(args.beyond_eds),
            redshift_bins=bool(args.redshift_bins),
            scale_bins=bool(args.scale_bins),
            mg_variant=str(args.mg_variant),
            prior_basis=prior_basis,
            kc=float(args.kc),
            scale_bins_method=str(args.scale_bins_method),
        )

        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emulator] ({file_tag}) exists → {emu_path.name}")
            else:
                print(f"[Emulator] ({file_tag}) fitting Taylor emulator (finite, order={args.emu_order})…")
                _ = theory.pt()  # force build
                emu_engine = TaylorEmulatorEngine(method="finite", order=int(args.emu_order))
                emu = Emulator(theory.pt, engine=emu_engine)
                emu.set_samples()
                emu.fit()
                emu.save(str(emu_path))
                print(f"[Emulator] ({file_tag}) saved → {emu_path.name}")

        if args.use_emu:
            if rank == 0 and not emu_path.exists():
                raise FileNotFoundError(f"[Emulator] Missing {emu_path} for {file_tag}. Run with --create-emu.")
            comm.Barrier()

            emu_loaded = EmulatedCalculator.load(str(emu_path))

            # important: share cosmology params into emulator init (as you did before)
            for p in cosmo.init.params:
                if p in emu_loaded.init.params:
                    emu_loaded.init.params.set(p)

            theory.init.update(pt=emu_loaded)
            if rank == 0:
                print(f"[{file_tag}] PT backend:", type(theory.pt).__name__)

        # -----------------------------
        # Nuisance param namespacing + alpha analytic marginalization (MATCH OLD SCRIPT)
        # -----------------------------
        is_phys = bool(getattr(theory, "is_physical_prior", False))
        suffix = "p" if is_phys else ""

        def pname(base: str) -> str:
            return f"{base}{suffix}"

        # alpha marg
        alpha_bases = ["alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot"]
        for par in theory.params.select(basename=[pname(b) for b in alpha_bases]):
            if par.varied:
                par.update(derived=".marg")

        # Optional: set b1/b2 refs (kept from your folps script; does not change the basis mapping itself)
        b1_name = pname("b1")
        if b1_name in theory.params:
            theory.params[b1_name].update(ref={"dist": "norm", "loc": float(b1_fid), "scale": 0.05})

        b2_name = pname("b2")
        if b2_name in theory.params:
            theory.params[b2_name].update(ref={"dist": "norm", "loc": float(b2_ref), "scale": 0.1})

        nuis_to_namespace = [
            "b1", "b2", "bs2", "b3nl",
            "alpha0", "alpha2", "alpha4",
            "alpha0shot", "alpha2shot",
            "ctilde",
            "PshotP",
        ]
        if prior_basis == "APscaling":
            nuis_to_namespace += ["bK2", "btd"]

        for par in theory.params.select(basename=[pname(nm) for nm in nuis_to_namespace]):
            par.update(namespace=namespace)

        # -----------------------------
        # Observable + covariance
        # -----------------------------
        observable = TracerPowerSpectrumMultipolesObservable(
            data=data_vec,
            theory=theory,
            k=[k_out for _ in ells],   # tells desilike how to split data_vec
            ells=list(ells),
        )
        covmeta = [{"name": "PowerSpectrumMultipoles", "x": [k_out] * len(ells), "projs": list(ells)}]
        covariance = ObservableCovariance(cov_mat, observables=covmeta)

        lk = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance, name=namespace)
        likelihoods.append(lk)

        if rank == 0:
            print(f"[{file_tag}] appended likelihood (namespace={namespace})")

    if args.create_emu:
        comm.Barrier()
        if rank == 0:
            print("[Emulator] Done. Exiting because --create-emu was set.")
        return

    likelihood = sum(likelihoods)

    # -----------------------------
    # Run
    # -----------------------------
    save_pattern = str(args.chains_dir / f"{prefix}_*.npy")

    if args.mode == "mcmc":
        if args.resume:
            existing = sorted(glob(str(args.chains_dir / f"{prefix}_*.npy")))
            existing = [f for f in existing if "profiles" not in Path(f).stem]
            chains_arg = existing if existing else args.nchains
            if rank == 0:
                if existing:
                    print(f"[resume/mcmc] Resuming {len(existing)} chains")
                else:
                    print(f"[resume/mcmc] No existing files found; starting {args.nchains} chains")
        else:
            chains_arg = args.nchains

        sampler = MCMCSampler(
            likelihood,
            chains=chains_arg,
            seed=42,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
        )
        sampler.run(check={"max_eigen_gr": 0.01}, check_every=args.check_every, max_iterations=args.max_iter)

    elif args.mode == "emcee":
        if EmceeSampler is None:
            raise ImportError("EmceeSampler could not be imported from desilike.samplers in this environment.")

        if args.resume:
            existing = sorted(glob(str(args.chains_dir / f"{prefix}_*.npy")))
            chains_arg = existing if existing else 1
        else:
            chains_arg = 1

        sampler = EmceeSampler(
            likelihood,
            chains=chains_arg,
            seed=42,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
            nwalkers=args.nwalkers,
        )
        sampler.run(check_every=args.check_every, max_iterations=args.emcee_max_iter)

    elif args.mode == "map":
        profiles_out = args.profiles_out or (args.chains_dir / f"{prefix}_profiles.npy")
        if rank == 0:
            print(f"[MAP] Saving profiles to {profiles_out}")

        profiler = MinuitProfiler(
            likelihood,
            gradient=args.gradient,
            rescale=args.rescale,
            covariance=args.covariance,
            save_fn=str(profiles_out),
            ref_scale=args.ref_scale,
        )
        profiler.maximize(max_iterations=args.max_calls)
        if rank == 0:
            print(f"[Profiles] saved to {profiles_out}")

    else:
        raise ValueError(f"Unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
