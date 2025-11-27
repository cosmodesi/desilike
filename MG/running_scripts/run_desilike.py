#!/usr/bin/env python3
from __future__ import annotations
import os

# ---------------------------------------------------------------
# Now it is safe to import desilike, cosmoprimo, mpi4py, etc.
# ---------------------------------------------------------------
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from mpi4py import MPI

from desilike import setup_logging, parameter
from desilike.theories import Cosmoprimo
from desilike.theories.galaxy_clustering import (
    DirectPowerSpectrumTemplate,
    fkptTracerPowerSpectrumMultipoles,
)
from cosmoprimo.fiducial import DESI
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.observables import ObservableCovariance
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.samplers import MCMCSampler, NautilusSampler
from desilike.samplers import StaticDynestySampler, DynamicDynestySampler
from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine
from desilike.profilers import MinuitProfiler

def parse_args():
    p = argparse.ArgumentParser(description="Multi-tracer fkpt with optional Taylor emulators and binning (MCMC or MAP).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--create-emu", action="store_true", help="Build per-tracer emulators then exit.")
    g.add_argument("--use-emu", action="store_true", help="Load and use per-tracer emulators.")

    # run mode
    p.add_argument(
        "--mode",
        choices=["mcmc", "nautilus", "dynesty-static", "dynesty-dynamic", "map"],
        default="mcmc",
        help="Choose 'mcmc' for MCMCSampler, 'nautilus' for nested sampling, 'dynesty-static', 'dynesty-dynamic', or 'map' for MinuitProfiler."
    )

    # IO
    p.add_argument("--chain-prefix", type=str, default="chain_fs_direct_fkpt_isitgr",
                   help="Base prefix for MCMC output chain files (desilike expands * to 0..N-1).")
    p.add_argument("--emu-dir", type=Path, default=Path("/n/netscratch/eisenstein_lab/Lab/cristhian/desilike/Emulators"),
                   help="Directory for emulator files.")
    p.add_argument("--chains-dir", type=Path, default=Path("./chains"), help="Output dir for chains (and MAP if not overridden).")

    # data / theory controls
    p.add_argument("--kmin-cut", type=float, default=0.02, help="k_min cut to align to data length.")
    p.add_argument("--ells", type=str, default="0,2,4", help='Multipoles to use, e.g. "0,2" or "0,2,4".')
    p.add_argument("--freedom", choices=["max", "min"], default="max",
                   help="fkpt nuisance freedom. 'min' typically omits bs/b3.")
    p.add_argument("--priors-basis", type=str, choices=["standard", "physical"], default="standard", help="prior basis to use, default is standard")
    p.add_argument("--fid-model", type=str, default="LCDM", help='Underlying model tag for IO (e.g., LCDM or F4).')
    p.add_argument("--skip-ns-prior", action="store_true", help="Do not apply a prior on n_s.")
    p.add_argument("--skip-bbn-prior", action="store_true", help="Do not apply the BBN prior on omega_b.")
    p.add_argument("--resume", action="store_true", help="Resume from existing chain files matching the built prefix.")

    # MG toggles
    p.add_argument("--force-GR", action="store_true", help="Force GR run.")
    p.add_argument("--MG-model", choices=["LCDM", "HS", "HDKI"], default="LCDM", help="Base model.")
    p.add_argument("--mg-variant", choices=["mu_OmDE", "BZ", "binning"], default="mu_OmDE") 
    p.add_argument("--beyond-eds", action="store_true", help="Enable beyond-EdS kernels (default False).")
    #p.add_argument("--rescale-PS", action="store_true", help="Rescale input PS (default False).")
    p.add_argument("--redshift-bins", action="store_true", help="Enable redshift binning.")
    p.add_argument("--scale-bins-method", type=str, default="traditional", help="Scale binning method tag.")
    p.add_argument("--scale-bins", action="store_true", help="Enable scale binning (requires --redshift-bins).")
    p.add_argument("--kc", type=float, default=0.1, help="Scale for scale transition between bins (k_c).")

    # MAP / MinuitProfiler options (ignored if --mode mcmc)
    p.add_argument("--profiles-out", type=Path, default=None, help="Path to save Profiles (.npy). Defaults to chains-dir/<prefix>_profiles.npy")
    p.add_argument("--nstart", type=int, default=None, help="Number of independent Minuit starts; default = MPI size.")
    p.add_argument("--max-calls", type=int, default=int(1e5), help="Max function calls per Minuit run.")
    p.add_argument("--gradient", action="store_true", help="Use JAX gradients if available.")
    p.add_argument("--rescale", action="store_true", help="Internally rescale params using covariance/proposals.")
    p.add_argument("--covariance", type=str, default=None, help="Path to covariance/chain to set scaling when --rescale is on.")
    p.add_argument("--ref-scale", type=float, default=1.2, help="Scale factor for parameter reference dists (for profiler starts).")

    # Nautilus-specific
    p.add_argument("--nlive", type=int, default=800, help="Number of live points for the Nautilus nested sampler.")

    return p.parse_args()

def parse_ells_str(s: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    allowed = {0, 2, 4}
    if not set(vals).issubset(allowed) or len(vals) == 0:
        raise ValueError(f"--ells must be a comma-separated subset of 0,2,4; got {s!r}")
    return tuple(sorted(vals))

def emu_filename(tag: str,
                 k: np.ndarray,
                 ells: tuple[int, ...],
                 beyond_eds: bool,
                 redshift_bins: bool,
                 scale_bins: bool,
                 mg_variant: str,
                 kc: float | None = None,
                 scale_bins_method: str | None = None) -> str:
    """Per-tracer PT emulator filename with optional binning tags, k_c and MG variant."""
    dk = float(np.median(np.diff(k))) if k.size > 1 else 0.0
    kmin_edge = float(k.min() - 0.5 * dk)
    kmax_edge = float(k.max() + 0.5 * dk)

    if scale_bins and not redshift_bins:
        raise ValueError("--scale-bins requires --redshift-bins")

    if redshift_bins and scale_bins:
        mode = "binning_zk"
        if scale_bins_method:
            mode = f"{mode}-{scale_bins_method}"
        if kc is not None:
            mode = f"{mode}_with_kc_{kc:g}"
    elif redshift_bins:
        mode = "binning_z"
    else:
        # Non-binned MG variants
        if mg_variant == "mu_OmDE":
            mode = "mu0"
        elif mg_variant == "BZ":
            mode = "BZ"
        elif mg_variant == "binning":
            # This shouldn't really happen without redshift_bins,
            # but keep a sensible label.
            mode = "binning"
        else:
            mode = mg_variant

    if not beyond_eds:
        mode = "eds_" + mode

    return (
        f"emu-fs_fkpt_isitgr_{mode}_{tag}"
        f"_k{kmin_edge:.3f}-{kmax_edge:.3f}"
        f"_l{''.join(map(str, ells))}.npy"
    )

def infer_present_ells(vec_size: int) -> tuple[int, ...]:
    if vec_size % 3 == 0:
        return (0, 2, 4)
    elif vec_size % 2 == 0:
        return (0, 2)
    else:
        return (0,)

def select_data_and_cov(Pvec: np.ndarray, cov: np.ndarray,
                        present_ells: tuple[int, ...], requested_ells: tuple[int, ...],
                        start: int, Ncut: int) -> tuple[np.ndarray, np.ndarray]:
    if not set(requested_ells).issubset(set(present_ells)):
        raise ValueError(f"Requested ells {requested_ells} not subset of present {present_ells}")
    Nraw = Pvec.size // len(present_ells)
    order_map = {ell: i for i, ell in enumerate(present_ells)}
    idx = []
    for ell in requested_ells:
        base = order_map[ell] * Nraw
        idx.extend(range(base + start, base + start + Ncut))
    idx = np.array(idx, dtype=int)
    return Pvec[idx], cov[np.ix_(idx, idx)]

def output_indices_for_ells(present_ells: tuple[int, ...], requested_ells: tuple[int, ...], Nraw: int) -> np.ndarray:
    """Indices that keep only requested ℓ blocks from a (stacked) emulator output."""
    order_map = {ell: i for i, ell in enumerate(present_ells)}  # e.g. {0:0, 2:1, 4:2}
    idx = []
    for ell in requested_ells:
        base = order_map[ell] * Nraw
        idx.extend(range(base, base + Nraw))
    return np.asarray(idx, dtype=int)

class SlicedEmu:
    """Adapter that slices a loaded emulator's vector output to a subset of ℓ blocks."""
    def __init__(self, base: EmulatedCalculator, idx: np.ndarray):
        self.base = base
        self.idx = np.asarray(idx)
        self.init = base.init  # preserve param interface
    def __call__(self, *args, **kwargs):
        y = self.base(*args, **kwargs)
        return y[self.idx]

def main():
    args = parse_args()
    setup_logging("info")

    # IO setup
    workdir = Path("/n/home12/cgarciaquintero/DESI/MG_validation/fR_noiseless_desilike")
    args.chains_dir.mkdir(parents=True, exist_ok=True)
    args.emu_dir.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Tracers: (file tag, tracer tag, b1 prior center, z_eff)
    tracer_table = [
        ("BGS",  "BGS",  1.5, 0.295, -0.52, 0.693770),
        ("LRG1", "LRG",  2.1, 0.510, -0.99, 0.620894),
        ("LRG2", "LRG",  2.1, 0.706, -1.12, 0.563789),
        ("LRG3", "LRG",  2.1, 0.919, -1.07, 0.510742),
        ("ELG",  "ELG",  1.2, 1.317, 0.03, 0.431973),
        ("QSO",  "QSO",  2.1, 1.492, -0.71, 0.403964),
    ]

    ells = parse_ells_str(args.ells)
    fid_model = str(args.fid_model)

    # Priors
    ns_prior  = None if args.skip_ns_prior else {'dist': 'norm', 'loc': 0.9649, 'scale': 0.02}
    bbn_prior = None if args.skip_bbn_prior else {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00055}

    # Enforce consistent use of binning vs MG variant
    if args.scale_bins and not args.redshift_bins:
        raise ValueError("--scale-bins requires --redshift-bins")
    if (args.redshift_bins or args.scale_bins) and args.mg_variant != "binning":
        raise ValueError("--redshift-bins / --scale-bins are only implemented for mg-variant='binning'.")
    if args.mg_variant == "binning" and not (args.redshift_bins or args.scale_bins):
        raise ValueError("mg-variant='binning' requires --redshift-bins and/or --scale-bins.")

    # ---------- Build chain prefix ----------
    prefix = args.chain_prefix
    ell_tag = f"_l{''.join(map(str, ells))}"
    if args.beyond_eds:
        prefix += "_beds"
    else:
        prefix += "_eds"
    if args.use_emu:
        prefix += "_emu"
    prefix += f"_{args.freedom}_{args.priors_basis}_{fid_model}"

    if args.redshift_bins or args.scale_bins:
        prefix += "_binning"
        if args.redshift_bins:
            prefix += "_z"
        if args.scale_bins:
            prefix += f"_k_kc{args.kc:g}"
        # also record which MG variant was used in the binned run
        if args.mg_variant != "mu_OmDE":
            prefix += f"_{args.mg_variant}"
    else:
        if args.force_GR:
            prefix += "_GR"
        else:
            if args.mg_variant == "mu_OmDE":
                prefix += "_mu0"
            else:
                prefix += f"_{args.mg_variant}"

    prefix += ell_tag

    print("Job will start with prefix:", prefix)

    # profiles output (for --mode map)
    profiles_out = args.profiles_out or (args.chains_dir / f"{prefix}_profiles.npy")
    profiles_out.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0 and args.mode == "map":
        print(f"[Profiles] Saving to: {profiles_out}")

    # ---------- Pipeline ----------
    likelihoods = []
    if args.create_emu and rank == 0:
        print("[Emulator] Build requested; constructing per-tracer emulators…")

    cosmo = None
    for file_tag, tracer_tag, b1_val, z_tracer, b2_val, sigma8_fid_val in tracer_table:
        namespace = file_tag.lower()

        # load data
        k_all   = np.loadtxt(workdir / f"{file_tag}_{fid_model}_k.txt")
        P_all   = np.loadtxt(workdir / f"{file_tag}_{fid_model}_P0P2P4.txt")
        cov_all = np.loadtxt(workdir / f"{file_tag}_{fid_model}_cov.txt")

        present_ells = infer_present_ells(P_all.size)
        if not set(ells).issubset(set(present_ells)):
            raise RuntimeError(f"[{file_tag}] requested ells {ells} not available; present={present_ells}")

        # k-range alignment
        dk_est  = float(np.median(np.diff(k_all))) if k_all.size > 1 else 0.0
        start   = int(np.searchsorted(k_all, args.kmin_cut - 0.5 * dk_est, side='left'))
        Ncut    = k_all.size - start
        if Ncut <= 0:
            raise RuntimeError(f"[{file_tag}] kmin-cut removed all bins (kmin_cut={args.kmin_cut}).")
        k_out = k_all[start:start+Ncut]

        data_vec, cov_mat = select_data_and_cov(P_all, cov_all, present_ells, ells, start, Ncut)

        dk = float(np.median(np.diff(k_out))) if k_out.size > 1 else dk_est
        kmin_edge = float(k_out[0] - 0.5 * dk)
        kmax_edge = float(k_out[-1] + 0.5 * dk)
        klim = {L: (kmin_edge, kmax_edge, dk) for L in ells}

        # Shared cosmology
        if cosmo is None:
            cosmo = Cosmoprimo(engine='isitgr',
                               redshift_bins=args.redshift_bins,
                               scale_bins=args.scale_bins,
                               scale_bins_method=args.scale_bins_method)

            # GR / base priors
            cosmo.init.params['tau_reio'].update(fixed=True)
            cosmo.init.params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}

            # neutrinos: keep standard Neff and (if present) lock m_ncdm (one 0.06 eV species)
            if 'N_eff' in cosmo.init.params:
                cosmo.init.params['N_eff'].update(fixed=True, value=3.046)
            if 'm_ncdm' in cosmo.init.params:
                cosmo.init.params['m_ncdm'].update(value=0.06, fixed=True)

            if ns_prior is not None:
                cosmo.init.params['n_s'].update(fixed=False, prior=ns_prior,
                                                ref={'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, delta=0.01)
            if bbn_prior is not None:
                cosmo.init.params['omega_b'].update(fixed=False, prior=bbn_prior, delta=0.0015)
            #cosmo.init.params['n_s'].update(fixed=True, value=0.9649)
            #cosmo.init.params['omega_b'].update(fixed=True, value=0.02237)


            # Flat priors + ref distributions for background cosmology
            prior_limits = {
                "h":         (0.4, 1.0),
                "omega_cdm": (0.001, 0.99),
                "logA":      (1.61, 3.91),
            }
            
            for name, scale, delta in [
                ("h",         0.001, 0.03),
                ("omega_cdm", 0.001, 0.007),
                ("logA",      0.001, 0.1),
            ]:
                if name in cosmo.init.params:
                    par = cosmo.init.params[name]
                    lo, hi = prior_limits[name]
                    par.update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (lo, hi)},
                        ref={"dist": "norm", "loc": par.value, "scale": scale},
                        delta=delta,
                    )
            #cosmo.init.params['h'].update(fixed=True, value=0.6736)
            #cosmo.init.params['omega_cdm'].update(fixed=True, value=0.12)
            #cosmo.init.params['logA'].update(fixed=True, value=3.0363942552728806)

            # -----------------------
            # MG knobs / parameters
            # -----------------------
            if args.mg_variant == "mu_OmDE":
                # Single-parameter mu(a) = 1 + mu0 * Omega_DE(a)
                if args.force_GR:
                    cosmo.init.params.data.append(
                        parameter.Parameter(basename="mu0", value=0.0, fixed=True)
                    )
                else:
                    cosmo.init.params.data.append(
                        parameter.Parameter(basename="mu0", value=0.0, fixed=True)
                    )
                    cosmo.init.params["mu0"].update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (-3.0, 3.0)},
                        ref={"dist": "norm", "loc": 0.0, "scale": 0.01},
                        delta=0.25,
                    )

            elif args.mg_variant == "BZ":
                # Bertschinger–Zukin: beta_1, lambda_1, exp_s
                for nm, val in [("beta_1", 0.0), ("lambda_1", 0.0), ("exp_s", 0.0)]:
                    cosmo.init.params.data.append(
                        parameter.Parameter(basename=nm, value=val, fixed=True)
                    )

                if args.force_GR:
                    # Lock to GR
                    for nm in ["beta_1", "lambda_1", "exp_s"]:
                        cosmo.init.params[nm].update(fixed=True, value=0.0)
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
                        delta=100,
                    )
                    #cosmo.init.params["lambda_1"].update(
                    #    fixed=True, value=100)
                    cosmo.init.params["exp_s"].update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (0.0, 5.0)},
                        ref={"dist": "norm", "loc": 2.0, "scale": 0.3},
                        delta=0.5,
                    )

            elif args.mg_variant == "binning":
                # μ(z, k) binned variant: μ1..μ4 + fixed transition knobs
                for nm in ["mu1", "mu2", "mu3", "mu4"]:
                    cosmo.init.params.data.append(
                        parameter.Parameter(basename=nm, value=1.0, fixed=True)
                    )
                    cosmo.init.params[nm].update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (-2.0, 4.0)},
                        ref={"dist": "norm", "loc": 1.0, "scale": 0.05},
                        delta=0.5,
                    )

                # Transition redshift / scale and smoothing parameters
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
                    cosmo.init.params.data.append(
                        parameter.Parameter(basename=nm, value=val, fixed=True)
                    )

            else:
                raise ValueError(f"Unknown mg-variant {args.mg_variant!r}.")

        # Template & theory for this tracer
        template = DirectPowerSpectrumTemplate(z=z_tracer, fiducial=DESI(), cosmo=cosmo)
        theory = fkptTracerPowerSpectrumMultipoles()
        if args.beyond_eds:
            beyond_eds = True
        else:
            beyond_eds = False
        if args.priors_basis=="standard":
            sigma8_fid = None
        elif args.priors_basis=="physical":
            sigma8_fid = sigma8_fid_val
        theory.init.update(freedom=args.freedom, prior_basis=args.priors_basis,
                           tracer=tracer_tag, template=template,
                           k=k_out, ells=list(ells), b3_coev=True,
                           model=args.MG_model, mg_variant=args.mg_variant,
                           beyond_eds=beyond_eds, rescale_PS=False, sigma8_fid=sigma8_fid
                           )

        # Emulator file path (built from requested ells, as in your original script)
        emu_path = args.emu_dir / emu_filename(
            file_tag, k_out, ells,
            beyond_eds=beyond_eds,
            redshift_bins=args.redshift_bins,
            scale_bins=args.scale_bins,
            mg_variant=args.mg_variant,
            kc=args.kc,
            scale_bins_method=args.scale_bins_method
        )

        # Create emulator?
        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emulator] ({file_tag}) exists → {emu_path.name}")
            else:
                print(f"[Emulator] ({file_tag}) fitting Taylor emulator (finite, order=4)…")
                _ = theory.pt()  # warm up
                emu_engine = TaylorEmulatorEngine(method='finite', order=6)
                emu = Emulator(theory.pt, engine=emu_engine)
                emu.set_samples()
                emu.fit()
                emu.save(str(emu_path))
                print(f"[Emulator] ({file_tag}) saved → {emu_path.name}")

        # Use emulator? (kept exactly like your original, including the probe try/except)
        if args.use_emu:
            if rank == 0 and not emu_path.exists():
                raise FileNotFoundError(f"[Emulator] Missing {emu_path} for {file_tag}. Run with --create-emu.")
            comm.Barrier()
            emu_loaded = EmulatedCalculator.load(str(emu_path))
            # sync params from cosmology into emulator
            for p in cosmo.init.params:
                if p in emu_loaded.init.params:
                    emu_loaded.init.params.set(p)

            # fragile probe retained as-is (you said to keep previous behavior)
            try:
                y0 = emu_loaded()
                Nraw_guess = k_out.size
                present_from_emu = infer_present_ells(y0.size // max(1, Nraw_guess) * len(ells))
                present_from_emu = present_from_emu if set(ells).issubset(present_from_emu) else present_ells
                if set(ells) != set(present_from_emu):
                    out_idx = output_indices_for_ells(present_from_emu, ells, Nraw_guess)
                    emu_loaded = SlicedEmu(emu_loaded, out_idx)
            except Exception:
                # If probing fails, assume it already matches
                pass

            theory.init.update(pt=emu_loaded)
            if rank == 0:
                print(f"[{file_tag}] PT backend:", type(theory.pt).__name__)

        # ----------------------------------------
        # Time to add the nuisance parameters
        # ----------------------------------------
    
        # detect whether we're in the physical prior basis
        is_physical = getattr(theory, "is_physical_prior", False) or (
            "prior_basis" in theory.options
            and theory.options["prior_basis"] == "physical"
        )
    
        suffix = "p" if is_physical else ""  # parameters become b1p, b2p, ... in physical basis
    
        def pname(base):
            return f"{base}{suffix}"
    
        # ----------------------------------------
        # 1) tracer priors (b1, b2, etc.)
        # ----------------------------------------
        b1_name = pname("b1")
        b2_name = pname("b2")
        bs2_name = pname("bs2")
        b3nl_name = pname("b3nl")
    
        # b1 prior: center ref around your fiducial b1
        if b1_name in theory.params:
            # In the physical basis b1p ≈ b1 * sigma8(z); if you want to be fancy you can
            # map Eulerian b1 -> b1p using sigma8_fid, but as a simple option you can just
            # shift the reference and keep the physical prior.
            theory.params[b1_name].update(
                # leave the physical prior limits as set in fkptTracerPowerSpectrumMultipoles._params
                ref={"dist": "norm", "loc": b1_val, "scale": 0.05},
            )
    
        # apply weak priors to higher-order bias parameters
        for base_name, value in [
            ("b2", b2_val),
            ("bs2", 0.0),
            ("b3nl", 0.0),
        ]:
            full_name = pname(base_name)
            if full_name in theory.params:
                theory.params[full_name].update(
                    ref={"dist": "norm", "loc": value, "scale": 0.1},
                    prior={"dist": "uniform", "limits": (-50.0, 50.0)},
                )
    
        # ----------------------------------------
        # 2) shot noise / PshotP
        # ----------------------------------------
        pshotp_name = pname("PshotP")
        if pshotp_name in theory.params:
            theory.params[pshotp_name].update(fixed=True, value=10000.0)
    
        # ----------------------------------------
        # 3) mark nuisance parameters (alphas) as analytically marginalized
        # ----------------------------------------
        alpha_basenames = ["alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot"]
        alpha_fullnames = [pname(b) for b in alpha_basenames]
    
        for p in theory.params.select(basename=alpha_fullnames):
            if p.varied:
                p.update(derived=".marg")
                # if you wanted numeric marg instead, you would set fixed=False, etc.
    
        # ----------------------------------------
        # 4) ensure all FKPT nuisance parameters are namespaced per tracer
        # ----------------------------------------
        names_to_namespace = [
            "b1", "b2", "bs2", "b3nl",
            "alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot",
        ]
    
        fullnames_to_namespace = [pname(b) for b in names_to_namespace]
    
        for p in theory.params.select(basename=fullnames_to_namespace):
            p.update(namespace=namespace)

       # # tracer priors
       # if 'b1' in theory.params:
       #     theory.params['b1'].update(
       #         value=b1_val,
       #         ref={'dist': 'norm', 'loc': b1_val, 'scale': 0.05},
       #         prior={'dist': 'uniform', 'limits': (0.0, 10.0)}
       #     )
       # theory.params['PshotP'].update(fixed=True, value=10000)
#
       # # apply weak priors to higher-order bias parameters
       # for name in ['b2', 'bs2', 'b3nl']:
       #     if name == 'b2':
       #         value = b2_val
       #     else:
       #         value = 0.0
       #     if name in theory.params:
       #         theory.params[name].update(
       #             ref={'dist': 'norm', 'loc': value, 'scale': 0.1},
       #             prior={'dist': 'uniform', 'limits': (-50.0, 50.0)}
       #         )
#
       # # mark nuisance parameters marginalized (alpha*, shot noise terms) 
       # for p in theory.params.select(basename=['alpha0', 'alpha2', 'alpha4', 'alpha0shot', 'alpha2shot']):
       #     if p.varied:
       #         p.update(derived='.marg')
       #         #p.update(fixed=False, value=0.0)
#
       # #theory.params['alpha0'].update(fixed=True, value = 3.0)
       # #theory.params['alpha2'].update(fixed=True, value = -1.0)
       # #theory.params['alpha4'].update(fixed=True, value = 0.0)
       # #theory.params['alpha0shot'].update(fixed=True, value = 0.08)
       # #theory.params['alpha2shot'].update(fixed=True, value = -2.0)
#
       # # ensure *all* FKPT nuisance parameters are namespaced per tracer
       # for p in theory.params.select(basename=[
       #     'b1', 'b2', 'bs2', 'b3nl', 'alpha0', 'alpha2', 'alpha4', 'alpha0shot', 'alpha2shot'
       # ]):
       #     p.update(namespace=namespace)

        # Observable & likelihood
        observable = TracerPowerSpectrumMultipolesObservable(
            data=data_vec, theory=theory, klim=klim, ells=list(ells),
        )
        covmeta = [{'name': 'PowerSpectrumMultipoles', 'x': [k_out]*len(ells), 'projs': list(ells)}]
        covariance = ObservableCovariance(cov_mat, observables=covmeta)
        lk = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance, name=namespace)
        likelihoods.append(lk)
        print("appended", namespace, "likelihood...")

    if args.create_emu:
        comm.Barrier()
        if rank == 0:
            print("[Emulator] Done. Exiting because --create-emu was set.")
        return

    likelihood = sum(likelihoods)

    outdir = args.chains_dir
    save_pattern = str(outdir / f"{prefix}_*.npy")  # desilike expands * to 0..N-1

    # ---------- Run mode ----------
    if args.mode == "mcmc":
        if args.resume:
            existing = sorted(glob(str(outdir / f"{prefix}_*.npy")))
            if not existing:
                print(f"[resume/mcmc] No existing files found for {prefix}; starting fresh 4 chains.")
                chains_arg = 4
            else:
                print(f"[resume/mcmc] Resuming {len(existing)} chains:")
                for f in existing:
                    print("  -", f)
                chains_arg = existing  # pass list to resume
            sampler = MCMCSampler(
                likelihood,
                chains=chains_arg,
                seed=42,
                save_fn=save_pattern,
                mpicomm=MPI.COMM_WORLD,
                ref_scale=args.ref_scale,
            )
        else:
            sampler = MCMCSampler(
                likelihood,
                chains=4,
                seed=42,
                save_fn=save_pattern,
                mpicomm=MPI.COMM_WORLD,
                ref_scale=args.ref_scale,
            )
        sampler.run(check={'max_eigen_gr': 0.1}, check_every=1000, max_iterations=50000)

    elif args.mode == "nautilus":
        # For nested sampling, usually a single run per MPI-world is enough.
        if args.resume:
            existing = sorted(glob(str(outdir / f"{prefix}_*.npy")))
            if not existing:
                print(f"[resume/nautilus] No existing files found for {prefix}; starting fresh nested run.")
                chains_arg = 1
            else:
                print(f"[resume/nautilus] Resuming Nautilus from:")
                for f in existing:
                    print("  -", f)
                chains_arg = existing  # pass list of existing chains to resume
        else:
            chains_arg = 1

        sampler = NautilusSampler(
            likelihood,
            #chains=chains_arg,
            seed=42,
            save_fn=save_pattern,    # required: Nautilus uses this to write .nautilus.state.h5
            #mpicomm=MPI.COMM_WORLD,
            #ref_scale=args.ref_scale,
            #nlive=args.nlive,
        )
        # Use Nautilus' own defaults for min_iterations / max_iterations / checks
        sampler.run()

    elif args.mode == "dynesty-static":
        # Static NestedSampler (evidence + posterior)
        outdir = args.chains_dir
        save_pattern = str(outdir / f"{prefix}_*.npy")

        if args.resume:
            existing = sorted(glob(str(outdir / f"{prefix}_*.npy")))
            if not existing:
                print(f"[resume/dynesty-static] No existing files found for {prefix}; starting fresh nested run.")
                chains_arg = 1
            else:
                print(f"[resume/dynesty-static] Resuming from:")
                for f in existing:
                    print("  -", f)
                chains_arg = existing  # list of existing chains to resume
        else:
            chains_arg = 1  # usually one dynesty run per MPI world

        sampler = StaticDynestySampler(
            likelihood,
            chains=chains_arg,
            seed=42,
            save_fn=save_pattern,     # dynesty wrapper uses this to write .dynesty.state
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
            nlive=args.nlive,         # number of live points
            # You can also pass: bound='multi', sample='auto', update_interval=None, ...
        )
        # Let dynesty’s own stopping criteria handle convergence
        sampler.run()

    elif args.mode == "dynesty-dynamic":
        # DynamicNestedSampler (posterior-oriented)
        outdir = args.chains_dir
        save_pattern = str(outdir / f"{prefix}_*.npy")

        if args.resume:
            existing = sorted(glob(str(outdir / f"{prefix}_*.npy")))
            if not existing:
                print(f"[resume/dynesty-dynamic] No existing files found for {prefix}; starting fresh nested run.")
                chains_arg = 1
            else:
                print(f"[resume/dynesty-dynamic] Resuming from:")
                for f in existing:
                    print("  -", f)
                chains_arg = existing
        else:
            chains_arg = 1

        sampler = DynamicDynestySampler(
            likelihood,
            chains=chains_arg,
            seed=42,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
            nlive=args.nlive,
            # bound / sample / update_interval use defaults for now
        )
        sampler.run()

    elif args.mode == "map":
        profiler = MinuitProfiler(
            likelihood,
            gradient=args.gradient,
            rescale=args.rescale,
            covariance=args.covariance,
            save_fn=str(profiles_out),
            # mpicomm=MPI.COMM_WORLD,  # uncomment if you really want MPI for profiling
            ref_scale=args.ref_scale
        )

        profiles = profiler.maximize(
            niterations=args.nstart,      # defaults to MPI size if None
            max_iterations=args.max_calls
        )

        # helpers to pick the best row
        def _np(arr_like):
            return np.asarray(arr_like[()])

        def _pick_max(arr):
            a = _np(arr)
            if a.ndim == 0:
                return 0, float(a.item())
            idx = int(np.nanargmax(a))
            return idx, float(a[idx])

        # choose MAP row using logposterior
        argmax, logpost = _pick_max(profiles.bestfit[profiles.bestfit._logposterior])

        try:
            _, loglike = _pick_max(profiles.bestfit[profiles.bestfit._loglikelihood])
        except KeyError:
            loglike = np.nan

        try:
            _, logprior = _pick_max(profiles.bestfit[profiles.bestfit._logprior])
        except KeyError:
            logprior = (logpost - loglike) if np.isfinite(loglike) else np.nan

        if rank == 0:
            print(f"logpost = {logpost:.6g}")
            if np.isfinite(loglike):
                print(f"loglike = {loglike:.6g}, logprior = {logprior:.6g}")

            # Pretty-print MAP params (skip log fields)
            best_map = profiles.bestfit.choice(index=argmax, input=True, return_type='dict')
            skip = {
                profiles.bestfit._loglikelihood,
                profiles.bestfit._logprior,
                profiles.bestfit._logposterior,
            }
            print("\n=== MAP parameters ===")
            for k, v in best_map.items():
                if k not in skip:
                    print(f"{k:20s} = {float(np.squeeze(v)):.6g}")
    else:
        raise ValueError(f"Unknown --mode {args.mode!r}; expected 'mcmc', 'nautilus', or 'map'")

if __name__ == "__main__":
    main()