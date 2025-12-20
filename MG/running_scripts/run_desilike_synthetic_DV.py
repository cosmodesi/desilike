#!/usr/bin/env python3
from __future__ import annotations
import os
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
    p.add_argument("--workdir", type=Path,
                   default=Path("/n/home12/cgarciaquintero/DESI/MG_validation/fR_noiseless_desilike"),
                   help="Directory containing synthetic data and covariances.")
    p.add_argument("--kmin-cut", type=float, default=0.02, help="k_min cut to align to data length.")
    p.add_argument("--ells", type=str, default="0,2,4", help='Multipoles to use, e.g. "0,2" or "0,2,4".')
    p.add_argument("--freedom", choices=["max", "min"], default="max",
                   help="fkpt nuisance freedom. 'min' typically omits bs/b3.")
    p.add_argument("--fid-model", type=str, default="LCDM", help='Underlying model tag for IO (e.g., LCDM or F4).')

    # IMPORTANT: basis selection (delegates nuisance priors/mapping to your internal code)
    p.add_argument("--prior-basis",
                   choices=["standard", "physical", "physical_velocileptors", "APscaling"],
                   default="standard",
                   help="Basis for nuisance parametrization; priors/mapping set internally in fkpt/full_shape.")
    p.add_argument("--h-fid", type=float, default=None,
                   help="Required for --prior-basis APscaling (fiducial h used to generate data).")

    # cosmology external priors toggles (keep as before)
    p.add_argument("--skip-ns-prior", action="store_true", help="Do not apply a prior on n_s.")
    p.add_argument("--skip-bbn-prior", action="store_true", help="Do not apply the BBN prior on omega_b.")
    p.add_argument("--resume", action="store_true", help="Resume from existing chain files matching the built prefix.")

    # MG toggles
    p.add_argument("--force-GR", action="store_true", help="Force GR run.")
    p.add_argument("--MG-model", choices=["LCDM", "HS", "HDKI"], default="LCDM", help="Base model.")
    p.add_argument("--mg-variant", choices=["mu_OmDE", "BZ", "binning"], default="mu_OmDE")
    p.add_argument("--beyond-eds", action="store_true", help="Enable beyond-EdS kernels (default False).")

    # binning toggles
    p.add_argument("--redshift-bins", action="store_true", help="Enable redshift binning.")
    p.add_argument("--scale-bins-method", type=str, default="traditional", help="Scale binning method tag.")
    p.add_argument("--scale-bins", action="store_true", help="Enable scale binning (requires --redshift-bins).")
    p.add_argument("--kc", type=float, default=0.1, help="Scale for scale transition between bins (k_c).")

    # MAP / MinuitProfiler options
    p.add_argument("--profiles-out", type=Path, default=None, help="Path to save Profiles (.npy). Defaults to chains-dir/<prefix>_profiles.npy")
    p.add_argument("--nstart", type=int, default=None, help="Number of independent Minuit starts; default = MPI size.")
    p.add_argument("--max-calls", type=int, default=int(1e5), help="Max function calls per Minuit run.")
    p.add_argument("--gradient", action="store_true", help="Use JAX gradients if available.")
    p.add_argument("--rescale", action="store_true", help="Internally rescale params using covariance/proposals.")
    p.add_argument("--covariance", type=str, default=None, help="Path to covariance/chain to set scaling when --rescale is on.")
    p.add_argument("--ref-scale", type=float, default=1.2, help="Scale factor for parameter reference dists (for profiler starts).")

    # Nautilus-specific
    p.add_argument("--nlive", type=int, default=800, help="Number of live points for the Nautilus nested sampler.")

    # emulator order
    p.add_argument("--emu-order", type=int, default=6, help="Taylor emulator order (finite).")

    return p.parse_args()


def parse_ells_str(s: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    allowed = {0, 2, 4}
    if not set(vals).issubset(allowed) or len(vals) == 0:
        raise ValueError(f"--ells must be a comma-separated subset of 0,2,4; got {s!r}")
    return tuple(sorted(vals))


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


def emu_filename(tag: str,
                 k: np.ndarray,
                 ells: tuple[int, ...],
                 beyond_eds: bool,
                 redshift_bins: bool,
                 scale_bins: bool,
                 mg_variant: str,
                 prior_basis: str,
                 kc: float | None = None,
                 scale_bins_method: str | None = None) -> str:
    """Per-tracer PT emulator filename.
    """
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
        if mg_variant == "mu_OmDE":
            mode = "mu0"
        elif mg_variant == "BZ":
            mode = "BZ"
        elif mg_variant == "binning":
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


def main():
    args = parse_args()
    setup_logging("info")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    args.chains_dir.mkdir(parents=True, exist_ok=True)
    args.emu_dir.mkdir(parents=True, exist_ok=True)

    workdir = args.workdir

    # Tracers: (file tag, tracer tag, z_eff)
    tracer_table = [
        ("BGS",  "BGS", 0.295),
        ("LRG1", "LRG", 0.510),
        ("LRG2", "LRG", 0.706),
        ("LRG3", "LRG", 0.919),
        ("ELG",  "ELG", 1.317),
        ("QSO",  "QSO", 1.492),
    ]

    ells = parse_ells_str(args.ells)
    fid_model = str(args.fid_model)
    prior_basis = str(args.prior_basis)

    # APscaling requires h_fid
    h_fid_global = None
    if prior_basis == "APscaling":
        if args.h_fid is not None:
            h_fid_global = float(args.h_fid)
        else:
            # fallback to DESI fiducial; ok for "real data" style runs but for synthetic you should pass --h-fid
            fid = DESI()
            try:
                h_fid_global = float(getattr(fid, "h", fid["h"]))
            except Exception:
                raise RuntimeError("Could not infer h_fid from DESI(); pass --h-fid explicitly.")

    # cosmology external priors (kept as in your old script)
    ns_prior  = None if args.skip_ns_prior else {'dist': 'norm', 'loc': 0.9649, 'scale': 0.02}
    bbn_prior = None if args.skip_bbn_prior else {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00055}

    # Enforce binning logic
    if args.scale_bins and not args.redshift_bins:
        raise ValueError("--scale-bins requires --redshift-bins")
    if (args.redshift_bins or args.scale_bins) and args.mg_variant != "binning":
        raise ValueError("--redshift-bins / --scale-bins are only implemented for mg-variant='binning'.")
    if args.mg_variant == "binning" and not (args.redshift_bins or args.scale_bins):
        raise ValueError("mg-variant='binning' requires --redshift-bins and/or --scale-bins.")

    # ---------- Build chain prefix ----------
    prefix = args.chain_prefix
    prefix += "_beds" if args.beyond_eds else "_eds"
    prefix += "_emu" if args.use_emu else ""
    prefix += f"_{args.freedom}_{prior_basis}_{fid_model}"

    if args.redshift_bins or args.scale_bins:
        prefix += "_binning"
        if args.redshift_bins:
            prefix += "_z"
        if args.scale_bins:
            prefix += f"_k_kc{args.kc:g}"
    else:
        if args.force_GR:
            prefix += "_GR"
        else:
            prefix += "_mu0" if args.mg_variant == "mu_OmDE" else f"_{args.mg_variant}"

    prefix += f"_l{''.join(map(str, ells))}"

    if rank == 0:
        print("Job will start with prefix:", prefix)

    profiles_out = args.profiles_out or (args.chains_dir / f"{prefix}_profiles.npy")
    profiles_out.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Pipeline ----------
    likelihoods = []

    cosmo = None
    for file_tag, tracer_tag, z_tracer in tracer_table:
        namespace = file_tag.lower()

        # load data
        k_all   = np.loadtxt(workdir / f"{file_tag}_{fid_model}_k.txt")
        P_all   = np.loadtxt(workdir / f"{file_tag}_{fid_model}_P0P2P4.txt")
        cov_all = np.loadtxt(workdir / f"{file_tag}_{fid_model}_cov.txt")

        present_ells = infer_present_ells(P_all.size)
        if not set(ells).issubset(set(present_ells)):
            raise RuntimeError(f"[{file_tag}] requested ells {ells} not available; present={present_ells}")

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

        # Shared cosmology (same as your old script; keep external cosmo priors here)
        if cosmo is None:
            cosmo = Cosmoprimo(engine='isitgr',
                               redshift_bins=args.redshift_bins,
                               scale_bins=args.scale_bins,
                               scale_bins_method=args.scale_bins_method)

            if 'tau_reio' in cosmo.init.params:
                cosmo.init.params['tau_reio'].update(fixed=True)

            if 'N_eff' in cosmo.init.params:
                cosmo.init.params['N_eff'].update(fixed=True, value=3.046)
            if 'm_ncdm' in cosmo.init.params:
                cosmo.init.params['m_ncdm'].update(value=0.06, fixed=True)

            if ns_prior is not None and 'n_s' in cosmo.init.params:
                cosmo.init.params['n_s'].update(fixed=False, prior=ns_prior,
                                                ref={'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, delta=0.01)
            if bbn_prior is not None and 'omega_b' in cosmo.init.params:
                cosmo.init.params['omega_b'].update(fixed=False, prior=bbn_prior, delta=0.0015)

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

            # MG parameters exposed for PT to read
            if args.mg_variant == "mu_OmDE":
                cosmo.init.params.data.append(parameter.Parameter(basename="mu0", value=0.0, fixed=True))
                cosmo.init.params["mu0"].update(
                    fixed=True if args.force_GR else False,
                    value=0.0,
                    prior=None if args.force_GR else {"dist": "uniform", "limits": (-3.0, 3.0)},
                    ref={"dist": "norm", "loc": 0.0, "scale": 0.01},
                    delta=0.25,
                )
            elif args.mg_variant == "BZ":
                for nm, val in [("beta_1", 0.0), ("lambda_1", 0.0), ("exp_s", 0.0)]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=val, fixed=True))
                if args.force_GR:
                    for nm in ["beta_1", "lambda_1", "exp_s"]:
                        cosmo.init.params[nm].update(fixed=True, value=0.0)
                else:
                    cosmo.init.params["beta_1"].update(fixed=False, prior={"dist": "uniform", "limits": (-5.0, 5.0)},
                                                       ref={"dist": "norm", "loc": 1.0, "scale": 0.1}, delta=0.3)
                    cosmo.init.params["lambda_1"].update(fixed=False, prior={"dist": "uniform", "limits": (0.0, 1e6)},
                                                         ref={"dist": "norm", "loc": 100.0, "scale": 100.0}, delta=100)
                    cosmo.init.params["exp_s"].update(fixed=False, prior={"dist": "uniform", "limits": (0.0, 5.0)},
                                                      ref={"dist": "norm", "loc": 2.0, "scale": 0.3}, delta=0.5)
            elif args.mg_variant == "binning":
                for nm in ["mu1", "mu2", "mu3", "mu4"]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=1.0, fixed=True))
                    cosmo.init.params[nm].update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (-2.0, 4.0)},
                        ref={"dist": "norm", "loc": 1.0, "scale": 0.05},
                        delta=0.5,
                    )
                for nm, val in [
                    ("z_div", 1.0), ("z_TGR", 2.0), ("z_tw", 0.05), ("k_tw", 0.01), ("k_c", args.kc),
                    ("Sigma1", 1.0), ("Sigma2", 1.0), ("Sigma3", 1.0), ("Sigma4", 1.0),
                ]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=val, fixed=True))
            else:
                raise ValueError(f"Unknown mg-variant {args.mg_variant!r}.")

        # Template & theory for this tracer
        template = DirectPowerSpectrumTemplate(z=z_tracer, fiducial=DESI(), cosmo=cosmo)
        theory = fkptTracerPowerSpectrumMultipoles()

        init_kwargs = dict(
            freedom=args.freedom,
            prior_basis=prior_basis,     # <-- key change
            tracer=tracer_tag,
            template=template,
            k=k_out,
            ells=list(ells),
            b3_coev=True,
            model=args.MG_model,
            mg_variant=args.mg_variant,
            beyond_eds=bool(args.beyond_eds),
            rescale_PS=False,
            shotnoise=1e4,               # keep consistent with your notebooks
        )
        if prior_basis == "APscaling":
            init_kwargs["h_fid"] = h_fid_global   # <-- required; b1_fid left to internal inference
        theory.init.update(**init_kwargs)

        # Emulator file path
        emu_path = args.emu_dir / emu_filename(
            tag=file_tag,
            k=k_out,
            ells=ells,
            beyond_eds=bool(args.beyond_eds),
            redshift_bins=args.redshift_bins,
            scale_bins=args.scale_bins,
            mg_variant=args.mg_variant,
            prior_basis=prior_basis,
            kc=args.kc,
            scale_bins_method=args.scale_bins_method,
        )

        # Create emulator?
        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emulator] ({file_tag}) exists → {emu_path.name}")
            else:
                print(f"[Emulator] ({file_tag}) fitting Taylor emulator (finite, order={args.emu_order})…")
                _ = theory.pt()  # warm up
                emu_engine = TaylorEmulatorEngine(method='finite', order=int(args.emu_order))
                emu = Emulator(theory.pt, engine=emu_engine)
                emu.set_samples()
                emu.fit()
                emu.save(str(emu_path))
                print(f"[Emulator] ({file_tag}) saved → {emu_path.name}")

        # Use emulator?
        if args.use_emu:
            if rank == 0 and not emu_path.exists():
                raise FileNotFoundError(f"[Emulator] Missing {emu_path} for {file_tag}. Run with --create-emu.")
            comm.Barrier()
            emu_loaded = EmulatedCalculator.load(str(emu_path))
            for p in cosmo.init.params:
                if p in emu_loaded.init.params:
                    emu_loaded.init.params.set(p)
            theory.init.update(pt=emu_loaded)
            if rank == 0:
                print(f"[{file_tag}] PT backend:", type(theory.pt).__name__)

        # ---- Namespace nuisance params (must handle standard vs physical/APscaling primed names) ----
        is_physical = bool(getattr(theory, "is_physical_prior", False))
        suffix = "p" if is_physical else ""

        def pname(base: str) -> str:
            return f"{base}{suffix}"

        nuis_to_namespace = [
            "b1", "b2", "bs2", "b3nl",
            "alpha0", "alpha2", "alpha4",
            "alpha0shot", "alpha2shot",
            "ctilde",
            "PshotP",
        ]
        if prior_basis == "APscaling":
            nuis_to_namespace += ["bK2", "btd"]

        # ----------------------------------------
        # mark nuisance parameters (alphas) as analytically marginalized
        # ----------------------------------------
        alpha_basenames = ["alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot"]
        alpha_fullnames = [pname(b) for b in alpha_basenames]
    
        for p in theory.params.select(basename=alpha_fullnames):
            if p.varied:
                p.update(derived=".marg")

        for par in theory.params.select(basename=[pname(nm) for nm in nuis_to_namespace]):
            par.update(namespace=namespace)

        # Observable & likelihood
        observable = TracerPowerSpectrumMultipolesObservable(
            data=data_vec, theory=theory, klim=klim, ells=list(ells),
        )
        covmeta = [{'name': 'PowerSpectrumMultipoles', 'x': [k_out]*len(ells), 'projs': list(ells)}]
        covariance = ObservableCovariance(cov_mat, observables=covmeta)
        lk = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance, name=namespace)
        likelihoods.append(lk)

        if rank == 0:
            print("appended", namespace, "likelihood...")

    if args.create_emu:
        comm.Barrier()
        if rank == 0:
            print("[Emulator] Done. Exiting because --create-emu was set.")
        return

    likelihood = sum(likelihoods)
    outdir = args.chains_dir
    save_pattern = str(outdir / f"{prefix}_*.npy")

    # ---------- Run mode ----------
    if args.mode == "mcmc":
        if args.resume:
            existing = sorted(glob(str(outdir / f"{prefix}_*.npy")))
            existing = [f for f in existing if "profiles" not in Path(f).stem]
            if rank == 0:
                if not existing:
                    print(f"[resume/mcmc] No existing files found for {prefix}; starting fresh 4 chains.")
                else:
                    print(f"[resume/mcmc] Resuming {len(existing)} chains:")
                    for f in existing:
                        print("  -", f)
            chains_arg = existing if existing else 4
        else:
            chains_arg = 4

        sampler = MCMCSampler(
            likelihood,
            chains=chains_arg,
            seed=42,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
        )
        sampler.run(check={'max_eigen_gr': 0.1}, check_every=1000, max_iterations=50000)

    elif args.mode == "nautilus":
        sampler = NautilusSampler(
            likelihood,
            seed=42,
            save_fn=save_pattern,
        )
        sampler.run()

    elif args.mode == "dynesty-static":
        sampler = StaticDynestySampler(
            likelihood,
            chains=1,
            seed=42,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
            nlive=args.nlive,
        )
        sampler.run()

    elif args.mode == "dynesty-dynamic":
        sampler = DynamicDynestySampler(
            likelihood,
            chains=1,
            seed=42,
            save_fn=save_pattern,
            mpicomm=MPI.COMM_WORLD,
            ref_scale=args.ref_scale,
            nlive=args.nlive,
        )
        sampler.run()

    elif args.mode == "map":
        profiler = MinuitProfiler(
            likelihood,
            gradient=args.gradient,
            rescale=args.rescale,
            covariance=args.covariance,
            save_fn=str(profiles_out),
            ref_scale=args.ref_scale
        )
        profiler.maximize(niterations=args.nstart, max_iterations=args.max_calls)
        if rank == 0:
            print(f"[Profiles] saved to {profiles_out}")

    else:
        raise ValueError(f"Unknown --mode {args.mode!r}")

if __name__ == "__main__":
    main()
