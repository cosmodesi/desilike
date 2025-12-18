import os,sys
from run_desilike_synthetic_DV import *

if __name__ == "__main__":
    args = parse_args()
    setup_logging("info")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    args.chains_dir.mkdir(parents=True, exist_ok=True)
    args.emu_dir.mkdir(parents=True, exist_ok=True)

    # Tracers: (file tag, tracer tag, b1 prior center, z_eff)
    ################################# might need to be adjusted, especially sigma8 value(last col)
    tracer_table = [
        ("LRG_abacusHF_EZcov", "LRG",  2.1, 0.8, -1.10, 0.538705, 'z08', 'EZmocks_LRG2ndGen_cubic', 'abacusHF_cutsky_LRG0p725'), 
        ("ELG_abacusHF_EZcov", "ELG",  1.2, 0.95, 0.03, 0.503071, 'z095','EZELG_boxz095_bin01',     'abacusHF_cutsky_ELG0p950'), 
        ("QSO_abacusHF_EZcov", "QSO",  2.1, 1.4, -0.71, 0.410860, 'z14', 'EZQSO_boxz1400_bin01',    'abacusHF_cutsky_QSO1p400'),
    ]

    ells = parse_ells_str(args.ells)
    fid_model = str(args.fid_model)

    prior_basis = args.prior_basis
    if args.use_apscaling:
        prior_basis = "APscaling"

    h_fid_global = None
    if prior_basis == "APscaling":
        if args.h_fid is not None:
            h_fid_global = float(args.h_fid)
        else:
            fid = DESI()
            try:
                h_fid_global = float(getattr(fid, "h", fid["h"]))
            except Exception:
                raise RuntimeError("Could not infer h_fid from DESI(); pass --h-fid.")

    ns_prior  = None if args.skip_ns_prior else {'dist': 'norm', 'loc': 0.9649, 'scale': 0.02}
    bbn_prior = None if args.skip_bbn_prior else {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00055}

    prefix = args.chain_prefix
    prefix += "_beds" if args.beyond_eds else "_eds"
    prefix += "_emu" if args.use_emu else ""
    prefix += f"_{args.freedom}_{prior_basis}_{fid_model}"
    prefix += f"_l{''.join(map(str, ells))}"
    prefix += f"_{args.mg_variant}" if not args.force_GR else "_GR"

    if rank == 0:
        print("Job will start with prefix:", prefix)

    profiles_out = args.profiles_out or (args.chains_dir / f"{prefix}_profiles.npy")

    # ---- Shared cosmology ----
    cosmo = Cosmoprimo(engine="isitgr")

    if "tau_reio" in cosmo.init.params:
        cosmo.init.params["tau_reio"].update(fixed=True, value=0.0544)
    if "N_eff" in cosmo.init.params:
        cosmo.init.params["N_eff"].update(fixed=True, value=3.046)
    if "m_ncdm" in cosmo.init.params:
        cosmo.init.params["m_ncdm"].update(value=0.06, fixed=True)

    if ns_prior is not None and "n_s" in cosmo.init.params:
        cosmo.init.params["n_s"].update(fixed=False, prior=ns_prior,
                                        ref={"dist": "norm", "loc": 0.9649, "scale": 0.004}, delta=0.01)
    if bbn_prior is not None and "omega_b" in cosmo.init.params:
        cosmo.init.params["omega_b"].update(fixed=False, prior=bbn_prior, delta=0.0015)

    prior_limits = {"h": (0.4, 1.0), "omega_cdm": (0.001, 0.99), "logA": (1.61, 3.91)}
    for name, scale, delta in [("h", 0.001, 0.03), ("omega_cdm", 0.001, 0.007), ("logA", 0.001, 0.1)]:
        if name in cosmo.init.params:
            lo, hi = prior_limits[name]
            par = cosmo.init.params[name]
            par.update(
                fixed=False,
                prior={"dist": "uniform", "limits": (lo, hi)},
                ref={"dist": "norm", "loc": par.value, "scale": scale},
                delta=delta,
            )

    # expose MG params (so your PT can read them from self.all_params)
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
        if not args.force_GR:
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
                fixed=False if not args.force_GR else True,
                prior=None if args.force_GR else {"dist": "uniform", "limits": (-2.0, 4.0)},
                ref={"dist": "norm", "loc": 1.0, "scale": 0.05},
                delta=0.5,
            )
        if args.force_GR:
            for nm in ["mu1", "mu2", "mu3", "mu4"]:
                cosmo.init.params[nm].update(value=1.0, fixed=True)
    elif args.mg_variant == "fR":
        cosmo.init.params.data.append(parameter.Parameter(basename="fR0", value=1e-15, fixed=True))
        cosmo.init.params["fR0"].update(
            fixed=True if args.force_GR else False,
            value=1e-15,
            prior=None if args.force_GR else {"dist": "loguniform", "limits": (1e-15, 1e-3)},
            ref={"dist": "norm", "loc": 1e-8, "scale": 1e-8},
            delta=1e-2,
        )

    likelihoods = []

    for file_tag, tracer_tag, b1_val, z_tracer, b2_val, sigma8_fid_val, zsnap, mid_ez, mid_ref in tracer_table:
        namespace = file_tag.lower()

        # load data
        ############################## adjusted to the format from the following file
        datadir = Path('/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/scoccimarro_basis/')
        k_all   = np.loadtxt(datadir / f"EZmocks/Pk/{tracer_tag}_{zsnap}" /f"Power_Spectrum_{mid_ez}_10.txt",usecols=0)
        # k-range alignment
        sel     = (args.kmin_cut<k_all)
        k_out   = k_all[sel]
        #################### define ells in advance
        present_ells = ells
        #################### Compute covariance with EZmocks
        Pk_all = []
        for i in range(1000): # LRG mock 933 had two fewer columns
            sel_pk    = (args.kmin_cut<np.loadtxt(datadir / f"EZmocks/Pk/{tracer_tag}_{zsnap}" /f"Power_Spectrum_{mid_ez}_{i+1}.txt",usecols=0))
            Pk_all.append(np.array([np.loadtxt(datadir / f"EZmocks/Pk/{tracer_tag}_{zsnap}" /f"Power_Spectrum_{mid_ez}_{i+1}.txt",usecols=2+k)[sel_pk] for k in range(len(present_ells))]).flatten())
        cov_mat = np.cov(np.array(Pk_all).T)
        #################### get data vector with Abacus 
        sel_pk = (args.kmin_cut<np.loadtxt(datadir / f"abacusHF/Pk/{tracer_tag}" /f"Power_Spectrum_{mid_ref}_0.txt",usecols=0))
        Pk_all = [np.array([np.loadtxt(datadir / f"abacusHF/Pk/{tracer_tag}" /f"Power_Spectrum_{mid_ref}_{i}.txt",usecols=2+k)[sel_pk] for k in range(len(present_ells))]).flatten() for i in range(25)]
        data_vec = np.mean(Pk_all,axis=0)

        if not set(ells).issubset(set(present_ells)):
            raise RuntimeError(f"[{file_tag}] requested ells {ells} not available; present={present_ells}")

        dk = float(np.median(np.diff(k_out))) 
        kmin_edge = float(k_out[0] - 0.5 * dk)
        kmax_edge = float(k_out[-1] + 0.5 * dk)
        klim = {L: (kmin_edge, kmax_edge, dk) for L in ells}

        template = DirectPowerSpectrumTemplate(z=z_tracer, fiducial=DESI(), cosmo=cosmo)

        theory = fkptTracerPowerSpectrumMultipoles()
        init_kwargs = dict(
            freedom=args.freedom,
            prior_basis=prior_basis,
            tracer=tracer_tag,
            template=template,
            k=k_out,
            ells=list(ells),
            b3_coev=True,
            model=args.MG_model,
            mg_variant=args.mg_variant,
            beyond_eds=bool(args.beyond_eds),
            rescale_PS=False,
            shotnoise=1e4,
        )
        if prior_basis == "APscaling":
            init_kwargs["h_fid"] = h_fid_global
            # NOTE: do NOT pass b1_fid here; let your class infer it from tracer
        theory.init.update(**init_kwargs)

        # ---- Emulator handling (PT only) ----
        emu_path = args.emu_dir / emu_filename(
            tag=file_tag, k=k_out, ells=ells,
            beyond_eds=bool(args.beyond_eds),
            mg_variant=args.mg_variant,
            prior_basis=prior_basis,
        )

        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emulator] ({file_tag}) exists → {emu_path.name}")
            else:
                print(f"[Emulator] ({file_tag}) fitting Taylor emulator (finite, order={args.emu_order}) …")
                _ = theory.pt()  # warmup / ensure PT is initialized
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

            # sync params from cosmology into emulator
            for p in cosmo.init.params:
                if p in emu_loaded.init.params:
                    emu_loaded.init.params.set(p)

            theory.init.update(pt=emu_loaded)
            if rank == 0:
                print(f"[{file_tag}] PT backend:", type(theory.pt).__name__)

        # Namespace tracer nuisance params so they don't collide
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

        for par in theory.params.select(basename=[pname(nm) for nm in nuis_to_namespace]):
            par.update(namespace=namespace)

        observable = TracerPowerSpectrumMultipolesObservable(
            data=data_vec, theory=theory, klim=klim, ells=list(ells)
        )
        covmeta = [{'name': 'PowerSpectrumMultipoles', 'x': [k_out] * len(ells), 'projs': list(ells)}]
        covariance = ObservableCovariance(cov_mat, observables=covmeta)

        lk = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance, name=namespace)
        likelihoods.append(lk)

        if rank == 0:
            print("appended", namespace, "likelihood...")

    if args.create_emu:
        comm.Barrier()
        if rank == 0:
            print("[Emulator] Done. Exiting because --create-emu was set.")
        sys.exit()

    likelihood = sum(likelihoods)
    outdir = args.chains_dir
    save_pattern = str(outdir / f"{prefix}_*.npy")

    if args.mode == "mcmc":
        if args.resume:
            all_files = sorted(glob(save_pattern))
            existing = [f for f in all_files if "profiles" not in Path(f).stem]
            if rank == 0:
                if not existing:
                    print(f"[resume/mcmc] No chain files for {prefix}; starting fresh 4 chains.")
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
        sampler.run(check={'max_eigen_gr': 0.01}, check_every=3000, max_iterations=50000)

    elif args.mode == "map":
        profiler = MinuitProfiler(
            likelihood,
            gradient=args.gradient,
            rescale=args.rescale,
            covariance=args.covariance,
            save_fn=str(profiles_out),
            ref_scale=args.ref_scale,
        )
        profiler.maximize(niterations=args.nstart, max_iterations=args.max_calls)
        if rank == 0:
            print(f"[Profiles] saved to {profiles_out}")

    else:
        raise ValueError(f"Unknown --mode {args.mode!r}.")