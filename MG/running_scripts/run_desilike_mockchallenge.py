import os,sys
#sys.path.insert(0, f"{os.getcwd()[:-18]}")
#sys.path.insert(0, f"{os.getcwd()}/MG/running_scripts")
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

    # APscaling convenience
    prior_basis_cli = args.priors_basis
    if args.use_apscaling:
        prior_basis_cli = "APscaling"

    h_fid_global = None
    if prior_basis_cli == "APscaling":
        if args.h_fid is not None:
            h_fid_global = float(args.h_fid)
        else:
            fid = DESI()
            try:
                h_fid_global = float(getattr(fid, "h", fid["h"]))
            except Exception:
                raise RuntimeError("Could not infer h_fid from DESI(); pass --h-fid.")

    # Priors on ns / omega_b
    ns_prior  = None if args.skip_ns_prior else {'dist': 'norm', 'loc': 0.9649, 'scale': 0.02}
    bbn_prior = None if args.skip_bbn_prior else {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00055}

    # Enforce consistent binning usage
    if args.scale_bins and not args.redshift_bins:
        raise ValueError("--scale-bins requires --redshift-bins")
    if (args.redshift_bins or args.scale_bins) and args.mg_variant != "binning":
        raise ValueError("--redshift-bins/--scale-bins only implemented for mg-variant='binning'.")
    if args.mg_variant == "binning" and not (args.redshift_bins or args.scale_bins):
        raise ValueError("mg-variant='binning' requires --redshift-bins and/or --scale-bins.")

    # ---------- Build chain prefix ----------
    prefix = args.chain_prefix
    prefix += "_beds" if args.beyond_eds else "_eds"
    prefix += "_emu" if args.use_emu else ""
    prefix += f"_{args.freedom}_{prior_basis_cli}_{fid_model}"
    prefix += f"_l{''.join(map(str, ells))}"
    prefix += f"_{args.mg_variant}" if not args.force_GR else "_GR"

    if rank == 0:
        print("Job will start with prefix:", prefix)

    profiles_out = args.profiles_out or (args.chains_dir / f"{prefix}_profiles.npy")

    # ---------- Load / build likelihoods ----------
    likelihoods = []
    cosmo = None

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

        # Shared cosmology
        if cosmo is None:
            cosmo = Cosmoprimo(
                engine='isitgr',
                redshift_bins=args.redshift_bins,
                scale_bins=args.scale_bins,
                scale_bins_method=args.scale_bins_method,
            )

            # base params
            if 'tau_reio' in cosmo.init.params:
                cosmo.init.params['tau_reio'].update(fixed=True, value=0.0544)
            cosmo.init.params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}

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

            # MG knobs
            if args.mg_variant == "mu_OmDE":
                cosmo.init.params.data.append(parameter.Parameter(basename="mu0", value=0.0, fixed=True))
                cosmo.init.params["mu0"].update(
                    fixed=True if args.force_GR else False,
                    prior=None if args.force_GR else {"dist": "uniform", "limits": (-3.0, 3.0)},
                    ref={"dist": "norm", "loc": 0.0, "scale": 0.01},
                    delta=0.25,
                )

            elif args.mg_variant == "BZ":
                for nm, val in [("beta_1", 0.0), ("lambda_1", 0.0), ("exp_s", 0.0)]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=val, fixed=True))
                if not args.force_GR:
                    cosmo.init.params["beta_1"].update(
                        fixed=False, prior={"dist": "uniform", "limits": (-5.0, 5.0)},
                        ref={"dist": "norm", "loc": 1.0, "scale": 0.1}, delta=0.3
                    )
                    cosmo.init.params["lambda_1"].update(
                        fixed=False, prior={"dist": "uniform", "limits": (0.0, 1e6)},
                        ref={"dist": "norm", "loc": 100.0, "scale": 100.0}, delta=100
                    )
                    cosmo.init.params["exp_s"].update(
                        fixed=False, prior={"dist": "uniform", "limits": (0.0, 5.0)},
                        ref={"dist": "norm", "loc": 2.0, "scale": 0.3}, delta=0.5
                    )

            elif args.mg_variant == "BZ_fR":
                for nm, val in [("beta_1", 4.0/3.0), ("lambda_1", 100.0), ("exp_s", 4.0)]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=val, fixed=True))
                if args.force_GR:
                    cosmo.init.params["beta_1"].update(fixed=True, value=1.0)
                    cosmo.init.params["lambda_1"].update(fixed=True, value=0.0)
                    cosmo.init.params["exp_s"].update(fixed=True, value=4.0)
                else:
                    cosmo.init.params["lambda_1"].update(
                        fixed=False, prior={"dist": "uniform", "limits": (0.0, 1e6)},
                        ref={"dist": "norm", "loc": 30.0, "scale": 10.0}, delta=10.0
                    )

            elif args.mg_variant == "binning":
                for nm in ["mu1", "mu2", "mu3", "mu4"]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=1.0, fixed=True))
                    cosmo.init.params[nm].update(
                        fixed=False, prior={"dist": "uniform", "limits": (-2.0, 4.0)},
                        ref={"dist": "norm", "loc": 1.0, "scale": 0.05}, delta=0.5
                    )
                for nm, val in [
                    ("z_div", 1.0), ("z_TGR", 2.0), ("z_tw", 0.05),
                    ("k_tw", 0.01), ("k_c", args.kc),
                    ("Sigma1", 1.0), ("Sigma2", 1.0), ("Sigma3", 1.0), ("Sigma4", 1.0),
                ]:
                    cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=val, fixed=True))
            else:
                raise ValueError(f"Unknown mg-variant {args.mg_variant!r}.")

        # Template & theory for this tracer
        template = DirectPowerSpectrumTemplate(z=z_tracer, fiducial=DESI(), cosmo=cosmo)

        theory = fkptTracerPowerSpectrumMultipoles()

        fkpt_mg_variant = "BZ" if args.mg_variant == "BZ_fR" else args.mg_variant
        if args.force_GR:
            # If your pyfkpt supports mg_variant='GR', you can switch here.
            # Otherwise mu0=0 (or BZ=0) is already GR.
            fkpt_mg_variant = fkpt_mg_variant

        # provide sigma8_fid for physical/APscaling so A=(sigma8/sigma8_fid)^2 is meaningful
        sigma8_fid = sigma8_fid_val if prior_basis_cli in ["physical", "physical_velocileptors", "APscaling"] else None

        theory.init.update(
            freedom=args.freedom,
            prior_basis=prior_basis_cli,
            tracer=tracer_tag,
            template=template,
            k=k_out,
            ells=list(ells),
            b3_coev=True,
            model=args.MG_model,
            mg_variant=fkpt_mg_variant,
            beyond_eds=bool(args.beyond_eds),
            rescale_PS=False,
            #sigma8_fid=sigma8_fid,
            h_fid=(h_fid_global if prior_basis_cli == "APscaling" else None),
        )

        # Emulator path
        emu_path = args.emu_dir / emu_filename(
            file_tag, k_out, ells,
            beyond_eds=bool(args.beyond_eds),
            redshift_bins=args.redshift_bins,
            scale_bins=args.scale_bins,
            mg_variant=fkpt_mg_variant,
            kc=args.kc,
            scale_bins_method=args.scale_bins_method,
        )

        # Create emulator?
        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emulator] ({file_tag}) exists → {emu_path.name}")
            else:
                print(f"[Emulator] ({file_tag}) fitting Taylor emulator (finite, order=6)…")
                _ = theory.pt()  # warmup
                emu_engine = TaylorEmulatorEngine(method='finite', order=6)
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

        # -------------------------------------------------
        # Tracer-specific prior tweaks + namespacing
        # -------------------------------------------------
        is_physical = bool(getattr(theory, "is_physical_prior", False))
        suffix = "p" if is_physical else ""

        def pname(base: str) -> str:
            return f"{base}{suffix}"

        # Center b1 ref per tracer (works for b1 or b1p)
        b1_name = pname("b1")
        if b1_name in theory.params:
            if is_physical:
                # b1p ~ b1 * sigma8_fid (simple centering; your class still defines priors)
                theory.params[b1_name].update(ref={"dist": "norm",
                                                  "loc": b1_val * (sigma8_fid_val if sigma8_fid_val is not None else 1.0),
                                                  "scale": 0.05 * (sigma8_fid_val if sigma8_fid_val is not None else 1.0)})
            else:
                theory.params[b1_name].update(ref={"dist": "norm", "loc": b1_val, "scale": 0.05})

        # Give weak ref centers to b2 / etc if they exist (won't override fixed params)
        for base_name, value in [("b2", b2_val), ("bs2", 0.0), ("b3nl", 0.0)]:
            nm = pname(base_name)
            if nm in theory.params:
                theory.params[nm].update(ref={"dist": "norm", "loc": value, "scale": 0.1})

        # Fix PshotP (though PT side overwrites PshotP=1/nd anyway)
        pshot = pname("PshotP")
        if pshot in theory.params:
            theory.params[pshot].update(fixed=True, value=10000.0)

        # Mark alpha nuisance parameters as analytically marginalized
        alpha_bases = ["alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot"]
        alpha_full = [pname(b) for b in alpha_bases]
        for ppar in theory.params.select(basename=alpha_full):
            if ppar.varied:
                ppar.update(derived=".marg")

        # Ensure nuisance params are namespaced (include APscaling knobs bK2/btd)
        names_to_namespace = [
            "b1", "b2", "bs2", "b3nl",
            "alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot",
            "ctilde",
        ]
        if prior_basis_cli == "APscaling":
            names_to_namespace += ["bK2", "btd"]

        for ppar in theory.params.select(basename=[pname(b) for b in names_to_namespace]):
            ppar.update(namespace=namespace)

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
        sys.exit()

    likelihood = sum(likelihoods)

    outdir = args.chains_dir
    save_pattern = str(outdir / f"{prefix}_*.npy")

    # ---------- Run mode ----------
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

