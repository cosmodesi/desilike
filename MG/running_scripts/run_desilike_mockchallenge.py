import os,sys
sys.path.insert(0, f"{os.getcwd()[:-18]}")
from run_desilike import *

if __name__ == "__main__":
    args = parse_args()
    setup_logging("info")

    # IO setup
    args.chains_dir.mkdir(parents=True, exist_ok=True)
    args.emu_dir.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Tracers: (file tag, tracer tag, b1 prior center, z_eff)
    ################################# might need to be adjusted, especially sigma8 value(last col)
    tracer_table = [
        ("LRG", "LRG",  2.1, 0.8, -1.10, 0.538705, 'z08', 'EZmocks_LRG2ndGen_cubic'), 
        ("ELG", "ELG",  1.2, 0.95, 0.03, 0.503071, 'z095','EZELG_boxz095_bin01'), 
        ("QSO", "QSO",  2.1, 1.4, -0.71, 0.410860, 'z14', 'EZQSO_boxz1400_bin01'),
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
    for file_tag, tracer_tag, b1_val, z_tracer, b2_val, sigma8_fid_val, zsnap, fn_mid in tracer_table:
        namespace = file_tag.lower()

        # load data
        ############################## adjusted to the format from the following file
        datadir = Path('/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/scoccimarro_basis/EZmocks/Pk/')
        k_all   = np.loadtxt(datadir / f"{tracer_tag}_{zsnap}" /f"Power_Spectrum_{fn_mid}_10.txt",usecols=0)
        # k-range alignment
        sel     = (args.kmin_cut<k_all)
        k_out   = k_all[sel]
        #################### define ells in advance
        present_ells = ells
        Pk_all = []
        for i in range(1000): # LRG mock 933 had two fewer columns
            sel_pk    = (args.kmin_cut<np.loadtxt(datadir / f"{tracer_tag}_{zsnap}" /f"Power_Spectrum_{fn_mid}_{i+1}.txt",usecols=0))
            Pk_all.append(np.array([np.loadtxt(datadir / f"{tracer_tag}_{zsnap}" /f"Power_Spectrum_{fn_mid}_{i+1}.txt",usecols=2+k)[sel_pk] for k in range(len(present_ells))]).flatten())
        data_vec= np.mean(Pk_all,axis=0)
        cov_mat = np.cov(np.array(Pk_all).T)

        if not set(ells).issubset(set(present_ells)):
            raise RuntimeError(f"[{file_tag}] requested ells {ells} not available; present={present_ells}")

        dk = float(np.median(np.diff(k_out))) 
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
            cosmo.init.params['tau_reio'].update(fixed=True, value=0.0544)
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
                    cosmo.init.params["exp_s"].update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (0.0, 5.0)},
                        ref={"dist": "norm", "loc": 2.0, "scale": 0.3},
                        delta=0.5,
                    )

            elif args.mg_variant == "BZ_fR":
                # f(R)-like subset of BZ:
                #   beta_1 = 4/3, exp_s = 4, only lambda_1 is free
                for nm, val in [("beta_1", 4.0/3.0), ("lambda_1", 100.0), ("exp_s", 4.0)]:
                    cosmo.init.params.data.append(
                        parameter.Parameter(basename=nm, value=val, fixed=True)
                    )

                if args.force_GR:
                    # GR limit: effectively B0 -> 0, so lambda_1 -> 0 and no MG
                    cosmo.init.params["beta_1"].update(fixed=True, value=1.0)
                    cosmo.init.params["lambda_1"].update(fixed=True, value=0.0)
                    cosmo.init.params["exp_s"].update(fixed=True, value=4.0)
                else:
                    # Only lambda_1 varies; choose prior so that your desired B0-range is covered.
                    cosmo.init.params["lambda_1"].update(
                        fixed=False,
                        prior={"dist": "uniform", "limits": (0.0, 1e6)},  # tweak as you like
                        ref={"dist": "norm", "loc": 30.0, "scale": 10.0},
                        delta=10.0,
                    )
                    # beta_1 and exp_s remain fixed to 4/3 and 4 respectively

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
        
        # Small tweak for BZ_fR
        fkpt_mg_variant = args.mg_variant
        if fkpt_mg_variant == "BZ_fR":
            fkpt_mg_variant = "BZ"
        theory.init.update(freedom=args.freedom, prior_basis=args.priors_basis,
                           tracer=tracer_tag, template=template,
                           k=k_out, ells=list(ells), b3_coev=True,
                           model=args.MG_model, mg_variant=fkpt_mg_variant,
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

        # Base (un-suffixed) names for the alpha nuisance parameters
        alpha_basenames = ["alpha0", "alpha2", "alpha4", "alpha0shot", "alpha2shot"]
        alpha_fullnames = [pname(b) for b in alpha_basenames]  # e.g. alpha0p, alpha2p, ...

        # Fiducial (mean) values you want to center the priors on
        alpha_fid = {
            "alpha0":     3.0,
            "alpha2":    -1.0,
            "alpha4":     0.0,
            "alpha0shot": 0.08,
            "alpha2shot": -2.0,
        }

        # Prior widths: depend on prior basis
        if args.priors_basis == "physical":
            # tighter "physical" priors
            width_EFT = 30.0
            width_SN0 = 2.0
            width_SN2 = 5.0
        else:
            # "standard" priors (your previous defaults)
            width_EFT = 125.0
            width_SN0 = 20.0
            width_SN2 = 50.0

        # EFT counterterms: alpha0, alpha2, alpha4
        for base in ["alpha0", "alpha2", "alpha4"]:
            full = pname(base)  # e.g. "alpha0" or "alpha0p"
            if full in theory.params:
                p = theory.params[full]
                if p.varied:
                    p.update(
                        prior={
                            "dist": "norm",
                            "loc": alpha_fid[base],   # center on your fiducial value
                            "scale": width_EFT,
                        }
                    )

        # Shot-noise-like alphas: alpha0shot, alpha2shot
        for base, width in [("alpha0shot", width_SN0), ("alpha2shot", width_SN2)]:
            full = pname(base)
            if full in theory.params:
                p = theory.params[full]
                if p.varied:
                    p.update(
                        prior={
                            "dist": "norm",
                            "loc": alpha_fid[base],   # center on fiducial
                            "scale": width,
                        }
                    )

        # Finally: mark all these alpha-parameters as analytically marginalized
        for p in theory.params.select(basename=alpha_fullnames):
            if p.varied:
                p.update(derived=".marg")
                # If you wanted numeric marg instead, you'd keep them "normal" varied params.
    
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
        sys.exit()

    likelihood = sum(likelihoods)

    outdir = args.chains_dir
    save_pattern = str(outdir / f"{prefix}_*.npy")  # desilike expands * to 0..N-1

    # ---------- Run mode ----------
    if args.mode == "mcmc":
        if args.resume:
            all_files = sorted(glob(save_pattern))
            # Ignore MAP profile files like "<prefix>_profiles.npy"
            existing = [f for f in all_files if "profiles" not in Path(f).stem]

            if MPI.COMM_WORLD.Get_rank() == 0:
                if not existing:
                    print(f"[resume/mcmc] No existing chain files found for {prefix}; starting fresh 4 chains.")
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

