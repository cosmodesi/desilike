import os,sys
from run_desilike_synthetic_DV import *

if __name__ == "__main__":
    args = parse_args()
    setup_logging("info")

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

    # Tracers: (file tag, tracer tag, b1 prior center, z_eff)
    ################################# 
    tracer_table = [
        ("LRG_EZmean_EZcov", "LRG",  2.1, 0.8, -1.10, 0.538705, 'z08', 'EZmocks_LRG2ndGen_cubic'),#, 'abacusHF_cutsky_LRG0p725'), 
        ("ELG_EZmean_EZcov", "ELG",  1.2, 0.95, 0.03, 0.503071, 'z095','EZELG_boxz095_bin01',   ),#  'abacusHF_cutsky_ELG0p950'), 
        ("QSO_EZmean_EZcov", "QSO",  2.1, 1.4, -0.71, 0.410860, 'z14', 'EZQSO_boxz1400_bin01',  ),#  'abacusHF_cutsky_QSO1p400'),
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
    prefix += f"_data-{args.fid_model}"
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

    for file_tag, tracer_tag, b1_fid, z_eff, b2_ref, sigma8_fid_val, zsnap, mid_ez in tracer_table:
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
        for i in range(1000): # LRG mock 933 had two fewer rows
            sel_pk    = (args.kmin_cut<np.loadtxt(datadir / f"EZmocks/Pk/{tracer_tag}_{zsnap}" /f"Power_Spectrum_{mid_ez}_{i+1}.txt",usecols=0))
            Pk_all.append(np.array([np.loadtxt(datadir / f"EZmocks/Pk/{tracer_tag}_{zsnap}" /f"Power_Spectrum_{mid_ez}_{i+1}.txt",usecols=2+k)[sel_pk] for k in range(len(present_ells))]).flatten())
        cov_mat = np.cov(np.array(Pk_all).T)
        #################### get data vector with Abacus: k is different from that of EZmock, so use mean(EZmock) instead
        # we may read the shhotnoise from the file as well
        shotnoise= np.loadtxt(datadir / f"EZmocks/Pk/{tracer_tag}_{zsnap}" /f"Power_Spectrum_{mid_ez}_{i+1}.txt",usecols=6)[0]
        data_vec = np.mean(Pk_all,axis=0)

        if not set(ells).issubset(set(present_ells)):
            raise RuntimeError(f"[{file_tag}] requested ells {ells} not available; present={present_ells}")

        dk = float(np.median(np.diff(k_out))) 
        kmin_edge = float(k_out[0] - 0.5 * dk)
        kmax_edge = float(k_out[-1] + 0.5 * dk)
        klim = {L: (kmin_edge, kmax_edge, dk) for L in ells}

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
            kc=float(args.kc),
            scale_bins_method=str(args.scale_bins_method),
        )

        if args.create_emu and rank == 0:
            if emu_path.exists():
                print(f"[Emulator] ({file_tag}) exists → {emu_path.name}")
            else:
                print(f"[Emulator] ({file_tag}) fitting Taylor emulator (finite, order={args.emu_order})…")
                import time
                T0=time.time()
                import pdb;pdb.set_trace()
                _ = theory()
                print(f'1 realisation took {time.time()-T0:.1f}s')
                from matplotlib import pyplot as plt
                plt.plot(theory.k,theory.k*theory()[0],'b')
                plt.plot(k_out,k_out*data_vec[:len(k_out)],'b--')
                plt.plot(theory.k,theory.k*theory()[1],'r')
                plt.plot(k_out,k_out*data_vec[len(k_out):],'r--')
                plt.plot([],[],'k',label='Pk emulator')
                plt.plot([],[],'k--',label='Pk EZmock')
                plt.title('issue: No finite log posterior after 1000 tries')
                plt.legend()
                plt.savefig('emu_test_fkptjax')
                plt.close()
                sys.exit()
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
        #comm.Barrier()
        if rank == 0:
            print("[Emulator] Done. Exiting because --create-emu was set.")
        sys.exit()

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
