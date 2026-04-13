#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from mpi4py import MPI
from scipy.interpolate import interp1d


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Small EFTCAMB RPH Horndeski + fkptjax/desilike smoke runner. "
            "This is for validating the current Horndeski implementation, not "
            "a production alpha-parameter MCMC."
        )
    )
    parser.add_argument("--mode", choices=["eval", "mcmc", "map"], default="eval")
    parser.add_argument("--base-dir", type=Path, default=None, help="HEFTCAMB_DESI checkout directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("/n/home12/cgarciaquintero/DESI/MG_validation/synthetic_noiseless/data_vectors/"))
    parser.add_argument("--chains-dir", type=Path, default=Path("./chains_horndeski_smoke"))
    parser.add_argument("--chain-prefix", type=str, default="chain_horndeski_smoke")
    parser.add_argument("--fid-model", type=str, default="LCDM", help="Synthetic data filename tag.")
    parser.add_argument("--tracer", choices=["BGS", "LRG1", "LRG2", "LRG3", "ELG", "QSO"], default="BGS")
    parser.add_argument("--ells", type=str, default="0,2")
    parser.add_argument("--kmin-cut", type=float, default=0.02)
    parser.add_argument("--kmax-cut", type=float, default=0.20)
    parser.add_argument("--cov-scale", type=float, default=1.0)
    parser.add_argument("--use-cov-x10", action="store_true")
    parser.add_argument("--synthetic-from-theory", action="store_true", help="Ignore files and use the fiducial theory as data with diagonal covariance.")
    parser.add_argument("--require-data-files", action="store_true", help="Raise an error if the requested synthetic data files are missing instead of falling back to --synthetic-from-theory.")
    parser.add_argument("--diag-cov-frac", type=float, default=0.02, help="Fractional diagonal covariance for --synthetic-from-theory.")

    # Cosmology is fixed in this smoke runner. Varying it would require rebuilding
    # the external EFTCAMB P_lin consistently inside the likelihood.
    parser.add_argument("--h", type=float, default=0.6711)
    parser.add_argument("--ombh2", type=float, default=0.022)
    parser.add_argument("--omch2", type=float, default=0.122)
    parser.add_argument("--As", type=float, default=2.0e-9)
    parser.add_argument("--ns", type=float, default=0.965)
    parser.add_argument("--Neff", type=float, default=3.046)
    parser.add_argument("--mnu", type=float, default=0.0)

    # RPH alpha_X(a) = c_X Omega_DE(a) parameters.
    parser.add_argument("--alphaM0", type=float, default=0.1)
    parser.add_argument("--alphaB0", type=float, default=0.0)
    parser.add_argument("--alphaK0", type=float, default=0.5)
    parser.add_argument("--direct-eftcamb", action="store_true", help="Recompute EFTCAMB inside the likelihood, allowing sampled cosmology / alpha parameters. Slow but physically consistent.")
    parser.add_argument("--vary-cosmology", action="store_true", help="With --direct-eftcamb, vary h, omega_b, omega_cdm, logA, and n_s.")
    parser.add_argument("--vary-alphaM", action="store_true", help="With --direct-eftcamb, vary RPH alpha_M0 with prior [0, 3].")
    parser.add_argument("--vary-alphaB", action="store_true", help="With --direct-eftcamb, vary RPH alpha_B0 with prior [-3, 0].")

    parser.add_argument("--freedom", choices=["max", "min"], default="max")
    parser.add_argument("--prior-basis", choices=["standard", "physical", "physical_velocileptors", "APscaling"], default="standard")
    parser.add_argument("--vary-nuisance", choices=["none", "b1", "b1b2", "all"], default="all")

    parser.add_argument("--nchains", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--min-iter", type=int, default=0)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--max-eigen-gr", type=float, default=0.2)
    parser.add_argument("--stable-over", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ref-scale", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=200)
    return parser.parse_args()


def parse_ells(value: str) -> tuple[int, ...]:
    ells = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ells or not set(ells).issubset({0, 2, 4}):
        raise ValueError(f"--ells must be a comma-separated subset of 0,2,4; got {value!r}")
    return tuple(sorted(ells))


def setup_paths(base_dir: Path):
    paths = [
        base_dir / "EFTCAMB",
        base_dir / "desilike_fkptdev",
        base_dir / "fkptjax_muMG" / "src",
        base_dir / "FolpsD",
    ]
    for path in reversed(paths):
        text = os.path.realpath(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def clear_local_imports():
    """Force imports below to resolve against the local HEFTCAMB_DESI checkout."""
    prefixes = ("camb", "desilike", "fkptjax", "cosmoprimo")
    for name in list(sys.modules):
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def tracer_info(tag: str):
    table = {
        "BGS": ("BGS", "BGS", 1.5, 0.295, -0.52),
        "LRG1": ("LRG1", "LRG1", 2.0, 0.510, -0.42),
        "LRG2": ("LRG2", "LRG2", 2.1, 0.706, -0.36),
        "LRG3": ("LRG3", "LRG3", 2.2, 0.934, -0.30),
        "ELG": ("ELG", "ELG", 1.3, 1.321, -0.62),
        "QSO": ("QSO", "QSO", 2.5, 1.484, -0.25),
    }
    return table[tag]


def select_data_and_cov(Pvec, cov, present_ells, requested_ells, start, ncut):
    Pvec = np.asarray(Pvec).reshape(-1)
    cov = np.asarray(cov)
    nraw = Pvec.size // len(present_ells)
    order_map = {ell: i for i, ell in enumerate(present_ells)}
    idx = []
    for ell in requested_ells:
        base = order_map[ell] * nraw
        idx.extend(range(base + start, base + start + ncut))
    idx = np.asarray(idx, dtype=int)
    return Pvec[idx], cov[np.ix_(idx, idx)]


def load_synthetic_data(args, file_tag, ells):
    k_path = args.data_dir / f"{file_tag}_{args.fid_model}_k.txt"
    p_path = args.data_dir / f"{file_tag}_{args.fid_model}_P0P2P4.txt"
    cov_suffix = "_cov_x10.txt" if args.use_cov_x10 else "_cov.txt"
    c_path = args.data_dir / f"{file_tag}_{args.fid_model}{cov_suffix}"
    if not (k_path.exists() and p_path.exists() and c_path.exists()):
        raise FileNotFoundError(
            "Missing synthetic input files. Either pass --synthetic-from-theory for a pure smoke test, "
            "or provide --data-dir / --fid-model.\n"
            f"k:   {k_path}\nP:   {p_path}\ncov: {c_path}"
        )
    k_all = np.loadtxt(k_path)
    P_all = np.loadtxt(p_path)
    cov_all = np.loadtxt(c_path)
    if args.cov_scale != 1.0:
        cov_all = float(args.cov_scale) * cov_all
    start = int(np.searchsorted(k_all, float(args.kmin_cut), side="left"))
    stop = int(np.searchsorted(k_all, float(args.kmax_cut), side="right"))
    ncut = stop - start
    if ncut <= 0:
        raise RuntimeError(f"k cuts [{args.kmin_cut}, {args.kmax_cut}] removed all bins.")
    k_data = k_all[start:stop]
    data, cov = select_data_and_cov(P_all, cov_all, (0, 2, 4), ells, start, ncut)
    return k_data, data, cov


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    os.environ.setdefault("FOLPS_BACKEND", "jax")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if args.base_dir is None:
        # This file lives in BASE/desilike_fkptdev/MG/running_scripts.
        args.base_dir = Path(__file__).resolve().parents[3]
    args.base_dir = args.base_dir.resolve()
    setup_paths(args.base_dir)
    clear_local_imports()

    import camb
    from camb import model as cambmodel
    from camb.baseconfig import CAMBError
    import fkptjax
    import desilike
    from desilike import setup_logging
    from desilike.theories import Cosmoprimo
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, fkptjaxTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering.power_template import ExternalLinearPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.observables import ObservableCovariance
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.samplers import MCMCSampler
    from desilike.profilers import MinuitProfiler
    from desilike.base import BaseCalculator
    from desilike.parameter import Parameter
    from cosmoprimo.fiducial import DESI

    setup_logging("info")
    if rank == 0:
        print("base_dir:", args.base_dir)
        print("camb:", getattr(camb, "__file__", None))
        print("desilike:", desilike.__file__)
        print("fkptjax:", fkptjax.__file__)

    if not os.path.realpath(getattr(camb, "__file__", "")).startswith(str(args.base_dir / "EFTCAMB")):
        raise RuntimeError("Wrong CAMB imported; expected the EFTCAMB/camb package under --base-dir.")
    if not os.path.realpath(desilike.__file__).startswith(str(args.base_dir / "desilike_fkptdev")):
        raise RuntimeError("Wrong desilike imported; expected --base-dir/desilike_fkptdev.")
    if not os.path.realpath(fkptjax.__file__).startswith(str(args.base_dir / "fkptjax_muMG" / "src")):
        raise RuntimeError("Wrong fkptjax imported; expected --base-dir/fkptjax_muMG/src.")

    ells = parse_ells(args.ells)
    file_tag, tracer_tag, b1_fid, z_eff, b2_ref = tracer_info(args.tracer)

    def make_cosmo(h=None, ombh2=None, omch2=None, logA=None, ns=None):
        h = float(args.h if h is None else h)
        ombh2 = float(args.ombh2 if ombh2 is None else ombh2)
        omch2 = float(args.omch2 if omch2 is None else omch2)
        logA = float(np.log(1e10 * args.As) if logA is None else logA)
        ns = float(args.ns if ns is None else ns)
        cosmo = Cosmoprimo(engine="camb", N_eff=args.Neff, m_ncdm=[])
        # Fixed background: external EFTCAMB P_lin is precomputed at these values.
        cosmo.init.params["h"].update(value=h, fixed=True)
        cosmo.init.params["omega_b"].update(value=ombh2, fixed=True)
        cosmo.init.params["omega_cdm"].update(value=omch2, fixed=True)
        cosmo.init.params["logA"].update(value=logA, fixed=True)
        cosmo.init.params["n_s"].update(value=ns, fixed=True)
        if "tau_reio" in cosmo.init.params:
            cosmo.init.params["tau_reio"].update(fixed=True)
        return cosmo

    def run_eftcamb(eft_params, h=None, ombh2=None, omch2=None, logA=None, ns=None):
        h = float(args.h if h is None else h)
        ombh2 = float(args.ombh2 if ombh2 is None else ombh2)
        omch2 = float(args.omch2 if omch2 is None else omch2)
        logA = float(np.log(1e10 * args.As) if logA is None else logA)
        ns = float(args.ns if ns is None else ns)
        As = float(np.exp(logA) / 1e10)
        eft_params = dict(eft_params)
        if eft_params.get("EFTflag", 0) == 0:
            eft_params = {"EFTflag": 0}
        pars = camb.set_params(
            H0=h * 100.0,
            ombh2=ombh2,
            omch2=omch2,
            As=As,
            ns=ns,
            nnu=float(args.Neff),
            num_massive_neutrinos=0,
            mnu=float(args.mnu),
            **eft_params,
        )
        pars.set_matter_power(redshifts=[float(z_eff), 0.0], kmax=max(1.0, float(args.kmax_cut) * 3.0))
        pars.NonLinear = cambmodel.NonLinear_none
        return camb.get_results(pars), pars

    stability = dict(
        feedback_level=0,
        EFT_ghost_math_stability=False,
        EFT_mass_math_stability=False,
        EFT_ghost_stability=True,
        EFT_gradient_stability=True,
        EFT_mass_stability=False,
        EFT_additional_priors=False,
    )
    rph_params = dict(
        EFTflag=2,
        AltParEFTmodel=1,
        RPHusealphaM=True,
        RPH_M0=0.0,
        RPHintegratefromtoday=False,
        RPHalphaMmodel=0,
        RPHalphaMmodel_ODE=2,
        RPHalphaM_ODE0=float(args.alphaM0),
        RPHkineticitymodel=0,
        RPHkineticitymodel_ODE=2,
        RPHkineticity_ODE0=float(args.alphaK0),
        RPHbraidingmodel=0,
        RPHbraidingmodel_ODE=2,
        RPHbraiding_ODE0=float(args.alphaB0),
        RPHtensormodel=0,
        RPHtensormodel_ODE=0,
        **stability,
    )

    if rank == 0:
        print("Running EFTCAMB RPH and GR for fixed smoke-test alpha parameters:")
        print({k: rph_params[k] for k in rph_params if k.startswith("RPH") or k in ("EFTflag", "AltParEFTmodel")})
        if args.direct_eftcamb:
            print(
                "NOTE: --direct-eftcamb is enabled. Sampled cosmology / alphaB values will recompute "
                "EFTCAMB P_lin and h1/h3/h5 inside the likelihood. This is slow but consistent."
            )
        else:
            print(
                "NOTE: this non-direct runner samples nuisance parameters only. "
                "Cosmology and Horndeski alpha parameters are fixed because EFTCAMB P_lin and h1/h3/h5 "
                "are precomputed once before the likelihood is built."
            )
    res_rph, pars_rph = run_eftcamb(rph_params)
    res_gr, pars_gr = run_eftcamb({"EFTflag": 0})

    kh_rph, z_arr, pk_rph_all = res_rph.get_matter_power_spectrum(minkh=1e-4, maxkh=1.0, npoints=500, params=pars_rph)
    kh_gr, _, pk_gr_all = res_gr.get_matter_power_spectrum(minkh=1e-4, maxkh=1.0, npoints=500, params=pars_gr)
    z_arr = np.asarray(z_arr, dtype=float)
    iz = int(np.argmin(np.abs(z_arr - float(z_eff))))
    Plin_rph = np.asarray(pk_rph_all[iz], dtype=float)
    Plin_gr = np.asarray(pk_gr_all[iz], dtype=float)
    Plin_gr_on_rph = np.interp(kh_rph, np.asarray(kh_gr, dtype=float), Plin_gr)

    a_grid = np.logspace(-2, 0, 150)
    eta_grid = np.log(a_grid)
    fields, cache_arr = pars_rph.EFTCAMB.get_eft_functions(res_rph, a_grid)
    for name in ("h1_loop", "h3_loop", "h5_loop"):
        if name not in fields:
            raise RuntimeError(f"EFTCAMB output does not contain {name}.")
    h1_arr = np.asarray(cache_arr["h1_loop"], dtype=float)
    h3_arr = np.asarray(cache_arr["h3_loop"], dtype=float)
    h5_arr = np.asarray(cache_arr["h5_loop"], dtype=float)
    h1_interp = interp1d(eta_grid, h1_arr, kind="linear", bounds_error=False, fill_value=(h1_arr[0], h1_arr[-1]))
    h3_interp = interp1d(eta_grid, h3_arr, kind="linear", bounds_error=False, fill_value=(h3_arr[0], h3_arr[-1]))
    h5_interp = interp1d(eta_grid, h5_arr, kind="linear", bounds_error=False, fill_value=(h5_arr[0], h5_arr[-1]))
    gr_h1 = interp1d(eta_grid, np.ones_like(eta_grid), kind="linear", bounds_error=False, fill_value=1.0)
    gr_h3 = interp1d(eta_grid, np.zeros_like(eta_grid), kind="linear", bounds_error=False, fill_value=0.0)
    gr_h5 = interp1d(eta_grid, np.zeros_like(eta_grid), kind="linear", bounds_error=False, fill_value=0.0)

    a_out = 1.0 / (1.0 + float(z_eff))
    eta_out = float(np.log(a_out))
    h1_val, h3_val, h5_val = float(h1_interp(eta_out)), float(h3_interp(eta_out)), float(h5_interp(eta_out))
    y_test = h1_val * (1.0 + 0.01 * h5_val) / (1.0 + 0.01 * h3_val)
    if rank == 0:
        print(f"z={z_eff:.3f}: h1={h1_val:.8g}, h3={h3_val:.8g}, h5={h5_val:.8g}, Y(k=0.1)={y_test:.8g}")
        print(f"EFTCAMB Plin ratio at k=0.1: {100.0 * (np.interp(0.1, kh_rph, Plin_rph) / np.interp(0.1, kh_rph, Plin_gr_on_rph) - 1.0):+.5f}%")

    if args.synthetic_from_theory:
        k_data = np.linspace(float(args.kmin_cut), float(args.kmax_cut), 20)
        data_vec = None
        cov_mat = None
    else:
        try:
            k_data, data_vec, cov_mat = load_synthetic_data(args, file_tag, ells)
        except FileNotFoundError:
            if args.require_data_files:
                raise
            if rank == 0:
                print("Synthetic data files were not found; falling back to --synthetic-from-theory smoke data.")
            k_data = np.linspace(float(args.kmin_cut), float(args.kmax_cut), 20)
            data_vec = None
            cov_mat = None

    varied_nuisance = {
        "none": set(),
        "b1": {"b1"},
        "b1b2": {"b1", "b2"},
        "all": None,
    }[args.vary_nuisance]
    fid_values = dict(
        b1=b1_fid,
        b2=b2_ref,
        bs2=0.0,
        b3nl=0.0,
        alpha0=3.0,
        alpha2=-29.0,
        alpha4=0.0,
        ctilde=0.0,
        alpha0shot=0.08,
        alpha2shot=-8.0,
    )

    def build_theory(h1_itp, h3_itp, h5_itp, input_pk, label):
        cosmo = make_cosmo()
        template = ExternalLinearPowerSpectrumTemplate(
            pklin_k=np.asarray(kh_rph, dtype=float),
            pklin_pk=np.asarray(input_pk, dtype=float),
            z=float(z_eff),
            fiducial=DESI(),
            cosmo=cosmo,
        )
        template.init.update(with_now="peakaverage")
        theory = fkptjaxTracerPowerSpectrumMultipoles()
        theory.init.update(
            freedom=args.freedom,
            prior_basis=args.prior_basis,
            tracer=tracer_tag,
            template=template,
            k=np.asarray(k_data, dtype=float),
            ells=list(ells),
            model="EFTCAMB_HORNDESKI",
            beyond_eds=True,
            b3_coev=True,
            eftcamb_h1_interp=h1_itp,
            eftcamb_h3_interp=h3_itp,
            eftcamb_h5_interp=h5_itp,
            rescale_PS=False,
            shotnoise=1e4,
        )
        for name, value in fid_values.items():
            if name in theory.init.params:
                fixed = False if varied_nuisance is None else name not in varied_nuisance
                update = dict(value=float(value), fixed=bool(fixed))
                if not fixed:
                    scale = 0.05 if name == "b1" else 0.1
                    update["ref"] = {"dist": "norm", "loc": float(value), "scale": scale}
                theory.init.params[name].update(**update)
        if rank == 0:
            print(f"Built {label}: fixed alpha, vary_nuisance={args.vary_nuisance}")
        return theory

    class DirectEFTCAMBHorndeskiPowerSpectrumMultipoles(BaseCalculator):
        """Slow direct model: recompute EFTCAMB and FKPT tables for sampled parameters."""

        def initialize(self, k=None, ells=None):
            self.k = np.asarray(k, dtype=float)
            self.ells = tuple(ells)
            self.inner_theory = None
            self.last_eftcamb_error = ""

            cosmo_specs = {
                "h": dict(value=float(args.h), fixed=not args.vary_cosmology,
                          prior={"dist": "uniform", "limits": (0.4, 1.0)},
                          ref={"dist": "norm", "loc": float(args.h), "scale": 0.01}, delta=0.01),
                "omega_b": dict(value=float(args.ombh2), fixed=not args.vary_cosmology,
                                prior={"dist": "norm", "loc": 0.02237, "scale": 0.00055},
                                ref={"dist": "norm", "loc": float(args.ombh2), "scale": 0.0002}, delta=0.0005),
                "omega_cdm": dict(value=float(args.omch2), fixed=not args.vary_cosmology,
                                  prior={"dist": "uniform", "limits": (0.001, 0.99)},
                                  ref={"dist": "norm", "loc": float(args.omch2), "scale": 0.005}, delta=0.005),
                "logA": dict(value=float(np.log(1e10 * args.As)), fixed=not args.vary_cosmology,
                             prior={"dist": "uniform", "limits": (1.61, 3.91)},
                             ref={"dist": "norm", "loc": float(np.log(1e10 * args.As)), "scale": 0.03}, delta=0.03),
                "n_s": dict(value=float(args.ns), fixed=not args.vary_cosmology,
                            prior={"dist": "norm", "loc": 0.9649, "scale": 0.02},
                            ref={"dist": "norm", "loc": float(args.ns), "scale": 0.004}, delta=0.01),
            }
            for name, spec in cosmo_specs.items():
                if spec.pop("fixed"):
                    self.params.set(Parameter(name, value=spec["value"], fixed=True))
                else:
                    self.params.set(Parameter(name, fixed=False, **spec))

            alpha_b_ref = min(max(float(args.alphaB0), -2.95), -0.05)
            alpha_m_ref = min(max(float(args.alphaM0), 0.05), 2.95)
            alpha_b_spec = dict(
                value=float(args.alphaB0),
                fixed=not args.vary_alphaB,
                prior={"dist": "uniform", "limits": (-3.0, 0.0)},
                ref={"dist": "norm", "loc": alpha_b_ref, "scale": 0.05},
                delta=0.05,
            )
            alpha_m_spec = dict(
                value=float(args.alphaM0),
                fixed=not args.vary_alphaM,
                prior={"dist": "uniform", "limits": (0.0, 3.0)},
                ref={"dist": "norm", "loc": alpha_m_ref, "scale": 0.05},
                delta=0.05,
            )
            if alpha_m_spec.pop("fixed"):
                self.params.set(Parameter("alphaM0", value=float(args.alphaM0), fixed=True))
            else:
                self.params.set(Parameter("alphaM0", fixed=False, **alpha_m_spec))
            if alpha_b_spec.pop("fixed"):
                self.params.set(Parameter("alphaB0", value=float(args.alphaB0), fixed=True))
            else:
                self.params.set(Parameter("alphaB0", fixed=False, **alpha_b_spec))

            for name, value in fid_values.items():
                fixed = False if varied_nuisance is None else name not in varied_nuisance
                if fixed:
                    self.params.set(Parameter(name, value=float(value), fixed=True))
                else:
                    scale = 0.05 if name == "b1" else 0.1
                    self.params.set(Parameter(
                        name, value=float(value), fixed=False,
                        ref={"dist": "norm", "loc": float(value), "scale": scale},
                        delta=scale,
                    ))

        def _rph_params(self, alphaM0, alphaB0):
            return dict(
                EFTflag=2,
                AltParEFTmodel=1,
                RPHusealphaM=True,
                RPH_M0=0.0,
                RPHintegratefromtoday=False,
                RPHalphaMmodel=0,
                RPHalphaMmodel_ODE=2,
                RPHalphaM_ODE0=float(alphaM0),
                RPHkineticitymodel=0,
                RPHkineticitymodel_ODE=2,
                RPHkineticity_ODE0=float(args.alphaK0),
                RPHbraidingmodel=0,
                RPHbraidingmodel_ODE=2,
                RPHbraiding_ODE0=float(alphaB0),
                RPHtensormodel=0,
                RPHtensormodel_ODE=0,
                **stability,
            )

        def calculate(self, **params):
            h = float(params.get("h", args.h))
            ombh2 = float(params.get("omega_b", args.ombh2))
            omch2 = float(params.get("omega_cdm", args.omch2))
            logA = float(params.get("logA", np.log(1e10 * args.As)))
            ns = float(params.get("n_s", args.ns))
            alphaM0 = float(params.get("alphaM0", args.alphaM0))
            alphaB0 = float(params.get("alphaB0", args.alphaB0))

            try:
                res, pars = run_eftcamb(
                    self._rph_params(alphaM0, alphaB0),
                    h=h, ombh2=ombh2, omch2=omch2, logA=logA, ns=ns,
                )
            except CAMBError as exc:
                # EFTCAMB rejects unstable Horndeski points in the sampled volume.
                # Return a finite but terrible model so the sampler can reject the
                # point instead of aborting the MPI job.
                self.inner_theory = None
                self.last_eftcamb_error = str(exc)
                self.power = np.full((len(self.ells), self.k.size), 1e30, dtype=float)
                return
            kh, z_tmp, pk_all = res.get_matter_power_spectrum(
                minkh=1e-4, maxkh=1.0, npoints=500, params=pars,
            )
            z_tmp = np.asarray(z_tmp, dtype=float)
            iz_tmp = int(np.argmin(np.abs(z_tmp - float(z_eff))))
            pklin = np.asarray(pk_all[iz_tmp], dtype=float)

            fields_tmp, cache_tmp = pars.EFTCAMB.get_eft_functions(res, a_grid)
            for field in ("h1_loop", "h3_loop", "h5_loop"):
                if field not in fields_tmp:
                    raise RuntimeError(f"EFTCAMB output does not contain {field}.")
            h1_itp = interp1d(eta_grid, np.asarray(cache_tmp["h1_loop"], dtype=float), kind="linear",
                              bounds_error=False, fill_value="extrapolate")
            h3_itp = interp1d(eta_grid, np.asarray(cache_tmp["h3_loop"], dtype=float), kind="linear",
                              bounds_error=False, fill_value="extrapolate")
            h5_itp = interp1d(eta_grid, np.asarray(cache_tmp["h5_loop"], dtype=float), kind="linear",
                              bounds_error=False, fill_value="extrapolate")

            cosmo = make_cosmo(h=h, ombh2=ombh2, omch2=omch2, logA=logA, ns=ns)
            template = ExternalLinearPowerSpectrumTemplate(
                pklin_k=np.asarray(kh, dtype=float),
                pklin_pk=pklin,
                z=float(z_eff),
                fiducial=DESI(),
                cosmo=cosmo,
            )
            template.init.update(with_now="peakaverage")

            theory = fkptjaxTracerPowerSpectrumMultipoles()
            theory.init.update(
                freedom=args.freedom,
                prior_basis=args.prior_basis,
                tracer=tracer_tag,
                template=template,
                k=self.k,
                ells=list(self.ells),
                model="EFTCAMB_HORNDESKI",
                beyond_eds=True,
                b3_coev=True,
                eftcamb_h1_interp=h1_itp,
                eftcamb_h3_interp=h3_itp,
                eftcamb_h5_interp=h5_itp,
                rescale_PS=False,
                shotnoise=1e4,
            )
            for name, default in fid_values.items():
                if name in theory.init.params:
                    theory.init.params[name].update(fixed=True, value=float(params.get(name, default)))

            self.inner_theory = theory
            self.power = np.asarray(theory())

        def get(self):
            return self.power

    if args.direct_eftcamb:
        theory_rph = DirectEFTCAMBHorndeskiPowerSpectrumMultipoles()
        theory_rph.init.update(k=np.asarray(k_data, dtype=float), ells=list(ells))
        if rank == 0:
            print(
                "Built direct EFTCAMB Horndeski theory: "
                f"vary_cosmology={args.vary_cosmology}, vary_alphaM={args.vary_alphaM}, vary_alphaB={args.vary_alphaB}, "
                f"vary_nuisance={args.vary_nuisance}. This recomputes EFTCAMB inside each likelihood call."
            )
    else:
        theory_rph = build_theory(h1_interp, h3_interp, h5_interp, Plin_rph, "RPH theory")

    if data_vec is None:
        if rank == 0:
            print("Building synthetic data from the RPH fiducial theory.")
        fid = np.asarray(theory_rph()).reshape(-1)
        data_vec = fid
        sigma = np.maximum(args.diag_cov_frac * np.abs(fid), 1.0)
        cov_mat = np.diag(sigma**2)

    observable = TracerPowerSpectrumMultipolesObservable(
        data=data_vec,
        theory=theory_rph,
        k=[k_data for _ in ells],
        ells=list(ells),
    )
    covmeta = [{"name": "PowerSpectrumMultipoles", "x": [k_data] * len(ells), "projs": list(ells)}]
    covariance = ObservableCovariance(cov_mat, observables=covmeta)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance, name=file_tag.lower())

    args.chains_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{args.chain_prefix}_{file_tag}_aM{args.alphaM0:g}_aB{args.alphaB0:g}_aK{args.alphaK0:g}"
        f"_k{args.kmin_cut:g}-{args.kmax_cut:g}_l{''.join(map(str, ells))}"
    )
    save_pattern = str(args.chains_dir / f"{prefix}_*.npy")

    if args.mode == "eval":
        value = likelihood()
        if rank == 0:
            print("One likelihood evaluation completed.")
            print("logposterior/loglikelihood value:", value)
        return

    if args.mode == "mcmc":
        existing = sorted(glob(save_pattern))
        chains = existing if (args.resume and existing) else int(args.nchains)
        sampler = MCMCSampler(
            likelihood,
            chains=chains,
            seed=42,
            save_fn=save_pattern,
            mpicomm=comm,
            ref_scale=float(args.ref_scale),
        )
        sampler.run(
            check={"max_eigen_gr": float(args.max_eigen_gr), "stable_over": int(args.stable_over)},
            check_every=int(args.check_every),
            min_iterations=int(args.min_iter),
            max_iterations=int(args.max_iter),
        )
        return

    profiles_out = args.chains_dir / f"{prefix}_profiles.npy"
    profiler = MinuitProfiler(likelihood, save_fn=str(profiles_out), ref_scale=float(args.ref_scale))
    profiler.maximize(max_iterations=int(args.max_calls))
    if rank == 0:
        print(f"Profiles saved to {profiles_out}")


if __name__ == "__main__":
    main()
