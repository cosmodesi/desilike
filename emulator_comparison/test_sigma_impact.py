"""
test_sigma_impact.py
--------------------
Quick check: how much do the binned Sigma (light-deflection) parameters affect
the LINEAR matter power spectrum P(k) computed with ISiTGR (binned mu/Sigma)?

Sigma enters the lensing/Weyl potential, NOT the growth of matter perturbations
(that is mu), so the expectation is ~no effect on the delta_cb linear P(k).
This plots P(k)/P_GR(k) for several Sigma variations at fixed mu, plus a few
mu variations for contrast.

Run (cosmodesi env + latest isitgr on PYTHONPATH):
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    export PYTHONPATH=/global/homes/p/prakharb/isitgr_private_latest:$PYTHONPATH
    python test_sigma_impact.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import isitgr

HERE = os.path.dirname(os.path.abspath(__file__))

# fiducial cosmology + ISiTGR binning constants (emulator training values)
Z = 0.5
FID = dict(H0=67.36, ombh2=0.02237, omch2=0.12, ln10As=3.044, ns=0.9649)
MNU, TAU, OMK = 0.06, 0.0568, 0.0
BIN = dict(z_div=1.0, z_TGR=2.0, z_tw=0.05, k_c=0.1, k_tw=0.001, k_TGR=0.01, k_S=0.2)

KH = np.logspace(-3, 0, 200)   # h/Mpc


def isitgr_plin(mu, Sigma):
    """Linear delta_cb P(k) [(Mpc/h)^3] on KH for binned (mu, Sigma)."""
    As = np.exp(FID["ln10As"]) / 1e10
    pars = isitgr.CAMBparams()
    pars.set_cosmology(
        H0=FID["H0"], ombh2=FID["ombh2"], omch2=FID["omch2"],
        mnu=MNU, omk=OMK, tau=TAU,
        MG_parameterization="muSigma", redshift_bins=True, scale_bins=True,
        mu1=mu[0], mu2=mu[1], mu3=mu[2], mu4=mu[3],
        Sigma1=Sigma[0], Sigma2=Sigma[1], Sigma3=Sigma[2], Sigma4=Sigma[3],
        **BIN,
    )
    pars.InitPower.set_params(As=As, ns=FID["ns"], r=0)
    pars.set_matter_power(redshifts=[Z], kmax=5.0)
    pars.NonLinear = isitgr.model.NonLinear_none
    res = isitgr.get_results(pars)
    k, _, pk = res.get_matter_power_spectrum(
        minkh=1e-3, maxkh=1.0, npoints=400, var1="delta_nonu", var2="delta_nonu")
    return np.interp(KH, k, pk[0])


GR = ([1.0] * 4, [1.0] * 4)

# Sigma variations at fixed mu=1 (GR growth) -> expect ratio ~ 1
SIGMA_CASES = [
    ("Sigma=1.5 (all bins)", [1.0]*4, [1.5, 1.5, 1.5, 1.5]),
    ("Sigma=0.5 (all bins)", [1.0]*4, [0.5, 0.5, 0.5, 0.5]),
    ("Sigma1=2.0 (low-z,low-k)", [1.0]*4, [2.0, 1.0, 1.0, 1.0]),
    ("Sigma mixed", [1.0]*4, [1.5, 0.8, 1.2, 1.0]),
]
# mu variations for contrast -> expect clear ratio != 1
MU_CASES = [
    ("mu=1.5 (all bins)", [1.5]*4, [1.0]*4),
    ("mu=0.5 (all bins)", [0.5]*4, [1.0]*4),
]


def main():
    print("Computing GR reference ...")
    p_gr = isitgr_plin(*GR)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    # Left: Sigma impact
    ax = axes[0]
    maxdev = 0.0
    for label, mu, Sig in SIGMA_CASES:
        p = isitgr_plin(mu, Sig)
        r = 100.0 * (p / p_gr - 1.0)
        ax.semilogx(KH, r, lw=1.6, label=label)
        maxdev = max(maxdev, float(np.max(np.abs(r))))
        print(f"  [Sigma] {label:28s} max|ΔP/P| = {np.max(np.abs(r)):.4f} %")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$P(k)/P_{\rm GR}(k) - 1\ [\%]$")
    ax.set_title(f"Impact of Sigma on linear P(k)  (max |dev| = {maxdev:.3f}%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: mu impact (contrast)
    ax = axes[1]
    for label, mu, Sig in MU_CASES:
        p = isitgr_plin(mu, Sig)
        r = 100.0 * (p / p_gr - 1.0)
        ax.semilogx(KH, r, lw=1.6, label=label)
        print(f"  [mu]    {label:28s} max|ΔP/P| = {np.max(np.abs(r)):.4f} %")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$P(k)/P_{\rm GR}(k) - 1\ [\%]$")
    ax.set_title("Impact of mu on linear P(k)  (for contrast)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"ISiTGR binned mu/Sigma: linear matter P(k) sensitivity (z={Z})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "test_sigma_impact.png")
    fig.savefig(out, dpi=150)
    print("\nSaved", out)


if __name__ == "__main__":
    main()
