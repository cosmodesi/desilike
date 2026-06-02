"""
test_matched_grid.py
--------------------
Follow-up to test_same_nw.py.  We showed the broadband original-vs-emulator
difference is the no-wiggle prescription; the only leftover was a spike in the
last 1-2 high-k bins (mostly P4), traced to the two pipelines feeding DIFFERENT
input k-grids to the fkpt loop (Kfuncs_to_tables):

   original (template):  geomspace(1e-3, 1.0, 500)
   emulator:             native emulator grid, 1e-4..10, 200 pts

Here the emulator provider is made to match the original pipeline fully:
  * resample the emulated plin onto the template grid geomspace(1e-3, 1, 500),
  * compute P_nw with the SAME peakaverage prescription on that grid.

Then emulator-class vs original-class should agree across ALL k, including the
high-k edge.  Two plots are produced: with P4 and without P4 (P0/P2 only).

Run with the same env / PYTHONPATH as compare_original_vs_emulator.py.
"""

import os
import sys
os.environ.setdefault("FOLPS_BACKEND", "jax")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_original_vs_emulator as C
from desilike.theories.galaxy_clustering import (
    MgEmulatorCosmology, fkpt_pkemu_TracerPowerSpectrumMultipoles)

ELLS = (0, 2, 4)
C.ELLS = ELLS

_FID_COSMO = [None]


def _fid_cosmo():
    if _FID_COSMO[0] is None:
        from cosmoprimo.fiducial import DESI
        _FID_COSMO[0] = DESI()
    return _FID_COSMO[0]


class MgEmulatorCosmologyMatched(MgEmulatorCosmology):
    """Emulator provider matched to the original template pipeline: emulated
    plin resampled onto the template k-grid, P_nw via the same peakaverage
    prescription on that grid."""

    # template grid for KOUT in (0.02, 0.2):  geomspace(_klim[0], _klim[1], _klim[2])
    TARGET_K = np.geomspace(1e-3, 1.0, 500)

    def calculate(self, logA, n_s, h, omega_b, omega_cdm,
                  mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4, **kwargs):
        from cosmoprimo import PowerSpectrumBAOFilter, PowerSpectrumInterpolator1D
        fid = _fid_cosmo()
        kt = self.TARGET_K
        H0 = h * 100.0
        Omega_m = (omega_b + omega_cdm + self.M_NCDM / 93.14) / h**2
        plist = [logA, n_s, H0, omega_b, omega_cdm, self.M_NCDM]
        self._results = {}
        for z in self.zs:
            k, plin, _pnw_emu, scalars = self._emu.predict_all(
                z=z, ln10As=logA, ns=n_s, H0=H0, ombh2=omega_b, omch2=omega_cdm,
                mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4,
                Sigma1=Sigma1, Sigma2=Sigma2, Sigma3=Sigma3, Sigma4=Sigma4)
            k = np.asarray(k); plin = np.asarray(plin)
            # resample plin onto the template grid (log-log interpolation)
            plin_t = np.exp(np.interp(np.log(kt), np.log(k), np.log(plin)))
            interp = PowerSpectrumInterpolator1D(kt, plin_t)
            filt = PowerSpectrumBAOFilter(interp, engine='peakaverage',
                                          cosmo=fid, cosmo_fid=fid)
            pnw_t = np.asarray(filt.smooth_pk_interpolator()(kt))
            chi_fid, e_fid = self._fid_scalars[z]
            qper = h * scalars['chi_z'] / (chi_fid * self.fiducial['h'])
            qpar = e_fid / scalars['e_z']
            self._results[z] = dict(
                k=kt, pk_dd=plin_t, pknow=pnw_t, Omega_m=Omega_m, h=h,
                sigma8=scalars['sigma8_z'], qper=qper, qpar=qpar,
                mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4, plist=plist)

    def get_at_z(self, z):
        return self._results[z]


def build_emulator_matched():
    prov = MgEmulatorCosmologyMatched(zs=[C.Z])
    for nm in C.MGNAMES:
        prov.init.params[nm].update(fixed=False)
    for nm, val in C.FID.items():
        prov.init.params[nm].update(value=val, fixed=True)
    theory = fkpt_pkemu_TracerPowerSpectrumMultipoles()
    theory.init.update(prior_basis="standard", tracer="LRG1", cosmo=prov,
                       z=C.Z, k=C.KOUT, ells=ELLS, beyond_eds=True)
    C._fix_bias(theory)
    return theory


def make_plot(results, ells_plot, fname, title):
    colors = {0: "C0", 2: "C1", 4: "C2"}
    ncase = len(results)
    fig, axes = plt.subplots(ncase, 2, figsize=(13, 3.4 * ncase))
    if ncase == 1:
        axes = axes[np.newaxis, :]
    summ = []
    for i, (label, PA, PB) in enumerate(results):
        axL, axR = axes[i, 0], axes[i, 1]
        worst = 0.0
        for ell in ells_plot:
            j = ELLS.index(ell)
            c = colors[ell]
            axL.plot(C.KOUT, C.KOUT * PA[j], color=c, lw=1.6, label=f"original l={ell}")
            axL.plot(C.KOUT, C.KOUT * PB[j], color=c, lw=1.4, ls="--", label=f"emulator l={ell}")
            diff = 100.0 * (PB[j] - PA[j]) / PA[j]
            axR.plot(C.KOUT, diff, color=c, lw=1.5, label=f"l={ell}")
            worst = max(worst, float(np.max(np.abs(diff))))
        axL.set_title(label, fontsize=10)
        axL.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axL.set_ylabel(r"$k\,P_\ell(k)$")
        axL.legend(fontsize=7, ncol=len(ells_plot)); axL.grid(alpha=0.3)
        axR.axhline(0, color="k", lw=0.8)
        for lvl in (1.0, -1.0):
            axR.axhline(lvl, color="gray", lw=0.7, ls=":")
        axR.set_title(f"emulator - original  (max |diff| = {worst:.2f}%)", fontsize=10)
        axR.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axR.set_ylabel("difference [%]")
        axR.legend(fontsize=8); axR.grid(alpha=0.3)
        summ.append((label, worst))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=150)
    print("Saved", out)
    return summ


def main():
    print("Building original ...")
    th_orig = C.build_original()
    print("Building emulator (matched grid + peakaverage nw) ...")
    th_emu = build_emulator_matched()

    results = []
    for label, mu, Sigma in C.CASES:
        print(f"  {label}")
        PA = C.evaluate(th_orig, mu, Sigma)
        PB = C.evaluate(th_emu, mu, Sigma)
        results.append((label, PA, PB))

    s_all = make_plot(results, (0, 2, 4),
                      "test_matched_grid_P0P2P4.png",
                      f"fkpt multipoles (matched loop grid + nw): original vs emulator "
                      f"(z={C.Z}, AP=1)  — with P4")
    s_p02 = make_plot(results, (0, 2),
                      "test_matched_grid_P0P2.png",
                      f"fkpt multipoles (matched loop grid + nw): original vs emulator "
                      f"(z={C.Z}, AP=1)  — P0,P2 only")

    print("\nMax |emulator - original| per case:")
    print(f"  {'case':26s} {'with P4':>9s} {'P0,P2 only':>12s}")
    for (label, wa), (_, wp) in zip(s_all, s_p02):
        print(f"  {label:26s} {wa:8.2f}% {wp:11.2f}%")


if __name__ == "__main__":
    main()
