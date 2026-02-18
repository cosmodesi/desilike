#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt


import sys
# sys.path.insert(0, '/global/homes/n/nishavk/FOLPSpipe/')
# sys.path.insert(0, '/global/homes/n/nishavk/FOLPSpipe/folps/')
sys.path.insert(0, '/global/homes/n/nishavk/fkptjax_muMG/')
sys.path.insert(0, '/global/homes/n/nishavk/isitIDE/')
sys.path.insert(0, '/global/homes/n/nishavk/ISiTGR/')
sys.path.insert(0, '/global/homes/n/nishavk/desilike/')
sys.path.insert(0, '/global/homes/n/nishavk/cosmoprimo/')
# sys.path.insert(0, '/global/homes/n/nishavk/isitIDE/')






# -----------------------------
# Optional JAX settings (debug)
# -----------------------------
os.environ.setdefault("JAX_DISABLE_JIT", "1")  # set "0" for speed once validated
import jax
jax.config.update("jax_enable_x64", True)

# -----------------------------
# desilike / cosmoprimo imports
# -----------------------------
from desilike.theories import Cosmoprimo
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike.theories.galaxy_clustering import fkptjaxTracerPowerSpectrumMultipoles
from desilike import parameter
from cosmoprimo.fiducial import DESI

import folps
print(folps.__file__)


# ============================================================
# USER SETTINGS
# ============================================================
z_eff = 0.295
k = np.linspace(0.02, 0.20, 40)
ells = (0, 2)

MG_model = "IDE"
# mg_variant = "mu_OmDE"
ide_variant = "IDEModel1"
beyond_eds = False     # set False for EdS kernels
rescale_PS = False

# Two mu0 values to compare
mu0_a = 0.0
mu0_b = 0.3

# ---- nuisance params (STANDARD basis names) ----
freedom = "max"
prior_basis = "standard"   # alias -> standard_folps
tracer_tag = "BGS"
b3_coev = True
shotnoise_opt = 1e4

b1 = 1.70
b2 = -0.45
bs2 = 0.0
b3nl = 0.0
alpha0, alpha2, alpha4 = 3.0, -29.0, 0.0
ctilde = 0.0
alpha0shot, alpha2shot = 0.08, -8.0

# ---- cosmology ----
h = 0.6711
ombh2 = 0.022
omch2 = 0.122
As = 2e-9
ns = 0.965
Neff = 3.046
mnu = 0.06  # eV


# ============================================================
# Define cosmology class (ISiTGR via Cosmoprimo)
# ============================================================

# ----- (A) mu0 = 0.0 -----
# cosmo = Cosmoprimo(engine="isitgr", MG_parameterization="muSigma", N_eff=Neff, m_ncdm=[mnu])
cosmo = Cosmoprimo(engine="isitide", dark_energy_model="IDEModel1", N_eff=Neff, m_ncdm=[mnu])
beta = 0.0

cosmo.init.params["h"].update(value=h)
cosmo.init.params["omega_b"].update(value=ombh2)
cosmo.init.params["omega_cdm"].update(value=omch2)
cosmo.init.params["logA"].update(value=float(np.log(1e10 * As)))
cosmo.init.params["n_s"].update(value=ns)

# # ensure mu0 exists (depends on build); add if missing
# if "mu0" not in cosmo.init.params:
#     cosmo.init.params.data.append(parameter.Parameter(basename="mu0", value=0.0, fixed=True))
# cosmo.init.params["mu0"].update(value=float(mu0_a), fixed=True)

if "beta" not in cosmo.init.params:
    cosmo.init.params.data.append(parameter.Parameter(basename="beta", value=0.0, fixed=True))
cosmo.init.params["beta"].update(value=float(beta), fixed=True)

# ============================================================
# Defining template (DirectPowerSpectrumTemplate)
# ============================================================
template = DirectPowerSpectrumTemplate(z=float(z_eff), fiducial=DESI(), cosmo=cosmo)
template.init.update(with_now="peakaverage")  # required by fkptjax path (needs pknow_dd)


# ============================================================
# Calling fkptjax tracer (mu0=0.0)
# ============================================================
theory_results_eds = fkptjaxTracerPowerSpectrumMultipoles()
theory_results_eds.init.update(
    freedom=freedom,
    prior_basis=prior_basis,
    tracer=tracer_tag,
    template=template,
    k=np.array(k, dtype=float),
    ells=list(ells),
    model=MG_model,
    # mg_variant=mg_variant,
    ide_variant= "IDEModel1",
    b3_coev=bool(b3_coev),
    beyond_eds=bool(False),
    rescale_PS=bool(rescale_PS),
    shotnoise=float(shotnoise_opt),
)

theory_results_Beds = fkptjaxTracerPowerSpectrumMultipoles()
theory_results_Beds.init.update(
    freedom=freedom,
    prior_basis=prior_basis,
    tracer=tracer_tag,
    template=template,
    k=np.array(k, dtype=float),
    ells=list(ells),
    model=MG_model,
    # mg_variant=mg_variant,
    ide_variant= "IDEModel1",
    b3_coev=bool(b3_coev),
    beyond_eds=bool(True),
    rescale_PS=bool(rescale_PS),
    shotnoise=float(shotnoise_opt),
)


# fix nuisance params (standard basis names)
for name, val in dict(
    b1=b1, b2=b2, bs2=bs2, b3nl=b3nl,
    alpha0=alpha0, alpha2=alpha2, alpha4=alpha4,
    ctilde=ctilde, alpha0shot=alpha0shot, alpha2shot=alpha2shot
).items():
    if name in [theory_results_eds.init.params, theory_results_Beds.init.params]:
        theory_results_eds.init.params[name].update(fixed=True, value=float(val))
        theory_results_Beds.init.params[name].update(fixed=True, value=float(val))

# Pell_GR = theory_results()  # evaluate

Pell_eds = theory_results_eds(beta=0.1)
Pell_Beds = theory_results_Beds(beta=0.1)

# unpack multipoles for eds and Beds
Pell_eds = np.asarray(Pell_eds)
P0_eds = Pell_eds[0]
P2_eds = Pell_eds[1]

Pell_Beds = np.asarray(Pell_Beds)
P0_Beds = Pell_Beds[0]
P2_Beds = Pell_Beds[1]

P0_rel_err = np.abs(P0_Beds - P0_eds)*100 / P0_eds
P2_rel_err = np.abs(P2_Beds - P2_eds)*100 / P2_eds

# unpack multipoles for mu0=0
# Pell_GR = np.asarray(Pell_GR)
# P0_GR = Pell_GR[0]
# P2_GR = Pell_GR[1]

# # ----- (B) mu0 = 0.3 -----
# Pell_mu0 = theory_results(beta=-0.1)  # evaluate

# # unpack multipoles for mu0=0
# Pell_mu0 = np.asarray(Pell_mu0)
# P0_mu0 = Pell_mu0[0]
# P2_mu0 = Pell_mu0[1]

# ============================================================
# Plot: monopole + quadrupole, both mu0 values on one figure
# ============================================================

fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.2), gridspec_kw={'height_ratios': [3, 1]})
axs[0].plot(k, k*P0_eds, color='b', label=r"$P_0(k)$, EdS")
axs[0].plot(k, k*P2_eds, color='r', label=r"$P_2(k)$, EdS")
axs[0].plot(k, k*P0_Beds, color='b', ls="--", label=r"$P_0(k)$, B-EdS")
axs[0].plot(k, k*P2_Beds, color='r', ls="--", label=r"$P_2(k)$, B-EdS")
# axs[0].set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
axs[0].set_ylabel(r"$P_\ell(k)\,[(\mathrm{Mpc}/h)^3]$")
axs[0].set_title(fr"z={z_eff}, {MG_model}/{ide_variant}, $\beta=0.1$")
axs[0].grid(alpha=0.25)
axs[0].legend(ncols=2, fontsize=10)

axs[1].plot(k, P0_rel_err, color='b', label=r"$\Delta P_0(k)/P_0(k)$")
axs[1].plot(k, P2_rel_err, color='r', label=r"$\Delta P_2(k)/P_2(k)$")
axs[1].set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
axs[1].set_ylabel(r"$\%$ Diff")
# axs[1].set_title(fr"z={z_eff}, {MG_model}/{ide_variant}, $\beta=-0.1$")
axs[1].grid(alpha=0.25)
axs[1].legend(ncols=1, fontsize=10)

plt.tight_layout()

# plt.figure(figsize=(8.5, 5.2))

# plt.plot(k, k*P0_eds, lw=2.2, color='b', label=r"$P_0(k)$, EdS")
# plt.plot(k, k*P2_eds, lw=2.2, color='r', label=r"$P_2(k)$, EdS")

# plt.plot(k, k*P0_Beds, lw=2.2, color='b', ls="--", label=r"$P_0(k)$, Beyond EdS")
# plt.plot(k, k*P2_Beds, lw=2.2, color='r', ls="--", label=r"$P_2(k)$, Beyond EdS")

# plt.xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
# plt.ylabel(r"$P_\ell(k)\,[(\mathrm{Mpc}/h)^3]$")
# plt.title(fr"z={z_eff}, {MG_model}/{ide_variant}, $\beta=-0.1$")
# plt.grid(alpha=0.25)
# plt.legend(ncols=2, fontsize=10)
# plt.tight_layout()
# plt.show()

plt.savefig('./IDE_plots_new/isitide_eds_vs_Beds_beta_pos01.png', dpi=300)