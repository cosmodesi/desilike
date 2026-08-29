"""Emulating the CMB spectra an SP4A-style likelihood asks for: LCDM + tau, budget 3.

The cut is the cosmology, not the likelihood. `CosmoprimoCosmology` flattens to its registered
requirement results, so emulating it replaces exactly the Boltzmann call and leaves every
foreground, calibration and window downstream untouched -- those nuisance parameters are cheap
and numerous, and emulating them would be paying to interpolate arithmetic.

Order matters: the likelihoods register their `harmonic.*` requirements (and their ellmax, which
comes from the data) at construction, so the emulator can only be built once they have. Here the
requirements are declared by hand to keep the example runnable without the SP4A data.

Run:  python example_emulate_cmb.py       (~35 min at budget 3; resumable, rerun if it stops)
"""

import numpy as np

from desilike import setup_logging
from desilike.base import build
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.emulators import Emulator, Space

# training progress goes to a logger, not to stdout: without this the run is silent
setup_logging()

# --- 0. the cosmology, with the requirements a CMB likelihood would have registered ----------
# The lens potential is five to seven orders of magnitude below the temperature spectrum: pp
# peaks around 1e-8, ep around 1e-12, against 1e-10 for tt. That is worth knowing before
# comparing anything -- `np.allclose`'s default `atol=1e-8` alone calls every one of these
# spectra equal to zero, and to each other. Compare relatively here, and gate the emulator on
# a chi2 (step 4) rather than on absolute differences.
ELLMAX_LENSED, ELLMAX_POTENTIAL = 2500, 3000

cosmo = CosmoprimoCosmology(engine='camb')
cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': ELLMAX_LENSED}],
                        'harmonic.lens_potential_cl': [{'ellmax': ELLMAX_POTENTIAL}]})


def check_the_target_is_sane():
    """The spectra the emulator will be trained on, against cosmoprimo computed directly.

    The metric is RMS(difference) / RMS(reference), the same one :meth:`validate` defaults to,
    and for the same two reasons: a pointwise ratio divides by zero at every sign change of a
    cross-spectrum and at the unpopulated ell = 0, 1, while an absolute tolerance is meaningless
    across spectra that span seven decades.
    """
    from cosmoprimo.fiducial import DESI

    build(cosmo)({})
    results = [child for child in cosmo.tree_flatten()[0][1:] if isinstance(child, dict)]
    potential = next(child for child in results if 'pp' in child)
    reference = DESI(engine='camb', lensing=True, ellmax_cl=4000,
                     non_linear='mead2016').get_harmonic().lens_potential_cl(ellmax=ELLMAX_POTENTIAL)
    for spectrum in ['pp', 'tp', 'ep']:
        predicted = np.asarray(potential[spectrum])
        expected = np.asarray(reference[spectrum])
        deviation = np.sqrt(np.mean((predicted - expected)**2) / np.mean(expected**2))
        assert deviation < 1e-4, f'{spectrum} is not what cosmoprimo computes ({deviation:.1e})'
        print('  {} max {:.3e}   vs cosmoprimo {:.1e}'.format(
            spectrum, np.max(np.abs(predicted)), deviation))


check_the_target_is_sane()


# --- 1. where it must be accurate: the Planck 2018 posterior --------------------------------
# Space(samples=chain) is the single largest lever there is -- the grid then sits on the
# posterior's principal axes rather than in a rectangle around them, measured 350x in the median
# at equal node count. Swap this Gaussian for your actual chain when you have one; the
# correlations, not the widths, are what it buys you.
#
# Planck 2018 TT,TE,EE+lowE+lensing (Table 2, arXiv:1807.06209), with the strong degeneracies:
# omega_cdm-h at -0.92 and logA-tau at +0.92 are the two that matter.
NAMES = ['omega_b', 'omega_cdm', 'h', 'tau_reio', 'logA', 'n_s']
MEAN = np.array([0.02237, 0.1200, 0.6736, 0.0544, 3.044, 0.9649])
SIGMA = np.array([0.00015, 0.0012, 0.0054, 0.0073, 0.014, 0.0042])
CORRELATION = np.array([
    [1.00, -0.40, 0.35, 0.05, 0.05, 0.55],
    [-0.40, 1.00, -0.92, 0.05, 0.10, -0.50],
    [0.35, -0.92, 1.00, -0.02, -0.05, 0.45],
    [0.05, 0.05, -0.02, 1.00, 0.92, 0.10],
    [0.05, 0.10, -0.05, 0.92, 1.00, 0.10],
    [0.55, -0.50, 0.45, 0.10, 0.10, 1.00]])

space = Space(mean=MEAN, covariance=CORRELATION * np.outer(SIGMA, SIGMA), params=NAMES)
print(space, '-> whitened' if space.is_correlated() else '')


# --- 2. free tau, or the emulator is blind to it ---------------------------------------------
# `tau_reio` is fixed in the proposed parameters. A fixed parameter is not emulated, so every
# spectrum would come back at tau = 0.0544 however the likelihood varied it -- silently, and the
# chain would simply not move in tau.
for param in cosmo.params:
    if param.basename in NAMES:
        param.update(fixed=False)
print('varied:', [param.basename for param in cosmo.params if not param.fixed])


# --- 3. build, size, train -------------------------------------------------------------------
# `CosmoprimoCosmology.get_emulator_cls` declares CMBEmulator, so this is picked up on its own:
# the amplitude and one exp(-tau) per screened leg ('tt' and 'ee' two, 'tp' and 'ep' one, 'pp'
# none) are divided out before the fit and put back at prediction. Both stay on the grid -- with
# lensing neither is exact -- so this buys node count, not exactness.
emu = Emulator(cosmo, space)
print('emulator:', type(emu).__name__, '| expands', emu.params)
print('nodes:', {budget: len(emu.nodes(budget=budget)) for budget in (1, 2, 3)})

# Measured on this space, sigma(dchi2) against `validate`'s default metric:
#
#     budget 1    13 nodes    ~1 min     5.7e-03
#     budget 2    85 nodes    ~8 min     2.0e-04
#     budget 3   365 nodes   ~30 min     ~1e-05 (extrapolated; ~28x per step)
#
# So budget 2 is already far inside what matters -- the gate is sigma <~ 0.5, and 2.0e-04 keeps
# essentially every sample. Budget 3 is headroom, and headroom is the right thing to buy here
# only because the levels are nested: raising the budget later reuses every evaluation already
# made, so starting at 2 and going to 3 costs the difference, not the total. `checkpoint` +
# `chunk` make that resumable -- a kill costs one node, not the training.
emu.train(budget=3, checkpoint='cmb_nodes.npz', chunk='25min')


# --- 4. the gate that matters ----------------------------------------------------------------
# not |dCl/Cl|. The cost of an emulator under importance reweighting is the scatter of dchi2
# against the real covariance -- ESS ~ exp(-(sigma/2)^2), so sigma <~ 0.5 keeps >= 94% of the
# samples, and a constant offset cancels exactly. Substitute your likelihood's covariance here;
# `validate`'s default metric (a ratio of norms, safe across TE's zero crossings) is only a
# stand-in for it.
report = emu.validate(npoints=20, seed=7)
print(report)

point_at_centre = dict(zip(NAMES, MEAN))

# Saved as HDF5, with the spectra stored under their own leaf paths ('1.tt', '2.pp') rather than
# 'child.7', so a dump of the file is readable.
# `read` is hung off the factory, and restores whichever subclass wrote the file.
path = emu.write('cmb_lcdm_tau.h5')
reloaded = Emulator.read(path)
before, after = emu.predict(**point_at_centre), reloaded.predict(**point_at_centre)
assert all(np.allclose(after[key], before[key], rtol=1e-12, atol=0.) for key in before)
print('round trip through {}: identical'.format(path))


# --- 5. deploy -------------------------------------------------------------------------------
# What comes back is a CosmoprimoCosmology whose requirement results are predicted rather than
# computed, so it drops into the pipeline the likelihood already built:
#
#     from desilike.base import replace
#     replace(likelihood, likelihood.cosmo, emu.to_calculator())
#     pipe = build(likelihood)
#
# `update()` is only legal during construction, which is why this is `replace` + recompile.
fast = emu.to_calculator()
graph = build(fast)
names = [param.name for param in graph.params if not param.derived]
graph({name: value for name, value in point_at_centre.items() if name in names})
print('emulated cosmology evaluated; predicted outputs:', len(emu.predict(**point_at_centre)))
