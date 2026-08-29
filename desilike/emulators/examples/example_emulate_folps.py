"""Emulating FOLPSD.

Same four steps as for a cosmology in cosmoprimo -- Space, emulate, train, use -- with a desilike
calculator instead. That is the point of the shared API: `emulate` builds, `train` pays, and
`to_calculator` hands back the thing you started with.

Two ways to use it, shown below:

  A. emulate the theory            -> a callable returning arrays
  B. emulate the PT sub-calculator -> a drop-in dependency, swapped into the pipeline

Run:  python example_emulate_folps.py     (a few minutes at this budget; rerun if it stops early)
"""

import numpy as np

from desilike.base import build, replace
from desilike.theories.galaxy_clustering import (DirectSpectrum2Template,
                                                 FOLPSTracerSpectrum2Poles)
from desilike import setup_logging
from desilike.emulators import Space, Emulator

# training progress goes to a logger, not to stdout: without this the run is silent
setup_logging()

# 1. where the emulator must be accurate -----------------------------------------------------
# `Space(calculator)` takes each Parameter's `ref` -- where the chain is expected to live, not
# where the prior allows. Over-covering is the most expensive mistake available: a prior-width
# box cost 23x a posterior-sized one at equal node count.
#
# A chain or a covariance is worth far more still, because the grid then sits on the posterior's
# principal axes rather than in a rectangle around them -- measured 350x in the median and 3600x
# at the 90th percentile, the largest single lever there is:
#
#     space = Space(samples=chain)                       # best
#     space = Space(mean=best_fit, covariance=fisher)    # good
CENTRE = {'h': 0.6736, 'omega_cdm': 0.1200, 'logA': 3.044}
SIGMA = {'h': 0.006, 'omega_cdm': 0.0012, 'logA': 0.014}
space = Space(limits={name: (CENTRE[name] - 3. * SIGMA[name], CENTRE[name] + 3. * SIGMA[name])
                      for name in CENTRE},
              # per-axis resolution: a different knob from `budget`. The level sets one axis's
              # own error; the budget buys only the interaction terms. Measured -- raising one
              # axis's level cut its error 276x for 4 extra nodes, while raising the budget left
              # every per-axis number unchanged.
              levels={'omega_cdm': 3})

template = DirectSpectrum2Template(z=0.8)
theory = FOLPSTracerSpectrum2Poles(template=template)

# The calculator is passed directly -- no adapter to construct. Unknown parameter names raise
# rather than being silently ignored, the failure mode that once left H0 at its fiducial through
# an entire comparison because desilike exposes `h`:
#     emu(H0=67.36)  ->  ValueError: ... does not expose 'H0' ...

# --- A. emulate the theory itself -------------------------------------------------------------
# `to_calculator()` returns an instance of the original class whose pytree state is predicted
# rather than computed, so it is a drop-in wherever the original went.
emulated_theory = Emulator(theory, space, params=list(CENTRE), budget=2).train(
    checkpoint='folps_theory.npz',   # resumable: a kill costs one node, not the training
    chunk='30min').to_calculator()   # stops cleanly; rerun to continue

# An emulated calculator returns `self`, not what the root's `__call__` returned: it is built to
# be a dependency, read through its attributes. So evaluate the graph and read `.poles` -- which
# is what a parent would do anyway.
def poles(calculator, point):
    graph = build(calculator)
    names = [param.name for param in graph.params if not param.derived]
    graph({name: value for name, value in point.items() if name in names})
    return np.asarray(calculator.poles)


reference = poles(theory, CENTRE)
predicted = poles(emulated_theory, CENTRE)
print(f'\nemulated theory: max |dP/P| = '
      f'{np.max(np.abs(predicted / np.where(reference == 0., 1., reference) - 1.)):.2e}')

# Validation is a comparison of pipelines -- compile both and evaluate. Report the scatter of
# the difference, not its mean: under importance reweighting a constant offset cancels exactly,
# and only the scatter costs effective sample size.
errors = []
rng = np.random.default_rng(42)
for _ in range(10):
    point = {name: float(rng.uniform(*space.limits[name])) for name in CENTRE}
    truth = poles(theory, point)
    guess = poles(emulated_theory, point)
    errors.append(np.max(np.abs(guess / np.where(truth == 0., 1., truth) - 1.)))
errors = np.array(errors)
print(f'over {errors.size} points: median {np.median(errors):.2e}, '
      f'sigma {errors.std():.2e}, worst {errors.max():.2e}')

# --- B. emulate the PT and swap it into the pipeline ------------------------------------------
# A calculator used as a dependency is read through its attributes (`theory` reads `pt.pktable`,
# not `pt()`), so the stand-in carries the pytree state rather than the return value -- which is
# why `to_calculator()` gives back a calculator of the same class.
#
# NOTE `theory.update(pt=...)` does not work on a constructed calculator -- desilike allows
# `update()` only during construction. Use `replace()` and recompile, or build the parent with
# the emulated child directly.
fresh = FOLPSTracerSpectrum2Poles(template=DirectSpectrum2Template(z=0.8))
pt = fresh.pt
emulated_pt = Emulator(pt, space, params=list(CENTRE), budget=2).train(
    checkpoint='folps_pt.npz', chunk='30min').to_calculator()

before = poles(fresh, CENTRE)
replace(fresh, pt, emulated_pt)
after = poles(fresh, CENTRE)
print(f'\nPT swapped into the pipeline: max |dP/P| = '
      f'{np.max(np.abs(after / np.where(before == 0., 1., before) - 1.)):.2e}')

# --- C. the exact channels ---------------------------------------------------------------------
# w0_fld, wa_fld and logA reach the one-loop tables only through four background scalars
# (sigma8(z), f, qpar, qper), so they can be divided out at fit time and put back exactly at
# prediction -- leaving the grid to expand the shape parameters alone. That is a subclass, and
# nothing else changes -- in fact nothing at all, because the theory declares it:
#
#     FOLPSPTSpectrum2Poles.get_emulator_cls()  ->  FOLPSDEmulator
#
# so `Emulator(pt, space)` above already uses the exact routing. Pass
# `cls=CalculatorEmulator` to force the generic expansion instead.
#
# Measured against the plain emulator over the same space, at half the nodes (5 against 11):
# with the shape parameters held at the space centre, so only the routed ones move, median
# max|dP/P| 3.6e-05 against 2.1e-02 -- a factor ~580. With all five varying, 1.0e-03 against
# 1.1e-02.
#
# `h` is expanded rather than frozen, but preconditioned: the tables are dilated back to the
# reference frame, the fixed-Mpc amplitude divided out, and the folps-convention nuisance
# parameters divided by their dilation powers (s^2 for alpha0/2/4, s^3 and s^5 for the shot
# terms) at combine time. So `h`'s expansion carries only the residual. Measured over
# h in [0.62, 0.76] at budget 1: median max|dP/P| 1.1e-03 with it, 2.8e-03 without,
# 3.1e-03 for the plain emulator. Switch it off with a subclass setting `precondition = ()`.
#
# The run-time scalar provider is emulated for you by `train`, over the same space: without that
# every prediction would pay for a Boltzmann call, which is the cost the pt emulator exists to
# remove. Pass `scalars=` to supply your own, `scalars_budget=` to set its resolution.
