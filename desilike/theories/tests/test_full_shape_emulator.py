"""Tests for the FOLPSD exact-scaling emulator.

Slow by nature -- each node is a FOLPS one-loop evaluation -- so the budget is the smallest that
still separates the two emulators, and the assertions are the measured numbers rather than round
ones.
"""

import numpy as np
import pytest

from desilike.base import compile, replace
from desilike.emulators import Emulator, Space, CalculatorEmulator
from desilike.theories.galaxy_clustering.full_shape import FOLPSDEmulator

Z = 0.8


def template():
    """The linear template.

    Measured, before reaching for `engine='eisenstein_hu'` to speed this up: it does not.  The
    cost here is one FOLPS one-loop evaluation per node, not the Boltzmann call, so the suite runs
    the same -- and the thresholds below are measured numbers tied to the cosmology, so swapping
    the engine invalidates them (0.058 against a 1e-3 gate).
    """
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template

    return DirectSpectrum2Template(z=Z)
LIMITS = {'h': (0.66, 0.70), 'omega_cdm': (0.115, 0.125), 'logA': (2.95, 3.15),
          'w0_fld': (-1.15, -0.85), 'wa_fld': (-0.5, 0.5)}
CENTRE = {name: 0.5 * (low + high) for name, (low, high) in LIMITS.items()}


def theory():
    from desilike.theories.galaxy_clustering import FOLPSTracerSpectrum2Poles

    return FOLPSTracerSpectrum2Poles(template=template())


def evaluate(calculator, point):
    graph = compile(calculator)
    names = [param.name for param in graph.params if not param.derived]
    return np.asarray(graph({name: value for name, value in point.items() if name in names}))


@pytest.fixture(scope='module')
def emulators():
    built = {}
    for label, cls in [('plain', CalculatorEmulator), ('folpsd', FOLPSDEmulator)]:
        pipeline = theory()
        emulator = Emulator(pipeline.pt, Space(limits=LIMITS), cls=cls)
        emulator.train(budget=1)
        swapped = theory()
        replace(swapped, swapped.pt, emulator.to_calculator())
        built[label] = (emulator, swapped)
    return built


def test_the_frozen_parameters_leave_the_grid(emulators):
    plain, folpsd = emulators['plain'][0], emulators['folpsd'][0]
    assert folpsd.params == ['h', 'omega_cdm']
    assert folpsd.exact_params == ['logA', 'w0_fld', 'wa_fld']
    # ... and cost fewer nodes for it: 5 against 11 at the same budget
    assert len(folpsd.nodes(budget=1)) < len(plain.nodes(budget=1))


def test_the_background_scalars_are_routed_exactly(emulators):
    """sigma8, f, qpar and the AP grid are computed by the run-time provider, not fitted over the
    pt's grid. A residual against the PROVIDER means the routing is not reaching the prediction,
    which is the bug this catches.

    Against the exact pt they agree only to the provider's own accuracy, and asserting otherwise
    was asking for something the construction cannot give: `ScalingScalarsEmulator` fits five
    smooth corrections and rebuilds each scalar as `analytic_core x correction`. The core carries
    (w0, wa) and the exp(dlogA/2) amplitude exactly -- measured, an UNfitted provider reproduces
    the pt's sigma8 to every digit -- but the correction is interpolated, and at the default
    `scalars_budget` that leaves 1.2e-6 on sigma8 here (still ~1e-6 at budget 3). So the exact
    comparison below is a gross-error gate, and the tight one is against the provider.
    """
    from desilike.theories.galaxy_clustering import FOLPSPTSpectrum2Poles

    emulator = emulators['folpsd'][0]
    moved = {**CENTRE, 'w0_fld': -0.85, 'logA': 3.15}

    reference = FOLPSPTSpectrum2Poles(template=template())
    graph = compile(reference)
    names = [param.name for param in graph.params if not param.derived]
    graph({name: value for name, value in moved.items() if name in names})
    truth = reference.tree_flatten()[0]

    predicted = emulator.predict(**moved)
    # The routing itself: what comes out must be the provider's value and not the interpolant's.
    scalars = emulator.compute_scalars(moved)
    for key in ('sigma8', 'fsigma8', 'f', 'qpar', 'qper'):
        assert np.allclose(np.asarray(predicted[key]), np.asarray(scalars[key]),
                           rtol=1e-12), key
    # not kap/jac: with the h preconditioning on, the AP grid carries the dilation (q / s) while
    # the tables live in the reference frame, so those children deliberately differ from the
    # undilated truth. What must match is the observable, which the accuracy test covers.
    for key in ('sigma8', 'f', 'qpar', 'qper'):
        assert np.allclose(np.asarray(predicted[key]),
                           np.asarray(truth[emulator.children_leafnames.index(key)]),
                           rtol=1e-4), key


def test_to_calculator_agrees_with_predict(emulators):
    """The bug this caught once: an emulated calculator that ignores its parameters and returns
    the fiducial state, indistinguishably from a plain emulator."""
    emulator, swapped = emulators['folpsd']
    fast = emulator.to_calculator()
    graph = compile(fast)
    names = [param.name for param in graph.params if not param.derived]
    moved = {**CENTRE, 'w0_fld': -0.85, 'logA': 3.15}
    graph({name: value for name, value in moved.items() if name in names})
    assert np.allclose(np.asarray(fast.sigma8),
                       np.asarray(emulator.predict(**moved)['sigma8']),
                       rtol=1e-12)


def test_the_routing_beats_expanding_the_same_parameters(emulators):
    """With the shape parameters held at the space centre, the only thing moving is what the
    routing handles. Measured: median 3.6e-05 against 2.1e-02, a factor ~580 -- at 5 nodes
    rather than 11."""
    reference = theory()
    rng = np.random.default_rng(3)
    errors = {label: [] for label in emulators}
    for _ in range(3):
        point = dict(CENTRE, b1=1.9)
        for name in ('logA', 'w0_fld', 'wa_fld'):
            point[name] = float(rng.uniform(*LIMITS[name]))
        truth = evaluate(reference, point)
        for label, (_, swapped) in emulators.items():
            guess = evaluate(swapped, point)
            errors[label].append(
                np.max(np.abs(guess / np.where(truth == 0., 1., truth) - 1.)))
    assert np.median(errors['folpsd']) < 1e-3
    assert np.median(errors['folpsd']) < 0.05 * np.median(errors['plain'])


def test_the_routing_applies_with_nothing_frozen():
    """A space that varies none of the routed parameters is not an error.

    It used to be refused -- `buys nothing over the plain emulator` -- which was wrong.  An
    EXPANDED parameter still moves sigma8, f and the AP grid, and the routing applies those
    channels exactly rather than leaving them to the polynomial, so the grid carries only the
    shape residual.  Measured on `h` alone, routed but expanded with preconditioning off: median
    max|dP/P| 2.8e-03, against 3.1e-03 for the plain emulator.  What frozen parameters buy on top
    of that is their dimension off the grid, and exactness instead of an interpolation.
    """
    emulator = Emulator(theory().pt, Space(limits={'h': (0.66, 0.70)}), cls=FOLPSDEmulator)
    assert emulator.params == ['h'] and emulator.exact_params == []


def test_the_theory_declares_its_own_emulator():
    """The caller should not have to know which subclass to import. `Emulator` asks the
    calculator, so a pt is routed exactly by default and the generic expansion is the thing you
    have to ask for."""
    pt = theory().pt
    # asked on the INSTANCE: the FOLPS pts dispatch on `output`, so the answer is not a property
    # of the class alone
    assert pt.get_emulator_cls() is FOLPSDEmulator

    space = Space(limits={'h': (0.66, 0.70), 'w0_fld': (-1.1, -0.9)})
    assert isinstance(Emulator(pt, space), FOLPSDEmulator)
    assert Emulator(pt, space).exact_params == ['w0_fld']
    # explicit still wins, including to force the generic one
    forced = Emulator(pt, space, cls=CalculatorEmulator)
    assert type(forced) is CalculatorEmulator and forced.exact_params == []


def test_every_folps_pt_declares_a_routing():
    """The bispectrum and fkpt pts have their own exact channels; all three are reachable the
    same way, so nothing downstream has to know which is which."""
    from desilike.theories.galaxy_clustering import (FOLPSPTSpectrum2Poles, FOLPSPTSpectrum3Poles,
                                                     FKPTJAXPTSpectrum2Poles)
    from desilike.theories.galaxy_clustering.full_shape import FOLPSD3PolesEmulator, FKPTEmulator

    assert FOLPSPTSpectrum2Poles().get_emulator_cls() is FOLPSDEmulator
    assert FOLPSPTSpectrum3Poles().get_emulator_cls() is FOLPSD3PolesEmulator
    assert FKPTJAXPTSpectrum2Poles.get_emulator_cls() is FKPTEmulator
    # fkpt routes the amplitude only: its growth comes from an internal ODE in (z, Omega_m),
    # which is blind to w0/wa, so rescaling it would be wrong (measured: dchi2 ~ 14 worse)
    assert 'growth' not in FKPTEmulator.transform.__doc__ if FKPTEmulator.transform.__doc__ else True


# ── every routing must actually run ───────────────────────────────────────────
# Wiring tests (which class does `get_emulator_cls` return?) let two real bugs through: FKPT
# dividing its q's by a `scale` it never defines, and `compute` reaching for a `precondition`
# only one subclass had. Both are NameError/AttributeError on the first prediction, and both
# were invisible because nothing executed those classes. One prediction each closes that.

@pytest.mark.parametrize('label', ['3poles', 'fkpt'])
def test_every_routing_predicts(label):
    from desilike.theories.galaxy_clustering import (FOLPSPTSpectrum3Poles,
                                                     FKPTJAXPTSpectrum2Poles)

    if label == 'fkpt':
        # fkptjax is an optional backend: without it there is nothing to exercise, and a failure
        # here would say "the routing is broken" when the truth is "the package is not installed".
        pytest.importorskip('fkptjax')
    make = {'3poles': FOLPSPTSpectrum3Poles, 'fkpt': FKPTJAXPTSpectrum2Poles}[label]
    pt = make(template=template())
    emulator = Emulator(pt, Space(limits=LIMITS))
    assert type(emulator) is not CalculatorEmulator, 'the pt should declare its own routing'
    emulator.train(budget=0)

    point = dict(CENTRE, w0_fld=-0.9, logA=3.1)
    predicted = emulator.predict(**point)          # <- the call both bugs died on
    assert predicted, 'no output'
    for name, value in predicted.items():
        assert np.all(np.isfinite(np.asarray(value))), name

    # the substituted scalars must be the provider's, exactly: they are computed, not fitted
    scalars = emulator.compute_scalars(point)
    for name in ('qpar', 'qper', 'sigma8', 'fsigma8'):
        assert np.allclose(np.asarray(predicted[name]),
                           np.asarray(scalars[name]), rtol=1e-12), name

    # ... and it deploys
    from desilike.base import compile

    fast = emulator.to_calculator()
    graph = compile(fast)
    names = [param.name for param in graph.params if not param.derived]
    graph({name: value for name, value in point.items() if name in names})
    assert np.all(np.isfinite(np.asarray(fast.sigma8)))


def test_the_scalar_provider_is_emulated_automatically(tmp_path):
    """The routing needs (qpar, qper, f, sigma8) at the parameters asked for. Left to a live
    ScalingScalars that is a Boltzmann call per prediction -- the cost the emulator exists to
    remove -- so `train` emulates the provider too, and `write` carries it."""
    from cosmoprimo.emulators.tools import Emulator as Template

    pt = theory().pt
    emulator = Emulator(pt, Space(limits=LIMITS))
    emulator.train(budget=0, scalars_budget=1)
    assert emulator._state_scalars is not None, 'no provider was trained'

    point = dict(CENTRE, w0_fld=-0.95)
    reference = emulator.predict(**point)['sigma8']

    # the provider travels inside the emulator's state: without it a loaded emulator cannot
    # predict at all, since a Calculator is not part of any state
    loaded = Template.read(emulator.write(str(tmp_path / 'pt.h5')))
    assert loaded._state_scalars is not None
    assert np.allclose(np.asarray(loaded.predict(**point)['sigma8']),
                       np.asarray(reference), rtol=1e-12, atol=0.)


def test_a_supplied_provider_is_used_as_is():
    """`scalars=` opts out of the automatic one -- for checking against the exact pipeline."""
    from desilike.theories.galaxy_clustering.template import ScalingScalars

    pt = theory().pt
    provider = ScalingScalars(z=Z)
    emulator = Emulator(pt, Space(limits=LIMITS), scalars=provider)
    emulator.train(budget=0)
    assert emulator._state_scalars is None, 'a supplied provider must not be re-fitted'
    assert emulator.input_scalars is provider


# ── the template decides what is routed ───────────────────────────────────────
# The routing used to be `frozen = ('w0_fld', 'wa_fld', 'logA')`, a class constant on the
# emulator -- cosmology parameter names, in a class with no business knowing them, and wrong for
# every template but the direct one.  The template is asked now.

def shapefit_theory():
    from desilike.theories.galaxy_clustering import (FOLPSTracerSpectrum2Poles,
                                                     ShapeFitSpectrum2Template)

    return FOLPSTracerSpectrum2Poles(template=ShapeFitSpectrum2Template(z=Z))


SHAPEFIT_LIMITS = {'qpar': (0.95, 1.05), 'qper': (0.95, 1.05), 'df': (0.85, 1.15),
                   'dA': (0.85, 1.15), 'dm': (-0.05, 0.05)}


def test_a_shapefit_template_routes_its_own_parameters():
    """`(qpar, qper, df, dA)` reach the tables only through the background scalars, so they are
    routed and `dm` is the only expanded parameter -- no cosmology anywhere."""
    emulator = Emulator(shapefit_theory().pt, Space(limits=SHAPEFIT_LIMITS))
    assert type(emulator) is FOLPSDEmulator
    assert emulator.params == ['dm']
    assert sorted(emulator.exact_params) == ['dA', 'df', 'qpar', 'qper']
    # a ShapeFit template varies no `h`, so there is nothing to precondition -- and leaving the
    # class value standing makes `emulator_namespace` a KeyError at the first prediction
    assert emulator.precondition == ()
    # closed-form scalars: nothing to emulate, so no provider is built or trained
    assert emulator._emulator_cls_scalars is None


def test_the_shapefit_routing_is_exact():
    """At budget 0 nothing is interpolated, so a routed parameter must reproduce the exact
    pipeline to machine precision -- however far it moves, including outside the box.

    rms(diff)/rms(ref), not np.allclose: its default atol=1e-8 passes on anything near zero.
    """
    space = Space(limits=SHAPEFIT_LIMITS)
    emulator = Emulator(shapefit_theory().pt, space)
    emulator.train(budget=0)
    emulated = shapefit_theory()
    replace(emulated, emulated.pt, emulator.to_calculator())
    exact, centre = shapefit_theory(), dict(space.center)

    reference = evaluate(exact, centre)
    for shift in [{'qpar': 1.05, 'qper': 0.95, 'df': 1.15, 'dA': 0.85},
                  {'qpar': 1.15, 'qper': 0.85, 'df': 1.4, 'dA': 0.6}]:   # the second is OUTSIDE
        point = dict(centre, **shift)
        truth = evaluate(exact, point)
        # `evaluate` drops names the graph does not know, so a routed parameter that never
        # reached the pipeline would leave both sides at the centre and score a perfect 0
        assert np.max(np.abs(truth / reference - 1.)) > 1e-6, 'the exact pipeline did not move'
        predicted = evaluate(emulated, point)
        score = np.sqrt(np.mean((predicted - truth)**2) / np.mean(truth**2))
        assert score < 1e-12, f'{shift}: {score:.3e}'


def test_a_template_that_routes_everything_says_so():
    """A BAO template pins the spectra to the fiducial, so every parameter it has is routed and
    nothing is left to expand.  `Emulator.__init__` would raise `select_params left nothing to
    expand`; the pt emulator says what that means."""
    from desilike.theories.galaxy_clustering import (FOLPSTracerSpectrum2Poles,
                                                     BAOSpectrum2Template)

    pt = FOLPSTracerSpectrum2Poles(template=BAOSpectrum2Template(z=Z)).pt
    with pytest.raises(ValueError, match='routes every parameter'):
        Emulator(pt, Space(limits={'qpar': (0.98, 1.02), 'qper': (0.98, 1.02)}))



def test_a_saved_emulator_can_be_deployed(tmp_path):
    """`Emulator.read(path).to_calculator(template=...)`.

    A Calculator is not part of any state, so a read-back emulator could predict into a dict and
    nothing more -- `to_calculator` died on `type(self.calculator)`.  What it actually needs is
    the calculator's CLASS and the parameter nodes the emulated object holds, and both are state;
    the constructor arguments are the caller's, and are passed in.
    """
    from cosmoprimo.emulators.tools import Emulator as Template
    from desilike.theories.galaxy_clustering import ShapeFitSpectrum2Template

    space = Space(limits=SHAPEFIT_LIMITS)
    emulator = Emulator(shapefit_theory().pt, space)
    emulator.train(budget=0)
    point = dict(space.center, qpar=1.02, df=1.05)

    live = shapefit_theory()
    replace(live, live.pt, emulator.to_calculator())
    reference = evaluate(live, point)

    loaded = Template.read(emulator.write(str(tmp_path / 'pt.h5')))
    assert loaded.calculator is None and loaded.graph is None, 'a saved emulator carries neither'
    # the template is the CALLER's, not something that came out of the file
    deployed = loaded.to_calculator(template=ShapeFitSpectrum2Template(z=Z))
    back = shapefit_theory()
    replace(back, back.pt, deployed)
    assert np.max(np.abs(evaluate(back, point) / reference - 1.)) == 0.


# ── the cosmology's parameter basis ───────────────────────────────────────────
# Every name the routing hard-codes -- `h`, `logA`, `w0_fld` -- is one parameterisation among
# several, and matching them against what a pipeline SAMPLES fails silently: the preconditioning
# switches itself off, the frozen set shrinks, the analytic core reads a fiducial value. So the
# names are matched by QUANTITY (`find_conflicts`) and the values come from the provider.

def test_find_conflicts_matches_by_quantity():
    from desilike.theories.primordial_cosmology import find_conflicts

    assert find_conflicts('h', ['H0', 'A_s', 'omega_cdm']) == ['H0']
    assert find_conflicts('logA', ['H0', 'A_s', 'omega_cdm']) == ['A_s']
    assert find_conflicts('omega_cdm', ['H0', 'A_s', 'Omega_m']) == ['Omega_m']
    assert find_conflicts('w0_fld', ['w', 'wa']) == ['w']          # pure aliases too
    assert find_conflicts('w0_fld', ['H0', 'A_s']) == []


def h0_theory():
    """The same pipeline, parameterised in `H0` and `A_s` rather than `h` and `logA`."""
    from desilike.theories.galaxy_clustering import (FOLPSTracerSpectrum2Poles,
                                                     DirectSpectrum2Template)
    from desilike.theories import CosmoprimoCosmology
    from desilike.parameter import Parameter, VariableCollection

    proposed = CosmoprimoCosmology.propose_params()
    params = VariableCollection([param for param in proposed
                                 if param.name not in ('h', 'logA')])
    params.set(Parameter('H0', value=67.36, prior=dict(limits=[62., 76.]),
                         ref=dict(dist='norm', loc=67.36, scale=0.5)))
    params.set(Parameter('A_s', value=2.083e-9, prior=dict(limits=[1.5e-9, 2.7e-9]),
                         ref=dict(dist='norm', loc=2.083e-9, scale=2e-11)))
    cosmo = CosmoprimoCosmology(engine='class', fiducial='DESI', params=params)
    return FOLPSTracerSpectrum2Poles(template=DirectSpectrum2Template(z=Z, cosmo=cosmo))


H0_LIMITS = {'H0': (66., 70.), 'A_s': (2.0e-9, 2.2e-9), 'w0_fld': (-1.15, -0.85)}


def test_a_different_parameter_basis_is_routed_the_same():
    """`H0` preconditions the dilation that `h` does, and `A_s` is routed like `logA`.

    Name matching would leave BOTH silently off: an empty `precondition` (quietly less accurate)
    and a smaller frozen set (nodes spent on what could be exact).
    """
    from desilike.theories.primordial_cosmology import find_conflicts

    emulator = Emulator(h0_theory().pt, Space(limits=H0_LIMITS))
    assert type(emulator) is FOLPSDEmulator
    # the QUANTITY is preconditioned -- `precondition` stays in canonical names, and the
    # pipeline's own name for it is resolved where the parameter is actually read
    assert emulator.precondition == ('h',), emulator.precondition
    assert find_conflicts('h', emulator.space.params) == ['H0']
    # `A_s` is routed exactly, as `logA` would be; only `H0` is left on the grid
    assert set(emulator.exact_params) == {'w0_fld', 'A_s'}, emulator.exact_params
    assert emulator.params == ['H0'], emulator.params
