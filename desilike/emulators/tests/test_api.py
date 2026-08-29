"""Tests for `desilike.emulators.emulate`.

The first test is the important one: it pins the behaviour whose absence invalidated a whole
comparison in practice.
"""

import numpy as np
import pytest

from desilike.base import Calculator, Variable
from desilike.emulators import Emulator, emulate, Space

K = np.linspace(0.01, 0.3, 20)


class Toy(Calculator):
    """A minimal calculator exposing `h` (not `H0`) and `amplitude`."""

    def __init__(self, h=0.7, amplitude=1.):
        self.h = h
        self.amplitude = amplitude

    def __call__(self):
        self.pk = self.amplitude * K**(-1.5) * (1. + self.h * K)
        return self.pk

    def tree_flatten(self):
        return [self.pk], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk = children[0]
        return obj


def toy():
    return Toy(h=Variable('h', value=0.7), amplitude=Variable('amplitude', value=1.))


def box():
    return Space(limits={'h': (0.6, 0.8), 'amplitude': (0.5, 2.)})


def test_unknown_parameter_raises_instead_of_being_ignored():
    """CompiledGraph merges the given dict with defaults and reads only the names it knows, so
    'H0' would be silently dropped and h left at its default -- which is exactly how a whole
    comparison ran with one parameter frozen and nothing said so."""
    emu = Emulator(toy(), box())
    with pytest.raises(ValueError, match='H0'):
        emu.compute({'H0': 70., 'amplitude': 1.2})


def test_the_error_suggests_a_close_name():
    emu = Emulator(toy(), box())
    with pytest.raises(ValueError, match='did you mean'):
        emu.compute({'amplitud': 1.})


def test_the_calculators_pytree_state_is_what_is_emulated():
    """Not the return value: a calculator used as a dependency is read through its attributes."""
    emu = Emulator(toy(), box())
    low = emu.compute({'h': 0.65, 'amplitude': 1.})
    high = emu.compute({'h': 0.75, 'amplitude': 2.})
    assert set(low) == {'0'}
    assert not np.allclose(low['0'], high['0'])
    assert np.allclose(high['0'] / 2., emu.compute({'h': 0.75, 'amplitude': 1.})['0'])


def test_emulate_a_desilike_calculator_end_to_end():
    emu = Emulator(toy(), box()).train(budget=3)
    point = {'h': 0.72, 'amplitude': 1.3}
    assert np.allclose(emu.predict(**point)['0'],
                       emu.compute(point)['0'], rtol=1e-6)


def test_partial_parameters_use_defaults():
    """Omitting a parameter is fine -- only unknown names are an error."""
    emu = Emulator(toy(), box())
    assert np.allclose(emu.compute({'h': 0.7})['0'],
                       emu.compute({'h': 0.7, 'amplitude': 1.})['0'])


def test_to_calculator_gives_back_the_original_class():
    """No adapter to type, and what comes back is a calculator of the same class.

    Driven through the graph, as desilike drives calculators: values come from the parameter
    nodes, not from attributes set on the instance. On a real theory the cosmological parameters
    live on a sub-calculator (the template) and are not attributes of the theory at all.
    """
    from desilike.base import compile

    emulated = Emulator(toy(), box()).train(budget=3).to_calculator()
    assert isinstance(emulated, Toy)

    point = {'h': 0.72, 'amplitude': 1.3}
    reference = toy()
    compile(emulated)(point)
    compile(reference)(point)
    assert np.allclose(emulated.pk, reference.pk, rtol=1e-6)


def test_space_is_derived_from_the_parameters_when_omitted():
    """A convenience, not a recommendation -- ref limits are a box, and a covariance is worth
    orders of magnitude more."""
    from desilike.parameter import Parameter

    # ref limits live on Parameter (Variable has none): the emulator reads ref, not prior,
    # because the region to cover is the posterior
    calculator = Toy(h=Parameter('h', value=0.7, ref={'limits': [0.65, 0.75]}),
                     amplitude=Parameter('amplitude', value=1., ref={'limits': [0.8, 1.2]}))
    space = Space(calculator)
    assert set(space.params) == {'h', 'amplitude'}
    assert space.limits['h'] == (0.65, 0.75)

    from desilike.base import compile

    emulated = Emulator(calculator).train(budget=3).to_calculator()  # space derived
    point = {'h': 0.70, 'amplitude': 1.05}
    reference = Toy(h=Parameter('h', value=0.7, ref={'limits': [0.65, 0.75]}),
                    amplitude=Parameter('amplitude', value=1., ref={'limits': [0.8, 1.2]}))
    compile(emulated)(point)
    compile(reference)(point)
    assert np.allclose(emulated.pk, reference.pk, rtol=1e-6)


class Parent(Calculator):
    """Reads its child through an attribute, as a real parent does."""

    def __init__(self, child=None):
        self.child = child

    def __call__(self):
        self.total = self.child.pk.sum()
        return self.total

    def tree_flatten(self):
        return [self.total], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.total = children[0]
        return obj


def test_emulated_calculator_substitutes_into_a_parent():
    """The use case: emulate an expensive sub-calculator and swap it into the pipeline.

    `update()` is only legal during construction (it raises on a constructed node), so the
    supported route is `replace(parent, old, new)` followed by a recompile.
    """
    from desilike.base import compile

    child = toy()
    parent = Parent(child=child)
    reference = compile(parent)({'h': 0.71, 'amplitude': 1.4})

    emulated_child = Emulator(child, box()).train(budget=3).to_calculator()
    swapped = Parent(child=emulated_child)
    graph = compile(swapped)
    assert {param.name for param in graph.params} >= {'h', 'amplitude'}
    assert np.allclose(np.asarray(graph({'h': 0.71, 'amplitude': 1.4})),
                       np.asarray(reference), rtol=1e-5)


def test_replace_swaps_the_dependency_in_place():
    from desilike.base import compile, replace

    child = toy()
    parent = Parent(child=child)
    reference = compile(parent)({'h': 0.69, 'amplitude': 0.9})

    emulated_child = Emulator(child, box()).train(budget=3).to_calculator()
    replace(parent, child, emulated_child)
    assert np.allclose(np.asarray(compile(parent)({'h': 0.69, 'amplitude': 0.9})),
                       np.asarray(reference), rtol=1e-5)


def test_a_subclass_can_divide_out_what_a_theory_knows():
    """The extension point: `pk` is exactly linear in `amplitude`, so a subclass takes it off the
    grid -- without the calculator, the target or `emulate` changing at all."""
    from desilike.emulators.api import CalculatorEmulator

    class Scaled(CalculatorEmulator):

        def select_params(self, names):
            return [name for name in names if name != 'amplitude']

        def transform(self, values, params):
            return {name: value / params['amplitude'] for name, value in values.items()}

        def inverse_transform(self, values, params):
            return {name: value * params['amplitude'] for name, value in values.items()}

    emu = Emulator(toy(), box(), cls=Scaled).train(budget=3)
    assert emu.params == ['h'] and emu.exact_params == ['amplitude']
    point = {'h': 0.72, 'amplitude': 1.3}
    assert np.allclose(emu.predict(**point)['0'], emu.compute(point)['0'], rtol=1e-6)


def test_emulate_builds_and_trains_in_one_call():
    """`Emulator` builds; `emulate` also pays."""
    from desilike.base import compile

    emulated = emulate(toy(), box(), budget=3).to_calculator()
    point = {'h': 0.72, 'amplitude': 1.3}
    reference = toy()
    compile(emulated)(point)
    compile(reference)(point)
    assert np.allclose(emulated.pk, reference.pk, rtol=1e-6)


def test_an_emulated_calculator_responds_to_parameters_on_a_SUB_calculator():
    """The regression the toy tests missed for a while.

    `Toy` holds its own parameter nodes, so reading a dict of nodes captured at fit time happened
    to work. A real theory does not: its cosmological parameters live on a sub-calculator, the
    graph evaluates a copy of the tree, and the captured nodes belong to a different copy whose
    `.value` never leaves its construction default. Measured before the fix: an emulated pt
    returned the fiducial spectrum whatever it was asked for -- silently, and identically to a
    plain emulator, which is what makes it worth a test.
    """
    from desilike.base import compile

    child = toy()
    parent = Parent(child=child)
    emulated = emulate(parent, Space(limits={'h': (0.6, 0.8), 'amplitude': (0.5, 2.)}),
                       budget=3).to_calculator()
    graph = compile(emulated)

    def at(point):
        # an emulated calculator's __call__ returns `self`: it is meant as a dependency, read
        # through its attributes
        graph(point)
        return float(np.asarray(emulated.total))

    low, high = at({'h': 0.65, 'amplitude': 0.8}), at({'h': 0.75, 'amplitude': 1.9})
    assert not np.isclose(low, high), 'the emulated calculator ignored its parameters'

    reference = compile(Parent(child=toy()))
    assert np.isclose(low, float(np.asarray(reference({'h': 0.65, 'amplitude': 0.8}))), rtol=1e-5)
    assert np.isclose(high, float(np.asarray(reference({'h': 0.75, 'amplitude': 1.9}))), rtol=1e-5)


# ── packing several calculators ───────────────────────────────────────────────

class Shaped(Calculator):
    """Like Toy, with a second parameter, so two of them can share one node."""

    def __init__(self, h=0.7, amplitude=1.):
        self.h, self.amplitude = h, amplitude

    def __call__(self):
        self.pk = self.amplitude * K**(-1.5) * (1. + self.h * K)
        return self.pk

    def tree_flatten(self):
        return [self.pk], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk = children[0]
        return obj


def test_a_list_of_calculators_is_packed_into_one_emulator():
    """One graph over both, so anything they share upstream is evaluated once per node instead of
    once per calculator -- which is the entire reason to pack."""
    from desilike.base import compile
    from desilike.emulators import PackedCalculatorEmulator

    shared = Variable('h', value=0.7)
    parts = [Shaped(h=shared, amplitude=Variable('amplitude', value=1.)),
             Shaped(h=shared, amplitude=Variable('other', value=1.))]
    space = Space(limits={'h': (0.6, 0.8), 'amplitude': (0.5, 2.), 'other': (0.5, 2.)})

    emu = Emulator(parts, space)
    assert isinstance(emu, PackedCalculatorEmulator)
    # the shared parameter appears once, not once per part
    assert sorted(emu.params) == ['amplitude', 'h', 'other']
    emu.train(budget=3)

    deployed = emu.to_calculator()
    assert len(deployed) == 2
    point = {'h': 0.72, 'amplitude': 1.3, 'other': 0.9}
    for emulated, reference in zip(deployed, [Shaped(h=Variable('h', value=0.7),
                                                     amplitude=Variable('amplitude', value=1.)),
                                              Shaped(h=Variable('h', value=0.7),
                                                     amplitude=Variable('other', value=1.))]):
        graph, truth = compile(emulated), compile(reference)
        # per-graph: the emulated part carries the whole packed parameter set (the packed
        # prediction needs every one of them), the reference only its own two.
        for pipe in [graph, truth]:
            pipe({name: value for name, value in point.items() if name in pipe.params})
        assert np.allclose(emulated.pk, reference.pk, rtol=1e-6)


def test_a_packed_dependency_that_is_never_read_would_be_pruned():
    """The bug this caught: `_PackedRoot.__call__` returned None without reading `self.calculators`,
    so the graph pruned every part and the packed emulator exposed no parameters at all."""
    from desilike.emulators.api import _PackedRoot
    from desilike.base import compile

    root = _PackedRoot([Shaped(h=Variable('h', value=0.7),
                               amplitude=Variable('amplitude', value=1.))])
    assert {param.name for param in compile(root).params} >= {'h', 'amplitude'}


def test_packing_an_empty_list_is_refused():
    with pytest.raises(ValueError, match='no calculator'):
        Emulator([], Space(limits={'h': (0.6, 0.8)}))


# ── derived parameters ────────────────────────────────────────────────────────

class WithDerived(Calculator):
    """Exposes a derived quantity alongside its state, as a cosmology exposes sigma8."""

    def __init__(self, h=0.7, amplitude=1.):
        self.h, self.amplitude = h, amplitude
        self.scale = Variable('scale', value=0., derived=True)

    def __call__(self):
        self.pk = self.amplitude * K**(-1.5) * (1. + self.h * K)
        self.scale.value = self.amplitude * self.h**2
        return self.pk

    def tree_flatten(self):
        return [self.pk], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk = children[0]
        return obj


def test_derived_parameters_are_emulated_and_written_back():
    """A derived quantity is an output of the pipeline. Emulating only the pytree state leaves
    anything downstream reading it stuck on the construction-time default for ever."""
    from desilike.base import compile

    calculator = WithDerived(h=Variable('h', value=0.7),
                             amplitude=Variable('amplitude', value=1.))
    emu = Emulator(calculator, Space(limits={'h': (0.6, 0.8), 'amplitude': (0.5, 2.)}))
    emu.train(budget=3)
    assert 'scale' in emu.derived_names

    point = {'h': 0.72, 'amplitude': 1.3}
    predicted = emu.predict(**point)
    assert np.isclose(np.asarray(predicted['derived.scale']), 1.3 * 0.72**2, rtol=1e-6)

    emulated = emu.to_calculator()
    graph = compile(emulated)
    _, derived = graph(point, return_derived=True)
    assert np.isclose(np.asarray(derived['scale']), 1.3 * 0.72**2, rtol=1e-6)
    # and it moves, which is the whole point
    _, other = graph({'h': 0.62, 'amplitude': 0.7}, return_derived=True)
    assert not np.isclose(np.asarray(derived['scale']), np.asarray(other['scale']))


class DictChild(Calculator):
    """A calculator whose state is a dict child, the way a cosmology's requirement results are.

    `CosmoprimoCosmology` flattens to `[parameter dict, {'tt': ..., 'ee': ...}, ...]`, so the
    children are not arrays and the emulator has to reach through the pytree to reach the leaves.
    """

    def __init__(self, h=0.7):
        self.h = h

    def __call__(self):
        self.cl = {'tt': self.h * K, 'ee': self.h**2 * K}
        return self

    def tree_flatten(self):
        return [self.cl], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.cl = children[0]
        return obj


def dict_child():
    return DictChild(h=Variable('h', value=0.7))


def test_a_dict_child_is_flattened_through_the_pytree():
    """`jnp.asarray` on a dict child is `dtype object is not a valid JAX array type`, so the
    children are flattened through the tree rather than one level."""
    emu = Emulator(dict_child(), Space(limits={'h': (0.6, 0.8)}))
    emu.train(budget=2)
    assert emu.children_leafnames == ['0.ee', '0.tt']
    assert np.allclose(emu.to_calculator()().cl['tt'], 0.7 * K, rtol=1e-10)


def test_a_trained_emulator_round_trips_through_hdf5(tmp_path):
    """The state must hold no live JAX object. Keeping a `PyTreeDef` in it -- the obvious way to
    remember the leaf layout -- makes every emulator of a dict-child calculator unsaveable, and
    only at the end of a training run: 'cannot write PyTreeDef to HDF5'. The leaf paths carry the
    same information and are strings.
    """
    emu = Emulator(dict_child(), Space(limits={'h': (0.6, 0.8)}))
    emu.train(budget=2)
    reloaded = Emulator.read(emu.write(str(tmp_path / 'dict_child.h5')))

    assert reloaded.children_leafnames == emu.children_leafnames
    before, after = emu.predict(h=0.72), reloaded.predict(h=0.72)
    assert all(np.allclose(after[key], before[key], rtol=1e-12, atol=0.) for key in before)


def test_a_training_restored_entirely_from_a_checkpoint_still_deploys(tmp_path):
    """A checkpoint that covers every node means `compute` is never called, so the pytree
    structure is never captured -- and `to_calculator()` then refused, which defeats checkpoints
    on exactly the long trainings they exist for."""
    checkpoint = str(tmp_path / 'nodes.npz')
    first = Emulator(toy(), box())
    first.train(budget=2, checkpoint=checkpoint, verbose=False)

    restored = Emulator(toy(), box())
    restored.train(budget=2, checkpoint=checkpoint, verbose=False)
    assert np.allclose(restored.predict(h=0.72, amplitude=1.3)['0'],
                       first.predict(h=0.72, amplitude=1.3)['0'], rtol=1e-12)
    # the part that used to raise
    assert restored.to_calculator()() is not None


def test_read_does_not_require_knowing_the_subclass(tmp_path):
    """`Emulator` is a factory -- a function -- so `read` is hung off it, and a saved emulator
    comes back as whatever subclass wrote it without the caller importing that class."""
    emu = Emulator(toy(), box())
    emu.train(budget=2, verbose=False)
    reloaded = Emulator.read(emu.write(str(tmp_path / 'toy.h5')))
    assert np.allclose(reloaded.predict(h=0.72, amplitude=1.3)['0'],
                       emu.predict(h=0.72, amplitude=1.3)['0'], rtol=1e-12)


class TupleChild(Calculator):
    """A calculator whose child is a TUPLE of arrays, not a dict or a bare array.

    The point of persisting jax's own `PyTreeDef`: reassembling from the leaf paths alone cannot
    tell a tuple from a list or a dict, so this would come back the wrong shape -- silently, and
    only once deployed.
    """

    def __init__(self, h=0.7):
        self.h = h

    def __call__(self):
        self.pair = (self.h * K, self.h**2 * K)
        return self

    def tree_flatten(self):
        return [self.pair], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pair = children[0]
        return obj


def test_a_tuple_child_survives_the_round_trip(tmp_path):
    emu = Emulator(TupleChild(h=Variable('h', value=0.7)), Space(limits={'h': (0.6, 0.8)}))
    emu.train(budget=2, verbose=False)

    rebuilt = emu.to_calculator()()
    assert isinstance(rebuilt.pair, tuple), f'container type lost: {type(rebuilt.pair).__name__}'
    assert np.allclose(rebuilt.pair[0], 0.7 * K, rtol=1e-10)

    # and through a file, where the structure travels as the pickled aux
    reloaded = Emulator.read(emu.write(str(tmp_path / 'tuple.h5')))
    assert reloaded.children_treedef == emu.children_treedef
