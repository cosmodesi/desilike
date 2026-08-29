"""``emulate`` for desilike calculators -- the same call as in cosmoprimo, no adapter to type.

    from desilike.emulators import emulate

    emu = Emulator(theory)                                  # Space from the parameters' own ref
    emu = Emulator(theory, Space(samples=chain), budget=4)  # or say where accuracy is required
    emu.train()

    emulated = emulate(theory, budget=4).to_calculator()    # or both in one call

Like cosmoprimo's, this is a :class:`~cosmoprimo.emulators.tools.Emulator` subclass -- what a
desilike calculator knows about itself (its compiled graph, its pytree state) lives in
:class:`CalculatorEmulator`, not in anything the user has to construct.
"""

import pickle

import numpy as np

from cosmoprimo.emulators.tools import Emulator as _Emulator, Space as BaseSpace

from desilike.base import Calculator
from desilike.parameter import VariableCollection


def _import(path):
    """``'module.Name'`` back to the class."""
    import importlib

    module, name = path.rsplit('.', 1)
    return getattr(importlib.import_module(module), name)


class Space(BaseSpace):
    """:class:`~cosmoprimo.emulators.tools.Space`, with a desilike calculator as a default.

        Space(theory)                       # extents from each Parameter's `ref`
        Space(theory, params=['h', 'n_s'])  # a subset
        Space(limits={'h': (0.6, 0.8)})     # or the base class's own forms

    ``ref`` is used rather than ``prior`` because the region an emulator must cover is where the
    chain lives, not where the prior allows. Over-covering is the most expensive mistake
    available: shrinking a prior-width box to a posterior-sized one was worth 23x at fixed node
    count. A chain or a covariance beats both -- ``Space(samples=chain)``.
    """
    def __init__(self, calculator=None, params=None, nsigma=3., **kwargs):
        """The box comes from each parameter's ``ref``, never from ``prior``.

        A prior says what is ALLOWED, not where the chain lives, and desilike's defaults are
        deliberately generous -- h in [0.1, 10], m_ncdm up to 5 eV.  Emulating over that box does
        not merely waste nodes: it asks the Boltzmann code for cosmologies it refuses, and the
        training dies at a node rather than at the call that set it up.
        """
        if calculator is None:
            super().__init__(params=params, nsigma=nsigma, **kwargs)
            return
        from desilike.base import get_params

        # `get_params`, not `build`: it builds the dependency graph and stops, where `build` would
        # also run every node's __post_init__ and __call__ -- a whole pipeline evaluation just to
        # read parameter metadata.
        #
        # VARIED only: a fixed parameter has a `ref` like any other, and emulating one is a silent
        # error -- the grid spends nodes on an axis the calculator never moves along, and can ask
        # for values it rejects (a fixed m_ncdm once produced a node at -8e-08).  `select` also
        # drops bare `Variable`s, which carry no `varied` attribute, so everything surviving it is
        # a `Parameter`: `ref` is always set (it falls back to a copy of the prior) and
        # `ParameterPrior.limits` is always a pair.
        selected = get_params(calculator).select(varied=True, derived=False)
        if params is not None:
            selected = [param for param in selected if param.name in params]
        limits, missing = {}, []
        for param in selected:
            if np.isfinite(param.ref.limits).all():
                limits[param.name] = tuple(float(value) for value in param.ref.limits)
            else:
                missing.append(param.name)
        if missing:
            raise ValueError(
                f'{type(calculator).__name__} parameters {missing} are varied but have no finite '
                f'`ref` limits. Give the Space explicit limits, a covariance or '
                f'samples, or fix the parameters you do not want emulated.')
        if not limits:
            raise ValueError(f'no varied parameter of {type(calculator).__name__} has a `ref`; '
                             f'nothing to emulate')
        super().__init__(limits=limits, nsigma=nsigma, **kwargs)


def _leafname(path):
    """A readable name for a pytree path: its keys joined by '.', e.g. ``'1.tt'``.

    JAX spells the two kinds of step differently -- a dict step carries ``key``, a sequence step
    carries ``idx`` -- which is all the getattr dance below is doing.

    '.' rather than '/' because these names are the keys of the values dict, and that dict is
    written to HDF5, where '/' is the group separator and so cannot appear in a name.  Nothing
    parses them back apart -- the pytree structure travels as a pickled ``PyTreeDef`` -- so they
    are labels, and a dict key containing a '.' of its own costs nothing.
    """
    parts = []
    for entry in path:
        key = getattr(entry, 'key', None)
        parts.append(str(key if key is not None else getattr(entry, 'idx', entry)))
    return '.'.join(parts)


#: Prefix marking an emulated DERIVED parameter, as opposed to a pytree child. Subclasses that
#: override ``transform`` must carry these through untouched -- they are not part of the state.
DERIVED = 'derived.'


class CalculatorEmulator(_Emulator):
    """Emulate a desilike calculator's pytree state, and get a calculator back.

    The state, not the return value: a calculator used as a dependency is read through its
    attributes -- a theory reads ``pt.pktable``, not whatever ``pt()`` returns -- so only the
    flattened state can stand in for it. That is what makes the result a drop-in for
    ``replace(likelihood, pt, emulated_pt)``.
    """
    def __init__(self, calculator, space, **options):
        from desilike.base import build

        import jax.numpy as jnp

        self.calculator = calculator
        # The class outlives the instance: a Calculator is not state, but `to_calculator` only
        # ever needed the class, and that is a name a file can carry.
        self._calculator_cls = type(calculator)
        # Flatten the children through the pytree, not one level: a calculator may return a
        # dict child -- CosmoprimoCosmology's first child is its parameter dict -- and
        # `jnp.asarray` on that is `dtype object is not a valid JAX array type`.
        import jax

        def output(calculator=calculator):
            leaves = jax.tree_util.tree_leaves(calculator.tree_flatten()[0])
            return tuple(jnp.asarray(leaf) for leaf in leaves)

        self.graph = build(calculator, output=output)
        # `build` runs every node's `__call__` to trace the graph, so the calculator carries real
        # state right here.  Everything reconstruction needs is therefore known at construction --
        # no "has it run yet" flag, and no recovery pass for a training restored entirely from a
        # checkpoint, which never calls `compute` at all.
        children, self.aux = calculator.tree_flatten()
        self.children_treedef = jax.tree_util.tree_structure(children)
        # A name for each leaf: a routing subclass keys its scalings off these, and an index
        # would silently point at a different quantity the moment a requirement is added.
        self.set_children_leafnames()
        leaves = jax.tree_util.tree_flatten(children)[0]
        if len(self.children_leafnames) != len(leaves):
            raise ValueError(
                f'{type(self).__name__}.set_children_leafnames gives '
                f'{len(self.children_leafnames)} '
                f'names for the {len(leaves)} flattened children of '
                f'{type(calculator).__name__}')
        # The derived names likewise: the graph returns `{p.name: p._value for p in params if
        # p.derived}` (base.py), so they are a property of the graph, not of having run it.
        self.derived_names = [param.name for param in self.graph.params if param.derived]
        super().__init__(self.compute, space, **options)
        # Every Parameter the pipeline exposes, kept once.  The emulated calculator has to HOLD
        # the ones it is emulated over, so that `build_graph` rediscovers them after the template
        # that declared them is pruned out of the emulated pipeline -- and a Parameter is state,
        # unlike the calculator that declared it, so these survive a write.
        self.graph_params = VariableCollection(self.graph.params)
        missing = [name for name in self.space.params if name not in self.graph_params]
        if missing:
            raise ValueError(f'parameters {missing} are not in the calculator graph')

    def set_children_leafnames(self):
        """Build ``self.children_leafnames``: a name for each flattened child, in order.

        The pytree path by default -- ``'harmonic.lensed_cl/tt'`` for a calculator that flattens
        to a dict, but a bare index for one that flattens to a sequence, which says nothing.  A
        routing subclass that knows its calculator's layout names them properly: the names are
        how the EMULATOR keys its state, so they are its business and not the calculator's.
        """
        import jax

        children = self.calculator.tree_flatten()[0]
        self.children_leafnames = [
            _leafname(path) for path, _ in jax.tree_util.tree_flatten_with_path(children)[0]]

    def compute(self, params):
        # The graph itself rejects a name the calculator does not expose (a typo would otherwise
        # train a perfectly good emulator of the wrong function).
        params = dict(params)
        # `return_derived` as well: a derived parameter is an output of the pipeline (sigma8 from
        # a cosmology, say), and anything downstream that reads one off an emulated calculator
        # would otherwise get its construction-time default for ever.
        children, derived = self.graph(params, return_derived=True)
        # keyed by the leaf's own PATH ('1.tt'), not by position: an index silently points at a
        # different quantity the moment a requirement is added, and a dump of the file is then
        # unreadable besides
        values = {name: np.asarray(child)
                  for name, child in zip(self.children_leafnames, children)}
        values.update({f'{DERIVED}{name}': np.asarray(derived[name])
                       for name in self.derived_names})
        return values

    def __getstate__(self):
        state = super().__getstate__()
        state['derived_names'] = list(self.derived_names)
        # the pytree aux: `to_calculator` rebuilds the state from it, and a routed subclass reads
        # its table layout out of it at every prediction -- so it has to survive the round trip
        state['aux'] = self.aux
        # The leaf structure, pickled into bytes.  A PyTreeDef is a live JAX object with no HDF5
        # form, but it pickles in a couple of hundred bytes and is the only thing that restores a
        # pytree EXACTLY -- container types included, which reassembling from the leaf paths
        # cannot do (a tuple child would come back as a dict).  The cost is a `pickle.loads` when
        # reading a file, so read emulators you trust.
        state['children_treedef'] = (np.frombuffer(pickle.dumps(self.children_treedef), dtype='u1')
                            if self.children_treedef is not None else np.zeros(0, dtype='u1'))
        state['children_leafnames'] = list(self.children_leafnames)
        # What `to_calculator` needs and cannot get from a Calculator, which is not itself state:
        # the class the emulated one subclasses, and the parameter nodes it holds so that
        # `build_graph` rediscovers them.  What the calculator was CONSTRUCTED with is not here --
        # a template, a k grid, are the caller's, and `to_calculator` takes them as arguments.
        state['calculator_cls'] = (f'{self._calculator_cls.__module__}.'
                                   f'{self._calculator_cls.__name__}')
        state['graph_params'] = self.graph_params.__getstate__()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.derived_names = list(state['derived_names'])
        self.aux = state['aux']
        blob = np.asarray(state.get('children_treedef', np.zeros(0, dtype='u1')))
        self.children_treedef = pickle.loads(blob.tobytes()) if blob.size else None
        self.children_leafnames = list(state.get('children_leafnames', []))
        self._calculator_cls = _import(state['calculator_cls'])
        self.graph_params = VariableCollection.__new__(VariableCollection)
        self.graph_params.__setstate__(state['graph_params'])
        # no Calculator in a saved emulator, and none is needed: `to_calculator` works from the
        # class above and the arguments its caller passes
        self.calculator, self.graph = None, None

    def emulator_namespace(self):
        """Extra methods and attributes for the emulated calculator's class.

        Empty by default. A subclass whose routing reaches beyond the pytree state -- one that
        rescales run-time nuisance parameters, say -- puts the overrides here rather than needing
        its own ``to_calculator``.
        """
        return {}

    def to_calculator(self, *args, **kwargs):
        """An instance of the original calculator's class, whose state is predicted, not computed.

        With no arguments, the ones the calculator was constructed with -- while this emulator
        still has it.  A saved one does not, because a Calculator is not state, and those
        arguments are the caller's anyway::

            emulated_pt = Emulator.read('pt.h5').to_calculator(template=my_template)

        Only ``__call__`` is overridden: ``__init__`` stays the root's own, so the emulated object
        holds the same Parameter/Variable nodes and the graph discovers them exactly as before.

        ``theory.update(pt=emulated)`` does not work on a constructed calculator: desilike allows
        ``update()`` only during construction. Use ``replace()`` and recompile.
        """
        from cosmoprimo.emulators.tools import NotTrained

        if not self.trained:
            raise NotTrained('call train() first')
        import jax

        root_cls, aux, predict = self._calculator_cls, self.aux, self.predict
        children_leafnames = list(self.children_leafnames)
        children_treedef = self.children_treedef
        nodes = {name: self.graph_params[name] for name in self.space.params}
        derived_nodes = {name: self.graph_params[name] for name in self.derived_names
                         if name in self.graph_params}

        def make_init():

            def __init__(self, *args, **kwargs):
                # explicit, not zero-arg `super()`: this is defined outside a class body, so
                # there is no __class__ cell for it to look up
                root_cls.__init__(self, *args, **kwargs)
                self.emulator_derived = dict(derived_nodes)
                # Hold the parameter nodes as an attribute of this object. `build_graph`
                # discovers Nodes nested in dicts, so the compiled graph threads values into
                # whatever copy of the calculator it evaluates -- and it is that copy's nodes
                # that carry the current values. Reading a dict captured at fit time reads
                # nodes belonging to a different copy, whose `.value` never moves off its
                # construction default: measured, an emulated pt then returned the fiducial
                # spectrum for every parameter it was asked about.
                self.emulator_params = dict(nodes)

            return __init__

        def make_call():

            def __call__(self):
                # NOTE: returns `self`, not whatever the root's __call__ returned. The emulated
                # object is meant to be a dependency -- a parent reads its attributes and ignores
                # the return value -- and the root's return value is not recoverable from the
                # emulated state. As a pipeline root it therefore behaves differently.
                # Read the parameter nodes, not `getattr(self, name)`: on a real theory the
                # cosmological parameters live on a sub-calculator (the template), not as
                # attributes of the theory, so an attribute lookup finds nothing.
                values = {name: node.value for name, node in self.emulator_params.items()}
                predicted = predict(**values)
                # count the children rather than subtracting the derived: the two need not be
                # in step if a derived name is not a graph parameter
                leaves = [predicted[name] for name in children_leafnames]
                children = jax.tree_util.tree_unflatten(children_treedef, leaves)
                rebuilt = root_cls.tree_unflatten(aux, children)
                for key, value in rebuilt.__dict__.items():
                    setattr(self, key, value)
                # in place, so the pipeline's existing reference to the node stays valid
                for name, node in self.emulator_derived.items():
                    node.value = predicted[f'{DERIVED}{name}']
                return self

            return __call__

        # built with `type` rather than a class statement so a subclass can add methods: the h
        # routing, for instance, has to override `combine_bias_terms_spectrum2_poles` as well
        namespace = {'__init__': make_init(), '__call__': make_call()}
        namespace.update(self.emulator_namespace())
        EmulatedCalculator = type(f'Emulated{root_cls.__name__}', (root_cls,), namespace)
        EmulatedCalculator.__qualname__ = EmulatedCalculator.__name__
        if not (args or kwargs) and getattr(self, 'calculator', None) is not None:
            args, kwargs = self.calculator._init
        return EmulatedCalculator(*args, **kwargs)


#: Keywords :func:`emulate` forwards to `train` rather than to the emulator it builds.
_TRAIN_OPTIONS = ('budget', 'checkpoint', 'chunk', 'batch_size', 'mpicomm')


class _PackedRoot(Calculator):
    """A structural root over several calculators: it computes nothing itself.

    Its point is that one compiled graph covers all of them, so anything they share upstream --
    a template, a cosmology -- is evaluated once per node instead of once per calculator. That
    sharing is the whole reason to pack rather than emulate each separately; the node set is
    shared too, but the node set is cheap.
    """
    def __init__(self, calculators):
        self.calculators = list(calculators)

    def __call__(self):
        # Read the dependencies. A Calculator dependency that is never read during __call__ is
        # pruned from the graph, and every parameter it exposes goes with it -- measured, the
        # packed graph came out with no parameters at all and every node failed as "does not
        # expose 'h'". Gathering the children is the read.
        self.children = [child for calculator in self.calculators
                         for child in calculator.tree_flatten()[0]]
        return None

    def tree_flatten(self):
        children, parts = [], []
        for calculator in self.calculators:
            own, part_aux = calculator.tree_flatten()
            own = list(own)
            parts.append({'aux': part_aux, 'n': len(own), 'cls': type(calculator)})
            children += own
        return children, {'parts': parts}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.calculators = []
        index = 0
        for part in aux['parts']:
            obj.calculators.append(
                part['cls'].tree_unflatten(part['aux'], children[index:index + part['n']]))
            index += part['n']
        return obj


class PackedCalculatorEmulator(CalculatorEmulator):
    """Emulate several calculators at once, as one emulator.

        emu = Emulator([theory1.pt, theory2.pt], space)
        emu.train(budget=3)
        pt1, pt2 = emu.to_calculator()

    Worth doing whenever the calculators share upstream work: one compiled graph evaluates a
    shared template or cosmology once per node rather than once per calculator, and the Boltzmann
    call is the entire cost. Emulating each separately pays for it twice.
    """
    def __init__(self, calculators, space, **options):
        self.parts = list(calculators)
        super().__init__(_PackedRoot(self.parts), space, **options)
        self._parts_cls = [type(part) for part in self.parts]

    def __getstate__(self):
        state = super().__getstate__()
        # one class per part: `to_calculator` deploys them, not the packed root
        state['parts_cls'] = [f'{cls.__module__}.{cls.__name__}' for cls in self._parts_cls]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._parts_cls = [_import(path) for path in state['parts_cls']]
        self.parts = None

    def to_calculator(self, *parts_args):
        """The deployed calculators, in the order they were packed.

        With no arguments, each part is constructed as it was -- while this emulator still holds
        them.  A saved one does not, so pass one dict of constructor arguments per part.

        One emulated calculator per part, each reconstructing from its own slice of the packed
        prediction -- not the packed root, whose ``.calculators`` are still the originals it was
        constructed with.

        The parts predict independently, so the interpolant runs once per part at evaluation
        time. That is deliberate: what packing saves is the training cost -- one shared upstream
        evaluation per node instead of one per calculator -- and a predicted spline is cheap. A
        one-entry memo keeps repeated identical calls from paying even that.
        """
        from cosmoprimo.emulators.tools import NotTrained

        if not self.trained:
            raise NotTrained('call train() first')
        nodes = {name: self.graph_params[name] for name in self.space.params}
        emulator = self
        children_leafnames = list(self.children_leafnames)

        def part_calculator(root_cls, init, spec, offset):
            """One emulated calculator over its slice of the packed prediction.

            A function called per part, NOT the loop body: ``__call__`` closes over ``root_cls``,
            ``aux`` and the leaf slice, and runs later -- so defining the class in the loop would
            hand every part the last iteration's values.  Silently, and only once more than one
            calculator is packed.
            """
            aux, count = spec['aux'], spec['n']
            leafnames = children_leafnames[offset:offset + count]

            class EmulatedPart(root_cls):

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.emulator_params = dict(nodes)

                def __call__(self):
                    values = {name: node.value
                              for name, node in self.emulator_params.items()}
                    predicted = emulator._predict_cached(values)
                    children = [predicted[name] for name in leafnames]
                    rebuilt = root_cls.tree_unflatten(aux, children)
                    for key, value in rebuilt.__dict__.items():
                        setattr(self, key, value)
                    return self

            EmulatedPart.__name__ = f'Emulated{root_cls.__name__}'
            EmulatedPart.__qualname__ = EmulatedPart.__name__
            args, kwargs = init
            return EmulatedPart(*args, **kwargs)

        classes = self._parts_cls
        if parts_args:
            if len(parts_args) != len(classes):
                raise ValueError(f'{len(parts_args)} sets of arguments for {len(classes)} parts')
            inits = [((), dict(item)) for item in parts_args]
        elif getattr(self, 'parts', None) is not None:
            inits = [part._init for part in self.parts]
        else:
            raise RuntimeError(
                f'a saved emulator carries no calculators, so the {len(classes)} parts cannot be '
                f'constructed on their own: pass one dict of constructor arguments per part, in '
                f'the order they were packed.')

        built, offset = [], 0
        for root_cls, init, spec in zip(classes, inits, self.aux['parts']):
            built.append(part_calculator(root_cls, init, spec, offset))
            offset += spec['n']
        return built

    def _predict_cached(self, values):
        """The packed prediction, memoised on the last parameter values.

        Every part asks for the same prediction; without this each would recompute all of them.
        """
        key = tuple(sorted((name, float(np.asarray(value))) for name, value in values.items()))
        cached = getattr(self, '_predict_memo', None)
        if cached is None or cached[0] != key:
            self._predict_memo = (key, self.predict(**values))
        return self._predict_memo[1]


def Emulator(calculator, space=None, params=None, cls=None, **options):
    """Build an emulator of a desilike calculator. It is not trained.

        pt  = FOLPSPTSpectrum2Poles(template=template)
        emu = Emulator(pt, Space(pt), budget=3)
        emu.nodes(budget=3)                          # size the run before paying for it
        emu.train(checkpoint='pt.npz', chunk='30min')

        theory = FOLPSTracerSpectrum2Poles(pt=emu.to_calculator(), template=template)
        replace(likelihood, pt, emu.to_calculator()); build(likelihood)   # or swap it in

    Parameters
    ----------
    calculator : Calculator, list
        Emulated directly -- no wrapper to construct. A list is packed into one emulator whose
        ``to_calculator()`` returns the deployed calculators in order: worth it when they share
        upstream work, since one graph then evaluates it once per node instead of once each.
    space : Space, default=None
        Where accuracy is required. Derived from the calculator's own parameter ``ref`` limits
        when omitted -- a convenience, not a recommendation: a chain or a covariance is worth
        orders of magnitude more (whitening beat plain ranges 350x in the median at equal node
        count).
    params : list, default=None
        Restrict to these parameters; the rest keep their current values.
    cls : type, default=None
        The :class:`~cosmoprimo.emulators.tools.Emulator` subclass to build. By default the
        calculator is asked -- ``calculator.get_emulator_cls()`` --
        so a theory that knows something exact about itself is emulated correctly without the
        caller having to know which subclass to import; :class:`CalculatorEmulator` when it
        declares nothing. Pass it explicitly to override, including to force the generic one.

    Other keyword arguments (``engine``, ``budget``, ``coverage``, ...) go to it.

    Returns
    -------
    emulator : CalculatorEmulator
        UNtrained. Call :meth:`~cosmoprimo.emulators.tools.Emulator.train`, then
        :meth:`~CalculatorEmulator.to_calculator`.
    """
    if isinstance(calculator, (list, tuple)):
        # several calculators: one graph over all of them, so shared upstream work is done once
        calculators = list(calculator)
        if not calculators:
            raise ValueError('no calculator to emulate')
        if space is None:
            space = Space(_PackedRoot(calculators), params=params)
        elif params is not None:
            space = space.marginal(params)
        return (cls if cls is not None else PackedCalculatorEmulator)(calculators, space,
                                                                     **options)
    if space is None:
        space = Space(calculator, params=params)
    elif params is not None:
        space = space.marginal(params)
    if cls is None:
        # The calculator is asked which subclass it wants: a theory that knows something exact
        # about itself declares it once in `get_emulator_cls` and every caller gets it, rather
        # than each having to know which subclass to import.  Asked on the INSTANCE, not the
        # class, so a pt whose answer depends on how it was configured -- a FOLPS pt with
        # output='monomials' bakes in what the direct-output one routes -- can dispatch on that.
        cls = calculator.get_emulator_cls() or CalculatorEmulator
    return cls(calculator, space, **options)



#: Read a trained emulator back, of whatever subclass wrote it (`from_state` rebuilds
#: the class named in the file).  Hung off the factory because that is the entry point
#: users already have: `Emulator` is a function, so `Emulator.read` would not otherwise
#: exist.
Emulator.read = CalculatorEmulator.read


def emulate(calculator, space=None, params=None, cls=None, **options):
    """Build and train, in one call.

        emulated_pt = emulate(pt, Space(pt), budget=3).to_calculator()

    The same as :func:`Emulator` followed by ``train``, for when the run is small enough that you
    do not need to size it first. For anything expensive prefer the two steps, and pass
    ``checkpoint`` and ``chunk``: a kill then costs one node rather than the training.

    Returns
    -------
    emulator : CalculatorEmulator
        trained. Call :meth:`~CalculatorEmulator.to_calculator` for a drop-in calculator.
    """
    training = {name: options.pop(name) for name in _TRAIN_OPTIONS if name in options}
    if 'engine' in options:
        training['engine'] = options['engine']
    return Emulator(calculator, space=space, params=params, cls=cls, **options).train(**training)
