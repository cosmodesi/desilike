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

from desilike.base import get_params
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
        # built on first use in `compute`, and kept: tracing a FOLPS-sized graph is not free
        self._graph_jitted = None
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
        # jitted, always: a FOLPS-sized graph is thousands of small kernels and the eager
        # per-node dispatch is a large share of the node cost (same reason _compile_scalars jits).
        # Nothing but the graph's own outputs is read here, which is what makes that safe -- the
        # graph evaluates a COPY of the calculator, so anything a subclass reads back off the
        # original afterwards is the previous node's value. A quantity the routing needs and the
        # pytree does not already carry therefore has to become a child of it, as
        # `FOLPSPTSpectrum2Poles._anchors` does, and not an attribute read after this call.
        if self._graph_jitted is None:
            import jax

            self._graph_jitted = jax.jit(lambda p: self.graph(p, return_derived=True))
        children, derived = self._graph_jitted(params)
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
        # ... and "survive" means EXACTLY. An aux can hold live objects with no HDF5 form:
        # CosmoprimoCosmology's carries its derived `Parameter`s and its requirement specs, and
        # those came back from a plain HDF5 round trip as bare floats, so `tree_unflatten` then
        # ran `param._value = val` on a numpy.float64 and a cached CMB emulator could not be
        # deployed at all. Pickle it, exactly as `children_treedef` above is pickled and for the
        # same reason; `state['aux']` stays for readability and for older files.
        state['aux_pickle'] = np.frombuffer(pickle.dumps(self.aux), dtype='u1')
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
        # Prefer the pickled aux (exact); fall back to the plain one for files written before
        # it existed, which is also what a subclass overriding `aux` on write ends up with.
        aux_blob = np.asarray(state.get('aux_pickle', np.zeros(0, dtype='u1')))
        self.aux = pickle.loads(aux_blob.tobytes()) if aux_blob.size else state['aux']
        blob = np.asarray(state.get('children_treedef', np.zeros(0, dtype='u1')))
        self.children_treedef = pickle.loads(blob.tobytes()) if blob.size else None
        self.children_leafnames = list(state.get('children_leafnames', []))
        self._calculator_cls = _import(state['calculator_cls'])
        self.graph_params = VariableCollection.__new__(VariableCollection)
        self.graph_params.__setstate__(state['graph_params'])
        # no Calculator in a saved emulator, and none is needed: `to_calculator` works from the
        # class above and the arguments its caller passes
        self.calculator, self.graph, self._graph_jitted = None, None, None

    def emulator_namespace(self):
        """Extra methods and attributes for the emulated calculator's class.

        Empty by default. A subclass whose routing reaches beyond the pytree state -- one that
        rescales run-time nuisance parameters, say -- puts the overrides here rather than needing
        its own ``to_calculator``.
        """
        return {}

    def to_calculator(self, *args, calculator=None, center=True, **kwargs):
        """An instance of the original calculator's class, whose state is predicted, not computed.

        With no arguments, the ones the calculator was constructed with -- while this emulator
        still has it.  A saved one does not, because a Calculator is not state, and those
        arguments are the caller's anyway::

            emulated_pt = Emulator.read('pt.h5').to_calculator(template=my_template)

        Only ``__call__`` is overridden: ``__init__`` stays the root's own, so the emulated object
        holds the same Parameter/Variable nodes and the graph discovers them exactly as before.

        ``theory.update(pt=emulated)`` does not work on a constructed calculator: desilike allows
        ``update()`` only during construction. Use ``replace()`` and recompile.

        Parameters
        ----------
        calculator : Calculator, default=None
            The calculator to deploy from, which an emulator read back from a file does not have.
            It supplies the constructor arguments, and it is kept, because prediction can need the
            calculator itself and not only its class: ``CMBEmulator`` reads ``m_ncdm``, ``N_ur``
            and ``T_cmb`` off it whenever they are not varied. Pass the one wired into the
            pipeline; that is the right one by construction. A live emulator already holds its
            own, and this replaces it.

            One argument rather than assigning ``emulator.calculator`` and then repeating
            ``*calculator._init`` at the call: one place, and no way for the two to disagree.
        center : bool or dict, default=True
            Deploy at the centre of the trained region, rather than wherever the parameters
            happen to point. They rarely point anywhere usable: training leaves the last node's
            values behind, and a box measured from someone else's posterior need not contain the
            constructor's defaults at all -- measured once at 4 sigma below a box's lower edge,
            which killed every rank. Outside the region the emulator refuses, and it is asked as
            soon as the pipeline compiles, so the default is to move rather than to fail.

            ``True`` uses :attr:`Space.center`; a mapping deploys at those values instead;
            ``False`` leaves every parameter as it is.
        *args, **kwargs
            Constructor arguments, overriding those of *calculator*.
        """
        from cosmoprimo.emulators.tools import NotTrained

        if not self.trained:
            raise NotTrained('call train() first')
        import jax

        emulator = self
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
                # Back-reference to the emulator, shared by every instance this method
                # returns: a downstream consumer that needs the object the predictions come
                # from reaches it here. `predict` above is a bound method of the same object.
                self._emulator = emulator

            return __init__

        def make_call():

            def __call__(self):
                # NOTE: returns `self`, not whatever the root's __call__ returned. The emulated
                # object is meant to be a dependency -- a parent reads its attributes and ignores
                # the return value -- and the root's return value is not recoverable from the
                # emulated state. As a pipeline root it therefore behaves differently.
                # The emulator does the predicting, so it needs to know whether the enclosing
                # graph is traced -- an emulator whose transform calls a non-jax library (the
                # scalars provider's analytic core calls cosmoprimo) must raise eagerly and
                # return NaN when traced, exactly as `CosmoprimoCosmology` does.
                emulator._is_tracing = getattr(self, '_is_tracing', False)
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
        def make_post_init():

            def __post_init__(self, *args, **kwargs):
                post_init = getattr(root_cls, '__post_init__', None)
                if post_init is not None:
                    post_init(self, *args, **kwargs)
                # An emulated calculator is pure JAX
                self._is_external = False

            return __post_init__

        namespace = {'__init__': make_init(), '__call__': make_call(),
                     '__post_init__': make_post_init()}
        namespace.update(self.emulator_namespace())
        EmulatedCalculator = type(f'Emulated{root_cls.__name__}', (root_cls,), namespace)
        EmulatedCalculator.__qualname__ = EmulatedCalculator.__name__
        if calculator is not None:
            # kept, not just read for its `_init`: `predict` above is a bound method of this
            # emulator, so whatever it looks up on `self.calculator` at evaluation time has to
            # find the pipeline's own calculator here.
            self.calculator = calculator
        if not (args or kwargs) and getattr(self, 'calculator', None) is not None:
            args, kwargs = self.calculator._init
        deployed = EmulatedCalculator(*args, **kwargs)
        if center is not None and center is not False:
            # On the deployed object, not on the emulator's own nodes: `build` resolves a
            # parameter from the calculator it evaluates, so moving anything else is a no-op that
            # looks like it worked.
            values = self.space.center if center is True else dict(center)
            params = get_params(deployed)
            for name, value in values.items():
                if name in params:
                    params[name].update(value=value)
        return deployed


#: Keywords :func:`emulate` forwards to `train` rather than to the emulator it builds.
_TRAIN_OPTIONS = ('budget', 'checkpoint', 'chunk', 'batch_size', 'mpicomm')


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
    calculator : Calculator
        Emulated directly -- no wrapper to construct.
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
        # Packing several calculators into one graph was removed (2026-09-01): the pack forced
        # the generic expansion on every part, throwing away each calculator's own routed
        # emulator class -- 561 nodes over 7 parameters against 121 over 4 for w0waCDM. A pack
        # that defers to `get_emulator_cls` per part would be worth having; this one was not.
        raise TypeError('emulate one calculator at a time: packing was removed, and one '
                        'emulator per calculator keeps each one its routed class')
    if space is None:
        space = Space(calculator, params=params)
    elif params is not None:
        space = space.marginal(params)
    if cls is None:
        # The calculator is asked which subclass it wants: a theory that knows something exact
        # about itself declares it once in `get_emulator_cls` and every caller gets it, rather
        # than each having to know which subclass to import.  Asked on the INSTANCE, not the
        # class, so a pt whose answer depends on how it was configured can dispatch on that.
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
