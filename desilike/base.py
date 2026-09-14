"""
JAX-friendly calculator pipeline for desilike.

Base class:
- Calculator: JAX-native by default (_is_external = False).
  Set _is_external = True on a subclass (or per-instance) to switch to
  arbitrary Python/numpy mode: __call__() is wrapped via pure_callback +
  custom_jvp (finite-difference backward pass) so that jit/vmap/grad all work.

Lifecycle:
- Calculator(*args, **kwargs) saves args and runs __init__ (inside a construction context).
  __init__ defines and updates every node (Parameters + Calculator deps + dep.update()), fixing
  node identity at construction (enabling replace()/share_params() and cheap construction).
- build() runs __post_init__ on each node in dependency order (then __call__). __post_init__
  is non-node setup only (numpy/scalars, non-Node helpers); it never creates nodes or calls update().
- _trace_graph/CompiledGraph discover dependencies by scanning instance attributes for Node objects
  set in __init__ (before __post_init__ runs).

__call__() interface:
- The pipeline sets param attributes and update dep on self before invoking __call__().
- __call__() reads own params and dep outputs directly from self (e.g. self.A, self.cosmo.growth_factor).
- __call__() sets named output attributes on self (e.g. self.growth_factor).
- __call__() may return any value (including self); that value
  is forwarded as the pipeline output if this node is the root. tree_flatten()
  defines all __call__ products that may be useful to downstream nodes, and not aimed to be derived as Variable.

tree_flatten / tree_unflatten:
- Each calculator must define tree_flatten(self) -> (children, aux) and
  tree_unflatten(cls, aux, children) -> instance. children are the output
  arrays produced by __call__(). The framework uses these to pass outputs between
  calculators.

Pipeline:
- CompiledGraph: builds a static DAG once, exposes a pure
  __call__(params) compatible with jax.jit / jax.vmap / jax.grad.
"""

import difflib
import functools
import os
import weakref
import logging
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from cosmoprimo.emulators.tools.utils import (fd_stencil as _fd_stencil, interpolation_weights,
                                              chebyshev_values, chebyshev_lobatto_nodes,
                                              chebyshev_vandermonde_inverse, nested_level_nodes,
                                              smolyak_combination, TRANSFORMS as _FD_TRANSFORMS)
from collections import defaultdict

jax.config.update('jax_enable_x64', True)

from .parameter import (Node, Variable, Parameter, VariableCollection, _compile_context,
                        _CompileContext, _iter_node, _replace_node)
from .distributed import default_mpicomm, get_mpicomm, gather as _mpi_gather


# ── base classes ──────────────────────────────────────────────────────────────

class Calculator(Node):
    """
    Base class for calculators implemented with JAX ops.

    Subclasses define:
      __init__(*args, **kwargs): both define and update all nodes here — create every
        Variable/Parameter and Calculator dependency as a public (non-underscore) attribute
        (self.b1 = Parameter(...), self.pt = pt) and call any dep.update(...). These attributes
        (incl. Nodes nested in list/tuple/dict) are auto-discovered as dependencies.
      __post_init__(*args, **kwargs): non-node setup only — numpy/scalar config and non-Node
        helper objects. May read what __init__ set; must NOT create Parameters or Calculator deps.
        __post_init__ may be called more than once (e.g. when build() is re-run). Any derived
        quantity that is computed from a raw input (e.g. precision from covariance) must be
        re-derived from the original value each time. Store the raw input under a private name
        in __init__ (e.g. self._precision) and read it in __post_init__ rather than modifying
        the already-derived value in place.
      __call__(self): read params via self.param.value and dep outputs via self.dep.attr;
        compute and store output attributes; return the output value (array, tuple, None, or self).
      tree_flatten(self) -> (children, aux): children = list of output arrays,
        aux = static data needed by tree_unflatten.
      tree_unflatten(cls, aux, children) -> instance: reconstruct an instance
        carrying only the output attrs (no dep refs, no init args).

    __init__ runs at construction (saving args and wiring all nodes); __post_init__ runs at
    build() in dependency order, then __call__(). _trace_graph scans attributes for Nodes
    (set in __init__) to discover dependencies before __post_init__ runs.
    """

    _is_calculator = True
    #: Nesting depth of the running ``__init__`` (see ``__init_subclass__``); 0 outside one.
    _init_depth = 0
    # Whether this node is evaluated as a non-JAX (numpy/arbitrary-Python) calculator,
    # wrapped via pure_callback + finite-difference JVP. Read off the *instance* at build
    # time, so a subclass may toggle it per-instance (e.g. a cosmology wrapper that is
    # JAX-traceable for some engines and external for others). Defaults to False (pure JAX).
    _is_external = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # Node.__init_subclass__ registers the pytree
        cls.logger = logging.getLogger(cls.__name__)
        # `__init__` and `__post_init__` are handed the same arguments -- the ones this object was
        # constructed with. A class whose `__post_init__` (its own, or the one it inherits) cannot
        # accept its `__init__`'s arguments would fail at build time, deep inside a build, on
        # whichever pipeline happens to use it. Checked here instead, when the class is defined:
        # an import-time error naming both methods.
        if '__init__' in cls.__dict__:
            import inspect
            owner = next((klass for klass in cls.__mro__ if '__post_init__' in klass.__dict__), None)
            if owner is not None:
                post = inspect.signature(owner.__dict__['__post_init__']).parameters
                init = inspect.signature(cls.__dict__['__init__']).parameters
                if not any(param.kind is param.VAR_KEYWORD for param in post.values()):
                    missing = [name for name in init if name not in post]
                    if missing:
                        raise TypeError(
                            f'{cls.__name__}.__init__ takes {missing}, which the __post_init__ it '
                            f'{"defines" if owner is cls else f"inherits from {owner.__name__}"} does not '
                            f'accept -- and the two are handed the same arguments. Give it a matching '
                            f'signature, or `*args, **kwargs` when it reads none of them.')
                required = [name for name, param in list(post.items())[1:]
                            if param.default is param.empty
                            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                            and name not in init]
                if required:
                    raise TypeError(
                        f'the __post_init__ {cls.__name__} inherits from {owner.__name__} requires '
                        f'{required}, which {cls.__name__}.__init__ does not take. Define a '
                        f'__post_init__ matching this class\'s own signature.')
        if '__init__' in cls.__dict__:
            _orig_init = cls.__dict__['__init__']
            @functools.wraps(_orig_init)
            def _wrapped_init(self, *args, _f=_orig_init, **kwargs):
                # How was this object built?  The outermost call answers that, and it is what
                # re-running the constructor needs (`copy`, or a build restarting the tree) and
                # what `__post_init__` is handed.  Recording at every level instead would leave
                # `_init` holding the arguments a subclass passed to `super().__init__(...)`, and
                # rebuilding from those raises `unexpected keyword argument` on the subclass's
                # own signature.
                depth = self._init_depth
                self._init_depth = depth + 1
                if depth == 0:
                    self._init = (args, kwargs)
                try:
                    _f(self, *args, **kwargs)
                finally:
                    self._init_depth = depth
            cls.__init__ = _wrapped_init

    def __init__(self, *args, **kwargs):
        # No custom __init__: nothing to wire at construction; __post_init__ runs at build().
        self._init = (args, kwargs)

    def update(self, *args, **kwargs):
        """Re-initialize in-place with overridden arguments; new kwargs override old ones.

        Meant for construction: a parent configuring the child it is handed (``theory.update(k=...)``
        from an observable's ``__init__``).  The child may already belong to a graph -- a built
        theory handed to a second observable is the ordinary way to reuse it -- and re-running
        its ``__init__`` swaps its dependencies under that graph, which would keep evaluating
        the old ones while the node reads the new (a template that created a fresh cosmology
        this way raised a ``KeyError`` two steps later, at the first point its cache did not
        cover).  So the node's claim is released here, and the graph that held it refuses to
        run from now on; build the new root and use that, or reconfigure :func:`copy` of the
        tree to keep both.
        """
        # Reconfiguring goes through -- configuring a calculator you were handed is how every
        # theory sizes its template, and swapping an emulator in is `update()` too. What it costs
        # is the graphs built over this node: they would go on holding the dependencies just
        # replaced, so they are invalidated here, and say so (with this call site) when next used.
        _invalidate_owners(self, f'{type(self).__name__} was reconfigured')
        old_args, old_kwargs = self._init
        merged_args = args if args else old_args
        merged_kwargs = {**old_kwargs, **kwargs}
        self.__init__(*merged_args, **merged_kwargs)

    def clone(self, **kwargs):
        """Return a new instance of the same type with updated keyword arguments.

        The existing constructor arguments are reused as-is; keyword overrides
        in *kwargs* replace matching keys.  Pass freshly constructed nodes in
        *kwargs* when independent objects are required.

        Returns
        -------
        A new instance of ``type(self)``.

        Examples
        --------
        Override a single argument while keeping the rest::

            spec2 = spec.clone(arg=...)
        """
        old_args, old_kwargs = self._init
        return type(self)(*old_args, **{**old_kwargs, **kwargs})

    def __post_init__(self, *args, **kwargs):
        pass

    @classmethod
    def get_emulator_cls(cls):
        """Return the :class:`~cosmoprimo.emulators.tools.Emulator` subclass to emulate this
        calculator with, or ``None`` for the generic one.

        A classmethod here, but :func:`desilike.emulators.Emulator` asks the instance, so a
        calculator whose answer depends on how it was configured may override this as a plain
        method -- the FOLPS pts do, to dispatch on ``output``.

        :func:`desilike.emulators.Emulator` consults this automatically, so a calculator that
        knows something exact about itself -- FOLPSD's background-scalar routing, say -- declares
        it once here and every caller gets it, without having to know which subclass to import.

        ``None`` (the default) lets the emulator use its generic
        :class:`~desilike.emulators.CalculatorEmulator`, which expands every varied parameter and
        reconstructs state via ``tree_unflatten``.

        A subclass may return a dedicated class instead — declaring, through the emulated-
        calculator protocol (``get_emulator_params``, ``flatten_root``,
        ``__init__(emulator, **kwargs)``, ``_init_graph``), which quantities are emulated
        coefficients, which parameters are routed exactly at run time instead of expanded, and
        which extra node dependencies (passed through ``to_calculator(**kwargs)``) that routing
        needs.  See ``FOLPSPTSpectrum2Poles.get_emulator_cls`` for the motivating case.
        """
        return None

    def __call__(self):
        raise NotImplementedError

    def tree_flatten(self):
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux, children):
        raise NotImplementedError


class Likelihood(Calculator):
    """
    Base class for likelihood calculators.

    Subclasses implement __post_init__() and __call__(). __call__() must set self.logpdf.
    tree_flatten/tree_unflatten are provided here; subclasses need not repeat them.
    """
    @property
    def ndata(self):
        return None

    #: A likelihood may expose ``flattheory``, its model vector at the current parameters, as an
    #: attribute or a property.  A Gaussian one sets it in ``__call__`` anyway; one that hands its
    #: parameters to an external code can compute it on access.  Whatever needs the model rather
    #: than the likelihood -- generating synthetic data, above all -- reads it, and an arm that is
    #: not Gaussian in any data vector simply does not have it.

    def tree_flatten(self):
        return [self.logpdf], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf = children[0]
        return obj


class GaussianLikelihood(Likelihood):
    """
    Base class for Gaussian chi-squared likelihoods.

    Subclasses must implement:
      __init__(): set self.flatdata (the observations) and self.precision (2D array, C⁻¹),
        along with any Calculator deps and Variable/Parameter instances.  Make the observations
        a ``Variable`` -- ``self.flatdata = Variable(f'{type(self).__name__}.flatdata',
        value=jnp.asarray(x))`` -- so they are a node the graph fills, and can be overridden by
        name (an Asimov vector, a mock realisation) instead of by assigning an attribute a
        traced graph would ignore.  It must not be a ``Parameter``: ``input``, ``varied`` and
        ``solved`` are Parameter properties, so a plain ``Variable`` is skipped by
        ``select(varied=True)`` and, being ``derived=False``, is written to no chain.
        A plain array still works: the arithmetic below is the same either way.
      __call__(): set self.flattheory (1D JAX array), then call super().__call__() to compute
        self.logpdf = -½ (flatdata - flattheory)ᵀ precision (flatdata - flattheory).

    tree_flatten exposes [logpdf, flattheory, precision] so downstream nodes can
    access them as dep outputs.
    """

    @property
    def ndata(self):
        return self.flatdata.size

    def __call__(self):
        r = self.flatdata - self.flattheory
        self.logpdf = -0.5 * r @ self.precision @ r
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf, self.flattheory, self.precision], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf, obj.flattheory, obj.precision = children
        return obj


class SumLikelihood(Likelihood):
    """
    Sums logpdf from multiple Likelihood components.

    __init__(*likelihoods): each argument must be a Likelihood instance.
    __call__(): sums logpdf across components (deps are called before this node in the pipeline).
    """

    def __init__(self, *likelihoods):
        # Nodes (the Likelihood dependencies) live in __init__.
        # Accept a single list/tuple argument: SumLikelihood([l1, l2]) == SumLikelihood(l1, l2).
        if len(likelihoods) == 1 and isinstance(likelihoods[0], (list, tuple)):
            likelihoods = likelihoods[0]
        self.likelihoods = list(likelihoods)

    @property
    def ndata(self):
        ndata = [getattr(like, 'ndata', None) for like in self.likelihoods]
        if all(n is not None for n in ndata):
            return sum(ndata)
        return None

    def __call__(self):
        self.logpdf = sum(like.logpdf for like in self.likelihoods)
        return self.logpdf


def _collect_likelihood_components(likelihood):
    """Walk a ``(Sum)Likelihood`` tree and split into Gaussian and non-Gaussian leaves.

    Parameters
    ----------
    likelihood : Likelihood
        Root likelihood node (may be a :class:`SumLikelihood`).

    Returns
    -------
    gaussians : list of GaussianLikelihood
    non_gaussians : list of Likelihood
        Leaf components that are not :class:`GaussianLikelihood` instances.
    """
    if isinstance(likelihood, SumLikelihood):
        gaussians, non_gaussians = [], []
        for component in likelihood.likelihoods:
            g, ng = _collect_likelihood_components(component)
            gaussians.extend(g)
            non_gaussians.extend(ng)
        return gaussians, non_gaussians
    if isinstance(likelihood, GaussianLikelihood):
        return [likelihood], []
    return [], [likelihood]


class Prior(Calculator):
    """
    Sums log-prior probabilities over non-fixed parameters.

    __post_init__(*args, **kwargs) collects Parameter arguments into self.params (a list).
    Positional args may be VariableCollection instances; keyword args are individual
    Parameters. Fixed parameters are silently skipped (they contribute 0).

    __call__() returns a scalar: sum of param.prior.logpdf(param) over non-fixed params.
    Returns -inf when any parameter is outside its prior support.

    To include a Prior in a pipeline it must be a dependency of the root node
    (directly or transitively). See :class:`Posterior`.
    """

    def __init__(self, *args, **kwargs):
        # Nodes (the collected Parameters) live in __init__.
        params = []
        for arg in args:
            if isinstance(arg, VariableCollection):
                for p in arg:
                    params.append(p)
            else:
                params.append(arg)
        for p in kwargs.values():
            params.append(p)
        # Parameters only.  `Prior(get_params(likelihood))` is on the critical path of every
        # run and hands over every Variable the graph has, including the likelihoods' data
        # vectors, which have neither a prior nor a `fixed` flag to read.  A plain Variable
        # contributes zero to the log-prior by definition, so dropping it here is the whole of
        # its treatment.
        self.params = VariableCollection([p for p in params if isinstance(p, Parameter)])

    def __call__(self):
        logprior = jnp.zeros(())
        for p in self.params:
            if (not p.fixed) and (not p.solved):
                # Sum over all elements for vector params (independent joint prior).
                logprior = logprior + jnp.sum(p.prior.logpdf(p))
        self.logpdf = logprior
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf = children[0]
        return obj


def _restrict(params, graph):
    """The subset of *params* that *graph* actually has.

    Internal plumbing routinely holds a superset: the whole likelihood's parameters handed to the
    prior sub-graph, to a marginalization group, to one component of a sum.  The graph rejects
    names it does not have (:meth:`CompiledGraph._check_names`), so these sites filter first --
    a superset is legitimate here, a typo is not, and only the caller can tell them apart.
    """
    return {name: value for name, value in params.items() if name in graph.params}


def _transitive_param_names(node: Calculator, pipe: 'CompiledGraph') -> set:
    """Return the set of param names that *node*'s subgraph transitively depends on within *pipe*."""
    visited_ids = set()
    stack = [node]
    param_names = set()
    while stack:
        n = stack.pop()
        nid = id(n)
        if nid in visited_ids:
            continue
        visited_ids.add(nid)
        for p in pipe._node_var_deps.get(nid, []):
            param_names.add(p.name)
        for dep in pipe._node_calc_deps.get(nid, []):
            stack.append(dep)
    return param_names


class SolvedBlock(NamedTuple):
    """One independent block of the marginalisation plan, before any graph is wired to it."""
    alpha_names: object
    alpha_sizes: object
    alpha_shapes: object
    gaussians: object       # the likelihood nodes this block's theory pipe must output
    components: object      # GaussianComponents, alpha_idx in this block's DOF numbering
    marg_local: object
    best_local: object
    prior_precision: object
    prior_center: object
    stage_i_ids: object


def plan_marginalisation(components, solved_params, stage_i_ids_of):
    """Partition the solved parameters into blocks that must be solved together, and re-index
    each component against the block that owns it.  Returns a list of :class:`SolvedBlock`.

    Two solved parameters belong together exactly when some Gaussian component depends on both,
    so union-find over the components gives the blocks, and each is then solved on its own,
    n_g x n_g.  Everything after that is bookkeeping: expressing each component's dependence in
    its block's own degree-of-freedom numbering, where a vector parameter contributes one DOF
    per element.

    It builds no graph and touches no compiled state -- that is the point of it being here
    rather than in ``Posterior.__init__``, which then only wires pipes to a plan it did not
    compute.

    Parameters
    ----------
    components : list of GaussianComponent
        Only those that depend on a solved parameter, ``alpha_idx`` indexing *solved_params*.
    solved_params : list of Parameter
        The likelihood's solved parameters, in the order ``alpha_idx`` refers to.
    stage_i_ids_of : callable
        ``{name} -> frozenset`` of node ids that do not depend on those parameters, so are
        evaluated once ahead of the linearisation rather than differentiated through.
    """
    parent = list(range(len(solved_params)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for comp in components:
        for i in range(1, len(comp.alpha_idx)):
            union(comp.alpha_idx[0], comp.alpha_idx[i])

    # Collect and sort global alpha indices per group root.
    root_globals = defaultdict(set)
    for comp in components:
        root_globals[find(comp.alpha_idx[0])].update(comp.alpha_idx)
    root_sorted = {root: sorted(indices) for root, indices in root_globals.items()}

    # Remap each component's alpha_idx to local (group-relative) indices.
    root_comps = defaultdict(list)
    for comp in components:
        global_idx = root_sorted[find(comp.alpha_idx[0])]
        g_to_l = {gi: li for li, gi in enumerate(global_idx)}
        root_comps[find(comp.alpha_idx[0])].append(
            comp._replace(alpha_idx=[g_to_l[g] for g in comp.alpha_idx]))

    # Per solved param: DOF count (np.prod(shape), 1 for a scalar), prior inverse-scale (0 when
    # improper) and prior center.  The prior is treated as independent per element, the same 1-D
    # ParameterPrior applied to each DOF.
    sizes = [int(np.prod(p.shape)) if p.shape else 1 for p in solved_params]
    shapes = [p.shape for p in solved_params]
    inv_scales, centers = [], []
    for p in solved_params:
        std = p.prior.std() if p.prior is not None else None
        inv_scales.append((1. / std) if (std is not None and np.isfinite(std)) else 0.)
        centers.append(float(p.prior.center()) if p.prior is not None else 0.)
    marg_global = {i for i, p in enumerate(solved_params) if p.derived == 'marg'}
    best_global = {i for i, p in enumerate(solved_params) if p.derived == 'best'}

    blocks = []
    for root, global_idx in root_sorted.items():
        comps = root_comps[root]
        alpha_sizes = [sizes[g] for g in global_idx]

        # DOF offset within the group for each local param index j.
        dof_offsets, offset = [], 0
        for g in global_idx:
            dof_offsets.append(offset)
            offset += sizes[g]

        def dofs_of(selected):
            """DOF indices of the selected global params, in this block's numbering."""
            return np.array([dof_offsets[j] + k for j, g in enumerate(global_idx)
                             if g in selected for k in range(sizes[g])], dtype=int)

        blocks.append(SolvedBlock(
            alpha_names=[solved_params[g].name for g in global_idx],
            alpha_sizes=alpha_sizes,
            alpha_shapes=[shapes[g] for g in global_idx],
            gaussians=[comp.likelihood for comp in comps],
            # param-level local indices -> DOF-level, so that a solved param of shape (2,)
            # contributes two columns
            components=[comp._replace(alpha_idx=[dof_offsets[j] + k for j in comp.alpha_idx
                                                 for k in range(alpha_sizes[j])])
                        for comp in comps],
            marg_local=dofs_of(marg_global),
            best_local=dofs_of(best_global),
            # the same value repeated for every DOF of a param
            prior_precision=jnp.array([inv_scales[g] ** 2 for g in global_idx for _ in range(sizes[g])]),
            prior_center=jnp.array([centers[g] for g in global_idx for _ in range(sizes[g])]),
            stage_i_ids=stage_i_ids_of({solved_params[g].name for g in global_idx})))
    return blocks


class GaussianComponent(NamedTuple):
    """One Gaussian leaf of a likelihood, with what analytic marginalisation needs of it.

    ``alpha_idx`` is re-indexed as the plan narrows: solved-parameter indices over the whole
    likelihood first, then group-local parameter indices, then group-local DOF indices (a
    vector parameter contributing one per element).  Each step is a ``_replace``.
    """
    likelihood: object
    theory: object          # CompiledGraph over its flattheory; None once a group owns it
    precision: object
    flatdata: object
    alpha_idx: object


class ComponentBlock(NamedTuple):
    """A component's slice of a group's concatenated theory vector."""
    precision: object
    flatdata: object
    dof_idx: object
    offset: int
    size: int


class MarginalisationGroup(NamedTuple):
    """Solved parameters that must be solved together, and the machinery to do it.

    Two parameters share a group when some component depends on both, so the blocks are
    independent and each is solved on its own.
    """
    alpha_names: object
    alpha_sizes: object
    alpha_shapes: object
    theory_pipe: object
    components: object      # list of ComponentBlock
    marg_local: object
    best_local: object
    prior_precision: object
    prior_center: object
    stage_i_pipe: object
    stage_i_ids: object


class Posterior(Calculator):
    """
    Log-posterior = log-likelihood + log-prior, with optional analytic treatment of solved params.

    __post_init__(likelihood, prior) compiles both into private internal pipelines. The
    likelihood's parameters are surfaced via inputs() so the outer compiled pipeline
    manages them uniformly.

    Solved parameters (derived='marg' or derived='best') are handled analytically for
    GaussianLikelihood / SumLikelihood theories that are linear in those params.

    derived='marg': Gaussian marginalization — the parameter is integrated out with its
      prior (param.prior.std() required). Produces the correct Bayesian evidence including
      the volume factor  + ½ log|P_α| − ½ log|F_eff|.

    derived='best': profile likelihood — the parameter is set to its best-fit (MLE) value.
      No prior required; no volume factor included.

    Mixed 'marg'/'best' is supported: the Schur-complement formula is used so the 'best'
    block contributes no volume correction.

    Marginalization is performed per Gaussian component: jacfwd is only computed for
    components whose theory actually depends on the solved parameters.

    __call__() evaluates the prior first (for non-solved params). In eager mode the
    likelihood is skipped when logprior == -inf. Under JAX tracing the full computation
    is always built.
    """

    def __init__(self, likelihood, prior=None):
        # Declarations only.  Every graph this class needs is built in `__post_init__`, which is
        # where the lifecycle puts setup -- and here that is not a style point: a constructor that
        # builds cannot survive a rebuild of the tree it is part of.  `_init_graph` re-runs each
        # `__init__` and `_bind_variables` then rewires the tree back to the caller's Parameter
        # objects, but it cannot reach inside a CompiledGraph an `__init__` had already built, so
        # that graph is left holding Parameters nobody else references.  Nothing restores them,
        # and a traced call left a JVPTracer on the derived one, which the next eager call
        # tripped over (measured).  Building here instead, after every `__init__` has run, the
        # graphs only ever see the settled tree.
        params = get_params(likelihood)
        self._likelihood_node = likelihood
        self._solved_params = params.select(solved=True)

        if prior is None:
            prior = Prior(params)
        prior.update(params.select(solved=False) if self._solved_params else params)
        self._prior_calculator = prior
        _prior_params = list(get_params(prior))
        self._prior_param_names = [p.name for p in _prior_params]

        # Public (scanned by _trace_graph) so non-solved likelihood params are in Posterior's
        # deps and get their values set before each __call__.  Solved params are excluded here
        # because they are exposed separately via self.solved_params below.
        self.likelihood_params = [p for p in params if not getattr(p, 'solved', False)]
        # Expose originals (derived='marg'/'best') as a public attribute so _trace_graph
        # discovers them as Posterior's deps and the pipeline tracks their best-fit values.
        self.solved_params = list(self._solved_params)
        # Derived outputs exposed to the pipeline (their .value is set in __call__).
        self.logposterior = Variable(basename='logposterior', value=0., derived=True, latex=r'\ln\mathcal{P}')
        self.logprior = Variable(basename='logprior', value=0., derived=True, latex=r'\ln\Pi')
        self.loglikelihood = Variable(basename='loglikelihood', value=0., derived=True, latex=r'\ln\mathcal{L}')
        # Number of data points (None when the likelihood does not expose it), surfaced
        # for ndof bookkeeping downstream (e.g. the profiler / Profiles.to_stats).
        self.ndata = getattr(likelihood, 'ndata', None)

    def __post_init__(self, likelihood, prior=None):
        likelihood, prior = self._likelihood_node, self._prior_calculator
        # One build; every CompiledGraph below (self._likelihood, per-component no-alpha pipes,
        # group_theory_pipes) is a view over its context.  Views share one claim on the nodes,
        # where a second build would supersede the first and leave it stale.
        _likelihood_ctx = _build_graph(likelihood)
        self._likelihood = CompiledGraph(likelihood, _likelihood_ctx)
        self._solved_params = self._likelihood.params.select(solved=True)

        # Public (scanned by _trace_graph) so non-solved likelihood params are in Posterior's
        # deps and get their values set before each __call__.  Solved params are excluded here
        # because they are exposed separately via self.solved_params below.
        self.likelihood_params = [p for p in self._likelihood.params if not getattr(p, 'solved', False)]

        if self._solved_params:
            gaussians, non_gaussians = _collect_likelihood_components(likelihood)
            if not gaussians:
                raise ValueError(
                    'Analytic marginalization requires at least one GaussianLikelihood '
                    '(or a SumLikelihood containing one) whose theory depends on the '
                    'solved parameters.'
                )

            alpha_names_set = set(p.name for p in self._solved_params)
            non_gaussian_comps = []
            for ng in non_gaussians:
                # A view over the one context rather than a fresh build, which would reconfigure
                # `ng`'s nodes under `self._likelihood`. A view's `.params` span the whole context,
                # so the dependence check reads `ng`'s own transitive parameters instead.
                ng_pipe = CompiledGraph(ng, _likelihood_ctx)
                bad = alpha_names_set & _transitive_param_names(ng, self._likelihood)
                if bad:
                    raise ValueError(
                        f'Non-Gaussian likelihood component depends on solved '
                        f'parameter(s) {sorted(bad)!r}; analytic marginalization '
                        f'requires that such components are GaussianLikelihood.'
                    )
                non_gaussian_comps.append(ng_pipe)
            self._non_gaussian_comps = non_gaussian_comps

            # Build one record per Gaussian component.  For alpha-dependent ones the theory pipe
            # is not used later (the group pipe covers them); their alpha params are discovered by
            # subgraph BFS over self._likelihood without triggering another build.  For the rest
            # the pipe is indeed used for evaluation, as a view over the one context so that
            # __post_init__ does not run again and reconfigure the nodes under self._likelihood.
            components = []
            for g in gaussians:
                comp_param_names = _transitive_param_names(g, self._likelihood)
                alpha_idx = [i for i, p in enumerate(self._solved_params) if p.name in comp_param_names]
                theory = None if alpha_idx else CompiledGraph(g, _likelihood_ctx, output=lambda g=g: g.flattheory)
                components.append(GaussianComponent(g, theory, g.precision, g.flatdata, alpha_idx))

            self._no_alpha_components = [comp for comp in components if not comp.alpha_idx]

            # Two-stage partition, computed before any group pipe exists so that the globally
            # stage-i nodes can be found.  Stage i = nodes whose transitive param deps have zero
            # overlap with the block's solved params; stage ii = the rest.  Only stage ii is traced
            # through jax.linearize, so stage-i external nodes (cosmo, PT emulators) are never
            # differentiated through.  Every group pipe shares this graph, so its
            # _node_var_deps / _node_calc_deps are self._likelihood's.
            def _compute_stage_i_ids(group_alpha_names_set):
                alpha_dep_ids = set()
                for pipe_node in self._likelihood.nodes:
                    direct_alpha = {p.name for p in self._likelihood._node_var_deps[id(pipe_node)]} & group_alpha_names_set
                    transitive_alpha = {id(dep) for dep in self._likelihood._node_calc_deps[id(pipe_node)]} & alpha_dep_ids
                    if direct_alpha or transitive_alpha:
                        alpha_dep_ids.add(id(pipe_node))
                return frozenset(id(pipe_node) for pipe_node in self._likelihood.nodes if id(pipe_node) not in alpha_dep_ids)

            # The plan itself: which solved params must be solved together, and each component's
            # place in its block.  Pure, and computed before a single pipe is built.
            pending_groups = plan_marginalisation(
                [comp for comp in components if comp.alpha_idx], list(self._solved_params),
                _compute_stage_i_ids)

            # Phase B: build a shared node_state dict for nodes that are stage-i in every group.
            # When multiple CompiledGraph instances share the same node_state object for a
            # globally-stage-i external node (e.g. CosmoprimoCosmology), the pure_callback
            # cache hit from the first group's pre-pass is visible to all other groups and to
            # self._likelihood's extra derived-params pass — CAMB runs once, not n_groups+1 times.
            global_stage_i_ids = frozenset.intersection(*[block.stage_i_ids for block in pending_groups]) if pending_groups else frozenset()
            shared_node_states = {
                id(node): {'last_params': None, 'was_called': False, 'last_result': None,
                           'dep_result': None, 'call_result': None, 'last_dep_args': None}
                for node in self._likelihood.nodes
                if id(node) in global_stage_i_ids and node._is_external
            }

            # Rebuild self._likelihood so its _fn_dep closures capture the shared node_state
            # dicts (the original self._likelihood used private dicts from before Phase B).
            if shared_node_states:
                self._likelihood = CompiledGraph(likelihood, _likelihood_ctx, shared_node_states=shared_node_states)

            def make_group_output(gaussians):
                return lambda: jnp.concatenate([jnp.ravel(jnp.asarray(g.flattheory)) for g in gaussians])

            # Phase C: create group CompiledGraphs with the shared node_states injected.
            self._groups = []
            for block in pending_groups:
                stage_i_ids = block.stage_i_ids
                group_theory_pipe = CompiledGraph(likelihood, _likelihood_ctx,
                                                  output=make_group_output(block.gaussians),
                                                  shared_node_states=shared_node_states)

                # Per-component metadata for splitting the concatenated theories/Jacobians.
                comp_meta = []
                data_offset = 0
                for comp in block.components:
                    flat_data = np.ravel(np.asarray(comp.flatdata))
                    comp_meta.append(ComponentBlock(comp.precision, flat_data, comp.alpha_idx,
                                                    data_offset, flat_data.size))
                    data_offset += flat_data.size

                stage_i_nodes_ordered = [pipe_node for pipe_node in group_theory_pipe.nodes
                                         if id(pipe_node) in stage_i_ids]

                # Build Stage-i pre-pass: runs every Stage-i node via group_theory_pipe with
                # Stage-ii nodes skipped (skip_ids), captures every Stage-i node's
                # tree_flatten leaves as the return value (output_override).  This fixes a
                # prior bug where build(stage_i_root, ...) only ran the sub-graph reachable
                # from the last Stage-i node in topo order, leaving earlier-branch Stage-i
                # nodes (e.g. pt_LRG / pt_ELG in a 3-tracer pipeline) with stale
                # build-time values in stage_i_flat.
                if stage_i_nodes_ordered:
                    stage_ii_ids = frozenset(id(pipe_node) for pipe_node in group_theory_pipe.nodes
                                             if id(pipe_node) not in stage_i_ids)

                    def make_stage_i_output(ordered_nodes):
                        return lambda: tuple(
                            leaf
                            for stage_i_node in ordered_nodes
                            for leaf in jax.tree_util.tree_leaves(stage_i_node.tree_flatten()[0])
                        )

                    stage_i_output_fn = make_stage_i_output(stage_i_nodes_ordered)

                    def make_stage_i_prepass(pipe, s2_ids, out_fn):
                        def fn(params):
                            ret, _, _ = pipe._run_graph_fn(
                                params, skip_ids=s2_ids, output_override=out_fn)
                            return ret
                        return fn

                    stage_i_pipe = make_stage_i_prepass(
                        group_theory_pipe, stage_ii_ids, stage_i_output_fn)
                else:
                    stage_i_pipe = None

                self._groups.append(MarginalisationGroup(
                    block.alpha_names, block.alpha_sizes, block.alpha_shapes, group_theory_pipe,
                    comp_meta, block.marg_local, block.best_local, block.prior_precision,
                    block.prior_center, stage_i_pipe, stage_i_ids))


        _prior_params = [p for p in get_params(prior)]
        _prior_ref = prior
        self._prior = build(prior, output=lambda: (_prior_ref.logpdf, [p.value for p in _prior_params]))
        self._prior_param_names = [p.name for p in _prior_params]

    def _marg_loglik(self, params):
        """Profile/marginalize over solved params, one independent group at a time.

        Returns
        -------
        logL : jax array
        solved_values : dict mapping each solved-param name to its best-fit value
        """
        logL = jnp.zeros(())
        solved_values = {}

        # Non-Gaussian components: add logpdf directly (they do not depend on solved params).
        for ng_pipe in self._non_gaussian_comps:
            ng_params = {p.name: jnp.asarray(params[p.name]) for p in ng_pipe.params}
            logL = logL + ng_pipe(ng_params)

        # Gaussian components with no solved-param dependence: standard chi-squared.
        for comp in self._no_alpha_components:
            comp_params = {p.name: jnp.asarray(params[p.name]) for p in comp.theory.params}
            r = comp.flatdata - comp.theory(comp_params)
            logL = logL - 0.5 * r @ (comp.precision @ r)

        # Per-group: independent block solve of size n_g × n_g.
        for group in self._groups:
            (group_alpha_names, group_alpha_sizes, group_alpha_shapes, group_theory_pipe,
             comp_meta, marg_local, best_local, prior_prec, prior_center,
             stage_i_pipe, stage_i_ids) = group
            # n_g: total DOF across all alpha params in this group (sum of per-param sizes).
            n_g = sum(group_alpha_sizes)

            # All params needed by the combined group pipe (includes both alpha and non-alpha params).
            group_params = {p.name: jnp.asarray(params[p.name]) for p in group_theory_pipe.params}
            # Flatten+concatenate alpha values → shape (n_g,); scalars (shape=()) become size-1 slices.
            alpha_vec = jnp.concatenate([jnp.ravel(jnp.asarray(group_params[name])) for name in group_alpha_names])

            def _unpack_alpha(alpha_vec, _names=group_alpha_names, _sizes=group_alpha_sizes,
                              _shapes=group_alpha_shapes, _params=group_params):
                """Reconstruct param dict from flat alpha_vec, respecting each param's shape."""
                p = dict(_params)
                offset = 0
                for name, size, shape in zip(_names, _sizes, _shapes):
                    p[name] = alpha_vec[offset:offset + size].reshape(shape if shape else ())
                    offset += size
                return p

            if stage_i_pipe is not None and stage_i_ids:
                # Two-stage JVP optimisation:
                # Stage i — run every non-alpha node once (via group_theory_pipe with Stage-ii
                # nodes skipped), capture every Stage-i node's tree_flatten leaves as a flat
                # tuple of live JAX values.  XLA computes this block once; none of it appears
                # inside the JVP binary.
                stage_i_flat = stage_i_pipe(group_params)

                # Stage ii — thin function that skips Stage i nodes (outputs injected from
                # stage_i_flat) and runs only alpha-dependent nodes (Tracer + downstream).
                # Called directly via _run_graph_fn, bypassing @custom_jvp, so JAX's native
                # forward-mode AD traces through Stage ii only.  stage_i_flat is a closed-over
                # constant: XLA reuses it without rematerialisation.
                def thin_group_fn(alpha_vec,
                                   _pipe=group_theory_pipe, _unpack=_unpack_alpha,
                                   _s1_ids=stage_i_ids, _s1_flat=stage_i_flat):
                    return_val, _, _ = _pipe._run_graph_fn(_unpack(alpha_vec),
                                                           stage_i_ids=_s1_ids, stage_i_flat=_s1_flat)
                    return return_val

                theories_concat, jvp_fn = jax.linearize(thin_group_fn, alpha_vec)
            else:
                # Fallback: no Stage i nodes identified, linearize the full pipeline.
                def group_fn(alpha_vec, _pipe=group_theory_pipe, _unpack=_unpack_alpha):
                    return _pipe(_unpack(alpha_vec))

                theories_concat, jvp_fn = jax.linearize(group_fn, alpha_vec)

            # jvp_fn(e_j) = j-th Jacobian column; vmap over identity → shape (n_g, total_n_data).
            B_rows = jax.vmap(jvp_fn)(jnp.eye(n_g))

            F_g = jnp.zeros((n_g, n_g))
            b_g = jnp.zeros(n_g)
            logL_g = jnp.zeros(())

            for block in comp_meta:
                data_offset, n_i, precision = block.offset, block.size, block.precision
                theory_i = theories_concat[data_offset:data_offset + n_i]
                # B_rows[dof_j, data_k] = Jacobian; transpose to (n_i, n_g), select local DOF columns.
                B_i = B_rows[:, data_offset:data_offset + n_i].T[:, block.dof_idx]  # (n_i, n_local_dof)
                r_i = block.flatdata - theory_i
                BtP = B_i.T @ precision
                ix = np.array(block.dof_idx)
                F_g = F_g.at[ix[:, None], ix[None, :]].add(BtP @ B_i)
                b_g = b_g.at[ix].add(BtP @ r_i)
                logL_g = logL_g - 0.5 * r_i @ (precision @ r_i)

            # Add prior precision for every solved DOF in the group.
            F_g = F_g + jnp.diag(prior_prec)

            # With a non-zero prior center μ the solve RHS gains a P·μ term, and the
            # log-posterior at α₀=0 gains the prior chi2 −½ μᵀP μ.
            b_tilde = b_g + prior_prec * prior_center
            delta_alpha = jnp.linalg.solve(F_g, b_tilde)
            logL = logL + logL_g - 0.5 * jnp.dot(prior_prec * prior_center, prior_center) + 0.5 * b_tilde @ delta_alpha

            # Store absolute best-fit values (linearisation point α₀=0 + delta).
            dof_off = 0
            for name, size, shape in zip(group_alpha_names, group_alpha_sizes, group_alpha_shapes):
                chunk = delta_alpha[dof_off:dof_off + size]
                solved_values[name] = jnp.asarray(params[name]) + chunk.reshape(shape if shape else ())
                dof_off += size

            # Volume factor: only 'marg' DOFs contribute; the 'best' block is profiled
            # out via the Schur complement  + ½ log|P_marg| − ½ log|F_g| + ½ log|F_g[best, best]|.
            # Empty marg/best index sets contribute 0, so no special-casing is needed.
            logdet_Pmarg = 0.  # jnp.sum(jnp.log(prior_prec[marg_local])) — omitted: prior_prec can be 0 (improper prior)
            _, logdet_F = jnp.linalg.slogdet(F_g)
            _, logdet_F_bb = jnp.linalg.slogdet(F_g[best_local[:, None], best_local[None, :]])
            logL = logL + 0.5 * (logdet_Pmarg - logdet_F + logdet_F_bb)

        return logL, solved_values

    def __call__(self):
        # Reset solved params to zero so _marg_loglik always starts from a neutral
        # linearisation point.  The Schur-complement formula is exact for linear theories
        # regardless of starting point, so this has no effect on the log-likelihood value,
        # but it prevents numerical drift when p._value was left at a previous best-fit.
        for sp in self._solved_params:
            sp._value = jnp.zeros(sp.shape) if sp.shape else jnp.zeros(())
        params = {p.name: p.value for p in self._likelihood.params}
        logprior, reparam_vals = self._prior(_restrict(params, self._prior))
        params = {**params, **dict(zip(self._prior_param_names, reparam_vals))}
        is_tracing = isinstance(logprior, jax.core.Tracer)
        if is_tracing or not bool(jnp.isneginf(logprior)):
            if self._solved_params:
                loglik, solved_values = self._marg_loglik(params)
                params = {**params, **solved_values}
                if not is_tracing:
                    for var in self.solved_params:
                        var.value = solved_values[var.name]
                # Extra standard pass to re-capture derived params: the inner
                # _run_graph restores p._value after each call; requesting
                # return_derived=True gives back the traced values so the outer
                # _run_graph reads them correctly.
                _, inner_derived = self._likelihood(_restrict(params, self._likelihood), return_derived=True)
            else:
                loglik, inner_derived = self._likelihood(_restrict(params, self._likelihood), return_derived=True)
            for p in self._likelihood._derived_params:
                p._value = inner_derived[p.name]
            logpdf = logprior + loglik
            self.logpdf = jnp.where(jnp.isnan(logpdf), -jnp.inf, logpdf)
        else:
            loglik = jnp.full((), -jnp.inf)
            self.logpdf = jnp.full((), -jnp.inf)
        # Expose derived outputs (captured by the pipeline as derived params).
        self.loglikelihood.value = loglik
        self.logprior.value = logprior
        self.logposterior.value = self.logpdf
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf = children[0]
        return obj


# _fd_stencil now lives in cosmoprimo.emulators.tools.utils (imported above).


def _jacfwd_dict_wrap(fn, names):
    """Lift *fn(p_dict) -> pytree* to one ``jax.jacfwd`` pass w.r.t. the dict of
    parameters *names*, all in a single trace.

    Follows ``jax.jacfwd`` pytree semantics on a dict input: each leaf of *fn*'s
    output is replaced by ``{name: d leaf / d name}``, with the parameter's axes
    appended after the leaf's own axes.  Nest k times for the k-th derivative;
    nesting appends one ``{name: ...}`` level per pass, outermost level first
    (``wrapped(p)[n1][n2]`` is ``d²/dn1 dn2`` with ``n1``'s axes before ``n2``'s,
    matching ``jax.hessian`` on a dict input)."""
    def wrapped(p_dict):
        values = {name: jnp.asarray(p_dict[name]) for name in names}
        return jax.jacfwd(lambda vals: fn({**p_dict, **vals}))(values)
    return wrapped


def _fd_parse_eps(fd_eps_val):
    """Parse an ``fd_eps`` value into ``(eps_below, eps_above, eps_avg)``.

    Accepts:
    - scalar float ``eps``:              both sides use ``eps``, average is ``eps``.
    - 2-tuple ``(eb, ea)``:              asymmetric steps (``ParameterFiniteDifference.eps``).
    - legacy 3-tuple ``(center, eb, ea)``: the ``center`` element (the expansion anchor,
                                          now ``param.fd.center``) is ignored here.
    """
    if fd_eps_val is None or (np.ndim(fd_eps_val) == 0 and not np.isfinite(fd_eps_val)):
        fd_eps_val = 1e-5
    if isinstance(fd_eps_val, (tuple, list)):
        eps_below, eps_above = (float(v) for v in fd_eps_val[-2:])
    else:
        eps_below = eps_above = float(fd_eps_val)
    return eps_below, eps_above, (eps_below + eps_above) * 0.5


# _FD_TRANSFORMS now lives in cosmoprimo.emulators.tools.utils (imported above).


#: Sentinel for "argument not given", where ``None`` is itself a meaningful value.
_UNSET = object()


class FDSpec(NamedTuple):
    """Everything :func:`_fd_direct_wrap` needs to build one parameter's stencil.

    Either a collocation spec (*nodes* set, *offsets* / *coeffs* / *eps* ``None``) or a
    finite-difference one (the reverse); *transform* applies to both.
    """
    offsets: object
    coeffs: object
    eps: object
    prior_limits: object
    transform: object
    nodes: object


def _fd_wrap(fn, name, spec, k, prior_limits=_UNSET):
    """:func:`_fd_direct_wrap` from an :class:`FDSpec`; *prior_limits* overrides the spec's."""
    return _fd_direct_wrap(fn, name, spec.offsets, spec.coeffs, spec.eps, k,
                           prior_limits=spec.prior_limits if prior_limits is _UNSET else prior_limits,
                           transform=spec.transform, nodes=spec.nodes)


def _fd_stencil_sum(fn, p_dict, name, p0, node_values, weights, divisor=None):
    """``sum_j w_j fn(p_dict with *name* at node j) / divisor``, over any pytree *fn* returns.

    The one shape every finite-difference stencil in this module has: evaluate at each node,
    weight, accumulate, divide.  What the three stencils differ in is only what they pass here
    -- fixed collocation nodes with interpolation weights and no divisor, a uniform grid with
    the static centered coefficients, or a boundary-shifted grid with weights solved at trace
    time -- so they share this.

    Parameters
    ----------
    p0 : array
        The parameter's current value; its ``ndim`` picks the scalar or the vectorised path.
    node_values : sequence
        One entry per node: the value *name* takes there.  Scalar *p0*: a scalar each.  Array
        *p0*: broadcastable to ``(p0.size,)``, the value an element takes when it is the one
        being perturbed -- every other element keeps its ``p0`` value.
    weights : array
        ``(n_nodes,)`` for a scalar *p0*, ``(p0.size, n_nodes)`` for an array one, so that each
        element may carry its own weights (they do, wherever the node grid depends on the point).
    divisor : float or None
        Divided out at the end; ``None`` to skip it.
    """
    n_nodes = len(node_values)

    if p0.ndim == 0:
        accumulated = None
        for node_idx in range(n_nodes):
            weight = weights[node_idx]
            contribution = jax.tree_util.tree_map(lambda x: weight * x,
                                                  fn({**p_dict, name: node_values[node_idx]}))
            if accumulated is None:
                accumulated = contribution
            else:
                accumulated = jax.tree_util.tree_map(lambda a, b: a + b, accumulated, contribution)
        if divisor is None:
            return accumulated
        return jax.tree_util.tree_map(lambda x: x / divisor, accumulated)

    # Array parameter: one vmapped pass per node over the one-hot basis directions, so the
    # Python loop runs over the stencil only.
    flat_size = p0.size
    p0_flat = p0.reshape(-1)
    basis = jnp.eye(flat_size)
    accumulated = None
    for node_idx in range(n_nodes):
        nodes_flat = jnp.broadcast_to(jnp.asarray(node_values[node_idx]), (flat_size,))

        def eval_along(one_hot, _nodes=nodes_flat):
            values = p0_flat * (1. - one_hot) + _nodes * one_hot
            return fn({**p_dict, name: values.reshape(p0.shape)})

        # values: pytree with each leaf of shape (flat_size, *leaf_shape)
        values = jax.vmap(eval_along)(basis)
        weight_column = weights[:, node_idx]
        scaled = jax.tree_util.tree_map(
            lambda x: x * weight_column.reshape((flat_size,) + (1,) * (x.ndim - 1)), values)
        if accumulated is None:
            accumulated = scaled
        else:
            accumulated = jax.tree_util.tree_map(lambda a, b: a + b, accumulated, scaled)

    if divisor is not None:
        accumulated = jax.tree_util.tree_map(lambda x: x / divisor, accumulated)

    # Move the leading batch axis (flat_size) to trailing axes matching the parameter's shape:
    # each leaf comes out of the vmap as (flat_size, *leaf_shape) and the caller wants
    # (*leaf_shape, *p0.shape).
    def _move_batch_to_param_axes(x):
        n_out = x.ndim - 1
        if n_out == 0:
            return x.reshape(p0.shape)
        moved = jnp.transpose(x, tuple(range(1, n_out + 1)) + (0,))   # (*leaf_shape, flat_size)
        return moved.reshape(moved.shape[:-1] + p0.shape)

    return jax.tree_util.tree_map(_move_batch_to_param_axes, accumulated)


def _fd_direct_wrap(fn, name, offsets, coeffs, eps, k, prior_limits=None, transform=None, nodes=None):
    """Lift *fn(p_dict) -> y* to the k-th order FD stencil derivative w.r.t. *name*.

    Uses the *direct* order-k stencil (k + acc - 1 evaluations, linear in k)
    rather than nesting k order-1 stencils (exponential cost).

    *fn* may return any JAX pytree (scalar, array, tuple, dict, …).  The
    derivative has the same pytree structure; each leaf's shape gains the
    trailing axes of *name* when *name* is array-valued.

    Scalar *name*: output shape of each leaf unchanged.
    Array *name* of shape S: each leaf gains trailing axes S; for order > 1
    these are the diagonal elements (cross-element terms are not computed).

    For array parameters the stencil is vectorised via ``jax.vmap`` over the
    basis directions, so the Python loop runs only over the stencil points.

    prior_limits : (lo, hi) or None
        Hard prior limits.  When set, the stencil *nodes* are shifted inward so
        all of them stay within ``[lo, hi]`` — but the derivative is still taken
        at the requested point: the uniform centered weights no longer apply on
        a shifted (or asymmetric-step) grid, so the weights are recomputed at
        trace time as the polynomial-interpolation (Fornberg) weights for the
        k-th derivative at ``p0``.  This restores desilike_bak's boundary
        behavior, where the earlier port silently returned the derivative at
        the shifted point instead.

    transform : str or None
        Name of an expansion-variable transform from ``_FD_TRANSFORMS``.  When
        set, the stencil is uniform in the transformed variable u (fd_eps in u
        units, prior limits mapped through the monotone forward map), function
        evaluations happen at the back-transformed parameter values, and the
        returned derivative is d^k/du^k.
    """
    import math
    fwd = inv = None
    if transform is not None:
        fwd, inv = _FD_TRANSFORMS[transform]

    if nodes is not None:
        # Chebyshev collocation: nodes fixed in transformed units, spanning the parameter's
        # fd.limits; the k-th derivative at the current point u(p0) is taken from the full
        # node set via polynomial-interpolation weights, so the order-n Taylor built from
        # these calls is identically the degree-n interpolant through the nodes.
        node_positions = np.asarray(nodes, dtype='f8')
        n_nodes = len(node_positions)
        if k >= n_nodes:
            raise ValueError(f'collocation for {name}: derivative order {k} needs more than {n_nodes} nodes')
        node_values = list(node_positions if inv is None else inv(node_positions))

        def wrapped(p_dict):
            p0 = jnp.asarray(p_dict[name])
            u0 = p0 if fwd is None else fwd(p0)
            # the same fixed nodes for every element; the weights are per element
            weights = interpolation_weights(node_positions, u0 if p0.ndim == 0 else u0.reshape(-1), k)
            return _fd_stencil_sum(fn, p_dict, name, p0, node_values, weights)

        return wrapped

    # Parse eps: scalar or (center, eps_below, eps_above).
    eps_below, eps_above, eps_avg = _fd_parse_eps(eps)
    h_k = eps_avg ** k

    # Resolve finite prior bounds as Python floats (None means unbounded on that side);
    # with a transform, in transformed units (forward map is monotone increasing).
    _prior_lo = _prior_hi = None
    if prior_limits is not None:
        _lo, _hi = prior_limits
        if np.isfinite(_lo): _prior_lo = float(_lo) if fwd is None else float(fwd(_lo))
        if np.isfinite(_hi): _prior_hi = float(_hi) if fwd is None else float(fwd(_hi))

    if _prior_lo is None and _prior_hi is None and eps_below == eps_above and transform is None:
        # Static path: symmetric steps, no boundary, no transform — the uniform centered
        # weights apply as-is and zero-weight nodes stay skipped.
        def wrapped(p_dict):
            p0 = jnp.asarray(p_dict[name])
            base = p0 if p0.ndim == 0 else p0.reshape(-1)
            node_values = [base + off * eps_avg for off in offsets]
            weights = jnp.asarray(coeffs)
            if p0.ndim:
                weights = jnp.broadcast_to(weights, (p0.size, len(coeffs)))
            return _fd_stencil_sum(fn, p_dict, name, p0, node_values, weights, h_k)

        return wrapped

    # Dynamic path: the node grid is shifted inside the prior limits and/or the steps
    # are asymmetric, so every node of the full contiguous stencil is needed (including
    # those whose uniform centered weight would vanish) and the weights are solved at
    # trace time from the Vandermonde system in the scaled node positions.
    nside = int(np.max(offsets))
    full_offsets = np.arange(-nside, nside + 1)
    n_nodes = len(full_offsets)
    # Signed distance of each node from the (possibly shifted) stencil base.
    signed_steps = np.array([off * (eps_below if off < 0 else eps_above) for off in full_offsets])
    rhs = np.zeros(n_nodes)
    rhs[k] = float(math.factorial(k))

    base_lo = -np.inf if _prior_lo is None else _prior_lo + nside * eps_below
    base_hi = np.inf if _prior_hi is None else _prior_hi - nside * eps_above
    if base_lo > base_hi:
        raise ValueError('cannot fit the order-{:d} stencil for {} (steps {}, {}) within prior limits {}; '
                         'decrease fd_eps or widen the prior'.format(k, name, eps_below, eps_above, prior_limits))

    def _node_weights(p0, p_base):
        """Interpolation weights w such that f^(k)(p0) = sum_j w_j f(node_j) / h_k.

        Vandermonde system in the eps_avg-scaled node positions relative to p0:
        sum_j w_j u_j^r = r! delta_{r k}, u_j = (node_j - p0) / eps_avg.
        ``p0`` may carry batch shape B; returns shape (*B, n_nodes).
        """
        u = (p_base - p0)[..., None] / eps_avg + jnp.asarray(signed_steps) / eps_avg  # (*B, n)
        rows = [jnp.ones_like(u)]
        for _ in range(n_nodes - 1):
            rows.append(rows[-1] * u)
        matrix = jnp.stack(rows, axis=-2)  # (*B, n_rows, n_nodes)
        rhs_b = jnp.broadcast_to(jnp.asarray(rhs), matrix.shape[:-1])
        return jnp.linalg.solve(matrix, rhs_b[..., None])[..., 0]

    def wrapped(p_dict):
        p0 = jnp.asarray(p_dict[name])
        # All stencil geometry (base shift, node positions, weights) lives in the
        # expansion variable u; only the function evaluations map back to p.
        u0 = p0 if fwd is None else fwd(p0)
        # shift the stencil base inward so every node stays within the prior limits
        u_base = jnp.clip(u0, base_lo, base_hi)
        if p0.ndim:
            u0, u_base = u0.reshape(-1), u_base.reshape(-1)
        # each element's own value sets its shift and so its weights
        weights = _node_weights(u0, u_base)
        node_values = [u_base + step if inv is None else inv(u_base + step) for step in signed_steps]
        return _fd_stencil_sum(fn, p_dict, name, p0, node_values, weights, h_k)

    return wrapped


# ── external function factory ─────────────────────────────────────────────────

def _make_external_fn(node: Calculator, params_list: list, calc_deps: list, call_return, node_state: dict):
    """
    Return (fn_dep, fn_call, call_kind).

    fn_dep  -> pure_callback-wrapped callable returning tree_flatten children
               (used to pass outputs downstream).
    fn_call -> pure_callback-wrapped callable returning the __call__() return value
               (used when this node is the pipeline root), or ``None`` when
               ``call_kind`` is ``'none'``/``'self'`` (no array output to marshal).
    call_kind -> ``'value'`` (a real array/tuple output), ``'none'`` (__call__ returned
               None) or ``'self'`` (__call__ returned the node itself).

    Differentiation is handled at the CompiledGraph level via _build_graph_call_fn;
    these functions carry no custom_jvp of their own.

    node() is skipped when own params and all dep outputs are unchanged from the previous
    *actual call* of this pure_callback. This is checked by comparing concrete values
    (own_params_tuple, dep_args) against the last-seen ones, NOT via a 'was a dep node
    called' flag: under jax.jit, the Python graph-walk in _run_graph (which would set such
    a flag) runs only once, at trace time, while this callback itself runs at every actual
    execution of the compiled program (jax.pure_callback always invokes it). A flag set once
    at trace time would stay stale forever after that, making every execution look like deps
    changed and defeating the cache; comparing actual values, fetched at call time, gives the
    right answer in eager mode too.
    """
    dep_schema = []
    for dep in calc_deps:
        dep_children_raw, dep_aux = dep.tree_flatten()
        dep_children_flat, dep_treedef = jax.tree_util.tree_flatten(dep_children_raw)
        dep_schema.append((dep, len(dep_children_flat), dep_treedef, dep_aux))

    own_children_raw, _ = node.tree_flatten()
    own_children, own_treedef = jax.tree_util.tree_flatten(own_children_raw)
    dep_sdt = tuple(jax.ShapeDtypeStruct(np.asarray(c).shape, np.asarray(c).dtype) for c in own_children)

    # __call__ may return None (outputs live in attributes) or self (the populated
    # node); neither carries a separate array output to marshal through pure_callback,
    # so no call-result callback is built and the pipeline handles them directly.
    if call_return is None:
        call_kind = 'none'
    elif call_return is node:
        call_kind = 'self'
    else:
        call_kind = 'value'

    def _run_or_cache(own_params_tuple, dep_args):
        last_params = node_state['last_params']
        params_changed = last_params is None or any(not np.array_equal(a, b) for a, b in zip(own_params_tuple, last_params))
        last_dep_args = node_state['last_dep_args']
        dep_args_changed = (last_dep_args is None or len(dep_args) != len(last_dep_args)
                            or any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(dep_args, last_dep_args)))
        if params_changed or dep_args_changed:
            node_state['last_params'] = tuple(np.asarray(a) for a in own_params_tuple)
            node_state['last_dep_args'] = tuple(np.asarray(a) for a in dep_args)
            for i, param in enumerate(params_list):
                if getattr(param, '_call_fn', None) is None:
                    param.value = np.asarray(own_params_tuple[i])
            for param in params_list:
                param()
            offset = 0
            for dep, n_children, dep_treedef, dep_aux in dep_schema:
                dep_flat_slice = list(dep_args[offset:offset + n_children])
                dep_children = jax.tree_util.tree_unflatten(dep_treedef, dep_flat_slice)
                proxy = dep.__class__.tree_unflatten(dep_aux, dep_children)
                dep.__dict__.update(proxy.__dict__)
                offset += n_children
            # See the matching comment in _run_graph: node_state['is_tracing'] (not a plain
            # node attribute) is what survives from trace time to this actual (deferred)
            # callback invocation; expose it as a node attribute now, right before node()
            # runs, so __call__ can read self._is_tracing.
            node._is_tracing = node_state.get('is_tracing', False)
            call_result = node()
            node_state['dep_result'] = tuple(np.asarray(c) for c in jax.tree_util.tree_leaves(node.tree_flatten()[0]))
            node_state['call_result'] = call_result
            node_state['was_called'] = True
        else:
            node_state['was_called'] = False

    def _inject_and_call_dep(own_params_tuple, *dep_args):
        _run_or_cache(own_params_tuple, dep_args)
        return node_state['dep_result']

    def _inject_and_call_call(own_params_tuple, *dep_args):
        _run_or_cache(own_params_tuple, dep_args)
        call_result = node_state['call_result']
        if isinstance(call_result, tuple):
            return tuple(np.asarray(r) for r in call_result)
        return np.asarray(call_result)

    def _make_fn(callback, result_sdt):
        def fn(own_params_tuple, dep_attr_flat):
            return jax.pure_callback(callback, result_sdt, own_params_tuple, *dep_attr_flat, vmap_method='sequential')
        return fn

    fn_dep = _make_fn(_inject_and_call_dep, dep_sdt)
    if call_kind != 'value':
        return fn_dep, None, call_kind
    if isinstance(call_return, tuple):
        call_sdt = tuple(jax.ShapeDtypeStruct(np.asarray(r).shape, np.asarray(r).dtype) for r in call_return)
    else:
        call_sdt = (jax.ShapeDtypeStruct(np.asarray(call_return).shape, np.asarray(call_return).dtype),)
    call_result_sdt = call_sdt[0] if len(call_sdt) == 1 else call_sdt
    return fn_dep, _make_fn(_inject_and_call_call, call_result_sdt), call_kind


# ── graph-level custom JVP ────────────────────────────────────────────────────

def _build_graph_call_fn(pipeline):
    """
    Build and return a jax.custom_jvp-wrapped call function for the CompiledGraph.

    The returned function has signature (fd_params_tuple, jax_params_tuple) ->
    (return_val, derived_dict, ext_outputs_flat).

    JVP strategy:
      fd_params (feed any external node directly or transitively): finite differences
        over the full graph — O(n_fd_params × n_stencil) full-graph evaluations.
      jax_params (only feed JAX nodes): exact forward-mode AD through the JAX
        sub-graph, with external outputs frozen at their primal values.
    """
    nodes = pipeline.nodes
    node_var_deps = pipeline._node_var_deps
    node_calc_deps = pipeline._node_calc_deps
    node_states = pipeline._node_states
    tree_own_aux = pipeline._tree_own_aux
    tree_own_treedef = pipeline._tree_own_treedef
    fn_dep = pipeline._fn_dep
    fn_call = pipeline._fn_call
    ext_n_children = pipeline._ext_n_children
    ext_call_kind = pipeline._ext_call_kind
    root = pipeline.root
    output = pipeline.output
    all_params = list(pipeline.params)
    derived_params = pipeline._derived_params
    fd_params = pipeline._fd_params
    jax_params = pipeline._jax_params
    fd_names = [p.name for p in fd_params]
    jax_names = [p.name for p in jax_params]

    def _run_graph(params, ext_flat=None, stage_i_ids=frozenset(), stage_i_flat=None,
                   skip_ids=frozenset(), output_override=None):
        """
        Execute the graph.
        ext_flat=None  : full run — External nodes execute via pure_callback.
        ext_flat=tuple : JAX sub-graph run — External outputs taken from ext_flat,
                         External nodes are not called. Used in the JAX-params JVP.
        stage_i_ids, stage_i_flat :
            Two-stage optimisation for analytic marginalisation.  When stage_i_flat is a
            flat tuple of pre-computed JAX leaves (one block per Stage-i node, in node
            topological order then tree_flatten order) and stage_i_ids is the corresponding
            frozenset of node id()s, those nodes are skipped — their outputs are injected
            from stage_i_flat via tree_unflatten rather than recomputed.  Their params are
            still set so that downstream nodes can read them via .value.
        skip_ids :
            Nodes whose id() is in this set are skipped entirely (not called, not injected).
            Used in the Stage-i pre-pass to execute only Stage-i nodes via the full
            group_theory_pipe while bypassing Stage-ii nodes.
        output_override :
            If not None, replaces the pipeline's compiled output function when computing
            return_val.  Used in the Stage-i pre-pass to capture Stage-i node leaves as the
            return value rather than the pipeline's normal output.
        Returns (return_val, derived_dict, ext_outputs_flat).
        """
        is_tracing = any(isinstance(params[p.name], jax.core.Tracer) for p in all_params)

        if is_tracing:
            saved_param_values = {p.name: p._value for p in all_params}
            saved_node_dicts = {id(n): dict(n.__dict__) for n in nodes}

        ext_collected = []
        result = None
        ext_flat_offset = 0
        stage_i_offset = 0

        for i, node in enumerate(nodes):
            if id(node) in skip_ids:
                continue
            # Every node, not just the External ones: a node that calls a non-jax library
            # directly (rather than through a pure_callback of its own) needs the same
            # eager-raise / traced-NaN contract, and the only place able to observe the outer
            # trace status is here. External nodes additionally stash it in `node_state` below,
            # because their callback runs after this loop has finished.
            node._is_tracing = is_tracing
            nvd = node_var_deps[id(node)]
            ncd = node_calc_deps[id(node)]

            if stage_i_flat is not None and id(node) in stage_i_ids:
                # Stage i injection: set params so downstream nodes can read them via .value,
                # unpack pre-computed tree_flatten outputs into node.__dict__, skip node().
                free_nvd = [param for param in nvd if getattr(param, '_call_fn', None) is None]
                for param in free_nvd:
                    param.value = params[param.name]
                for param in nvd:
                    param()
                n_ch = tree_own_treedef[i].num_leaves
                flat_children = list(stage_i_flat[stage_i_offset:stage_i_offset + n_ch])
                stage_i_offset += n_ch
                children = jax.tree_util.tree_unflatten(tree_own_treedef[i], flat_children)
                proxy = node.__class__.tree_unflatten(tree_own_aux[i], children)
                node.__dict__.update(proxy.__dict__)
                node_states[id(node)]['was_called'] = True
            elif node._is_external:
                # Threaded through so the node's __call__ (running inside pure_callback,
                # where values are always concrete) can tell whether the *enclosing* graph
                # execution is jax-traced — pure_callback itself can never expose that.
                # Written into the persistent node_state dict (read back by _run_or_cache
                # right before it actually calls node()), rather than a plain node attribute:
                # jax.pure_callback defers the actual callback invocation to program
                # *execution* time, which happens after this trace-time loop (and the
                # is_tracing-restore block below it) has already finished running, so a
                # plain `node._is_tracing = ...` here would be wiped by the restore before
                # the deferred callback ever got a chance to read it.
                node_states[id(node)]['is_tracing'] = is_tracing
                n_ch = ext_n_children[id(node)]
                if ext_flat is not None:
                    # JAX sub-graph mode: unpack frozen External outputs.
                    flat_children = list(ext_flat[ext_flat_offset:ext_flat_offset + n_ch])
                    children = jax.tree_util.tree_unflatten(tree_own_treedef[i], flat_children)
                    proxy = node.__class__.tree_unflatten(tree_own_aux[i], children)
                    node.__dict__.update(proxy.__dict__)
                    ext_collected.extend(flat_children)
                else:
                    own_params_tuple = tuple(jnp.asarray(params[p.name]) for p in nvd)
                    dep_attr_flat = tuple(v for dep in ncd for v in jax.tree_util.tree_leaves(dep.tree_flatten()[0]))
                    # Always call fn_dep to capture tree_flatten children into ext_collected.
                    raw_dep = fn_dep[i](own_params_tuple, dep_attr_flat)
                    children = jax.tree_util.tree_unflatten(tree_own_treedef[i], list(raw_dep))
                    proxy = node.__class__.tree_unflatten(tree_own_aux[i], children)
                    node.__dict__.update(proxy.__dict__)
                    ext_collected.extend(raw_dep)
                    if node is root and output is None:
                        kind = ext_call_kind[id(node)]
                        if kind == 'none':
                            result = None
                        elif kind == 'self':
                            result = node  # populated above via tree_unflatten
                        else:
                            # fn_call reuses the cached computation from fn_dep above.
                            result = fn_call[i](own_params_tuple, dep_attr_flat)
                ext_flat_offset += n_ch
            else:
                node_state = node_states[id(node)]
                if is_tracing:
                    for param in nvd:
                        if getattr(param, '_call_fn', None) is None and getattr(param, 'derived', None) is not True:
                            param.value = params[param.name]
                    for param in nvd:
                        param()
                    result = node()
                    node_state['was_called'] = True
                else:
                    # Derived parameters are outputs, computed by node(): seeding them from the
                    # input dict would overwrite the value the node just produced (and, on a
                    # second _run_graph pass, leave the stale input value standing).
                    free_nvd = [param for param in nvd if getattr(param, '_call_fn', None) is None
                                and getattr(param, 'derived', None) is not True]
                    own_params_np = np.concatenate([np.ravel(np.asarray(params[param.name])) for param in free_nvd]) if free_nvd else np.array([])
                    dep_states_list = [node_states[id(dep)] for dep in ncd]
                    dep_was_called = any(s['was_called'] for s in dep_states_list)
                    params_changed = node_state['last_params'] is None or not np.array_equal(own_params_np, node_state['last_params'])
                    if dep_was_called or params_changed:
                        for param in free_nvd:
                            param.value = params[param.name]
                        for param in nvd:
                            param()  # _call_fn (derived parameters) if it is the case
                        result = node()
                        # Write back derived param values: after __call__ the node may have
                        # set self.<name> = computed_value, overwriting the Parameter reference.
                        # Capture that value into p._value so callers can read it.
                        for param in nvd:
                            if param.derived is True:
                                val = node.__dict__.get(param.name)
                                if val is not None and not isinstance(val, Variable):
                                    param._value = np.asarray(val)
                        node_state['last_params'] = own_params_np
                        node_state['last_result'] = result
                        # Cache the output-function result for the root node, to guard
                        # against stale live attributes when multiple CompiledGraph
                        # instances share the same node.
                        if node is root and output is not None:
                            node_state['last_output'] = output()
                        node_state['was_called'] = True
                    else:
                        result = node_state['last_result']
                        node_state['was_called'] = False

        # Use the cached output value when available, so live attributes updated by
        # a different CompiledGraph sharing the same node are not incorrectly returned.
        root_state = node_states.get(id(root))
        if (output_override is None and output is not None and root_state is not None
                and 'last_output' in root_state and not root_state.get('was_called', True)):
            return_val = root_state['last_output']
        else:
            effective_output = output_override if output_override is not None else output
            return_val = effective_output() if effective_output is not None else result
        # Capture derived (incl. solved) values before the tracing restore below
        # overwrites _value back to the pre-call snapshot.
        derived_dict = {p.name: p._value for p in derived_params}
        ext_flat_out = tuple(ext_collected)

        if is_tracing:
            for node in nodes:
                node.__dict__.clear()
                node.__dict__.update(saved_node_dicts[id(node)])
            for p in all_params:
                p._value = saved_param_values[p.name]
            # The shallow dict save/restore above cannot undo in-place mutations of
            # nested mutable objects (e.g. a cosmoprimo _results dict written during
            # __call__ with Tracer-valued inputs). Reset last_params for every non-external
            # node so the next eager call re-runs them and overwrites any stale
            # trace-escaped state (including JAX Tracers trapped in nested attributes).
            # External nodes are excluded: their attributes are always set from concrete
            # pure_callback outputs (never from Tracer-valued writes), so their last_params
            # cache is safe to preserve across the jax.linearize boundary. This allows the
            # shared-node_state caching (e.g. CosmoprimoCosmology shared across groups) to
            # remain valid after stage-ii tracing, so sibling groups find a cache hit.
            for node in nodes:
                if not node._is_external:
                    node_states[id(node)]['last_params'] = None

        return return_val, derived_dict, ext_flat_out

    # Expose _run_graph directly so callers (e.g. Posterior._marg_loglik) can bypass
    # the @custom_jvp wrapper and let JAX trace through the Stage-ii sub-graph natively.
    pipeline._run_graph_fn = _run_graph

    @jax.custom_jvp
    def call_fn(fd_params_tuple, jax_params_tuple):
        params = {**dict(zip(fd_names, fd_params_tuple)), **dict(zip(jax_names, jax_params_tuple))}
        return _run_graph(params)

    def call_fn_jvp(primals, tangents):
        fd_p, jax_p = primals
        vfd, vjax = tangents

        primal_val, primal_derived, primal_ext = call_fn(fd_p, jax_p)

        # ── FD tangent for fd_params ──────────────────────────────────────────
        # One full-graph call per stencil point per scalar element of each fd_param.
        # return_val may be an arbitrary pytree (array, None for a side-effect-only
        # root, or the node itself), so all arithmetic goes through tree_map.
        # ``symbolic_zeros=True`` lets us skip params that are not being differentiated:
        # this avoids needless graph re-evaluations and, crucially, never perturbs a
        # fixed/constant param out of its valid domain (e.g. a negative neutrino mass).
        tangent_val = jax.tree_util.tree_map(jnp.zeros_like, primal_val)
        tangent_derived = jax.tree_util.tree_map(jnp.zeros_like, primal_derived)
        for i, param in enumerate(fd_params):
            if isinstance(vfd[i], SymbolicZero):
                continue
            param_fd = param.fd
            fd_eps_param = param_fd.eps
            if fd_eps_param is None and getattr(param, 'ref', None) is not None:
                fd_eps_param = param.ref.std()
            eps_below, eps_above, eps_avg = _fd_parse_eps(fd_eps_param)
            eps_max = max(eps_below, eps_above)
            offsets, coeffs = _fd_stencil(1, param_fd.acc)
            param_arr = jnp.asarray(fd_p[i])
            hsize = len(offsets) // 2  # half-width of stencil (= 1 for acc=2)
            # Prior hard limits for boundary-safe stencil shifting.
            _prior_lo = _prior_hi = None
            if param.prior is not None:
                _lo, _hi = param.prior.limits
                if np.isfinite(_lo): _prior_lo = float(_lo)
                if np.isfinite(_hi): _prior_hi = float(_hi)
            for idx in np.ndindex(param_arr.shape):
                v_ij = vfd[i][idx] if param_arr.ndim > 0 else vfd[i]
                x_ij = param_arr[idx] if param_arr.ndim > 0 else param_arr
                # Shift the stencil base so all points [x_base ± k*eps] stay within prior limits.
                # This mirrors the desilike_bak grid-shifting strategy: if x is near a hard
                # boundary, the stencil is shifted inward rather than clipped asymmetrically.
                x_base = x_ij
                if _prior_lo is not None:
                    x_base = jnp.maximum(x_base, _prior_lo + hsize * eps_max)
                if _prior_hi is not None:
                    x_base = jnp.minimum(x_base, _prior_hi - hsize * eps_max)
                df = jax.tree_util.tree_map(jnp.zeros_like, primal_val)
                df_derived = jax.tree_util.tree_map(jnp.zeros_like, primal_derived)
                for off, coeff in zip(offsets, coeffs):
                    # Use eps_below for negative offsets and eps_above for positive offsets.
                    # This gives the correct non-uniform central difference for 1st order acc=2:
                    #   (f(x+eps_above) - f(x-eps_below)) / (eps_below+eps_above).
                    step = eps_below if off < 0 else eps_above
                    shifted = list(fd_p)
                    shifted[i] = param_arr.at[idx].set(x_base + off * step) if param_arr.ndim > 0 else x_base + off * step
                    fi, fi_derived, _ = call_fn(tuple(shifted), jax_p)
                    df = jax.tree_util.tree_map(lambda a, b, c=coeff: a + c * b, df, fi)
                    df_derived = jax.tree_util.tree_map(lambda a, b, c=coeff: a + c * b, df_derived, fi_derived)
                tangent_val = jax.tree_util.tree_map(lambda t, d, vv=v_ij: t + vv * d / eps_avg, tangent_val, df)
                tangent_derived = jax.tree_util.tree_map(lambda a, b: a + v_ij * b / eps_avg, tangent_derived, df_derived)

        # ── JAX tangent for jax_params ────────────────────────────────────────
        # Forward-mode AD through the JAX sub-graph; External outputs frozen at
        # primal_ext so their contribution is zero (treated as constants).
        # Materialise symbolic-zero tangents to concrete zeros for jax.jvp; skip the
        # sub-graph entirely when no jax_param is being differentiated.
        if jax_names and any(not isinstance(v, SymbolicZero) for v in vjax):
            vjax_concrete = tuple(jnp.zeros_like(jax_p[j]) if isinstance(vjax[j], SymbolicZero) else vjax[j]
                                  for j in range(len(jax_p)))

            def _jax_sub(jp):
                params = {**dict(zip(fd_names, fd_p)), **dict(zip(jax_names, jp))}
                return _run_graph(params, ext_flat=primal_ext)
            _, (jax_tv, jax_tv_derived, _) = jax.jvp(_jax_sub, (jax_p,), (vjax_concrete,))
            tangent_val = jax.tree_util.tree_map(lambda a, b: a + b, tangent_val, jax_tv)
            tangent_derived = jax.tree_util.tree_map(lambda a, b: a + b, tangent_derived, jax_tv_derived)

        tangent_out = (tangent_val, tangent_derived, tuple(jnp.zeros_like(e) for e in primal_ext))
        return (primal_val, primal_derived, primal_ext), tangent_out

    call_fn.defjvp(call_fn_jvp, symbolic_zeros=True)

    return call_fn


def _topo_sort_params(params):
    """Return params in topological order w.r.t. their .depends DAG.

    If param A has derived='B * 2' with depends containing param B, B is
    guaranteed to appear before A so that param() calls evaluate in the
    correct order. Only edges to params within this list are followed;
    cross-node dependencies are ignored.
    """
    param_ids = {id(p) for p in params}
    order = []
    visiting = set()
    visited = set()

    def visit(param):
        if id(param) in visited:
            return
        if id(param) in visiting:
            raise ValueError(f"Cycle detected in derived-parameter depends involving '{param.name}'")
        visiting.add(id(param))
        for dep in getattr(param, 'depends', {}).values():
            if id(dep) in param_ids:
                visit(dep)
        visiting.discard(id(param))
        visited.add(id(param))
        order.append(param)

    for param in params:
        visit(param)
    return order


# ── compiled pipeline ─────────────────────────────────────────────────────────

class CompiledGraph:
    """
    Static computation graph compiled from a root calculator.

    ``__call__(params)`` is a pure function fully compatible with
    jax.jit, jax.vmap, and jax.grad.
    """

    def __init__(self, root: Calculator, ctx: _CompileContext, output=None, input=None, shared_node_states=None):
        self.root = root
        self.output = output
        self.input = input
        self.nodes = ctx.node_order
        # One concept for "is this graph still the current one for its nodes": it owns them until
        # something else claims them or they are reconfigured, and then carries the reason.
        self._context_id = id(ctx)
        self._invalid = None
        _take_ownership(self, self.nodes, self._context_id)

        # Per-node dep lists split by type.
        self._node_var_deps = {}
        self._node_calc_deps = {}
        for node in self.nodes:
            deps = ctx.node_deps.get(id(node), [])
            self._node_var_deps[id(node)] = _topo_sort_params([d for d in deps if isinstance(d, Variable)])
            self._node_calc_deps[id(node)] = [d for d in deps if isinstance(d, Calculator)]

        # Collect unique Variable objects preserving DAG order, deduplicated by identity.
        # Two distinct Variable objects with the same name is an error — share the instance.
        self.params = VariableCollection()
        seen_ids = set()
        for node in self.nodes:
            for param in self._node_var_deps[id(node)]:
                if id(param) not in seen_ids:
                    if param.name in self.params:
                        raise ValueError(f'Variable {param.name!r} is owned by multiple nodes as distinct objects; pass the same Variable instance to both')
                    seen_ids.add(id(param))
                    self.params.set(param)

        self._derived_params = [p for p in self.params if p.derived]  # True, 'marg', 'best'

        # Per-node cache state for all nodes (keyed by node id).
        self._node_states = {id(node): {'last_params': None, 'was_called': False, 'last_result': None,
                                        'dep_result': None, 'call_result': None, 'last_dep_args': None}
                             for node in self.nodes}
        # Allow callers to share a node_state dict across multiple CompiledGraph instances so
        # that a pure_callback cache hit in one graph is visible to sibling graphs.  The shared
        # dict must be installed before _fn_dep closures are built below, because those closures
        # capture node_state by reference at creation time.
        if shared_node_states:
            for nid, shared_state in shared_node_states.items():
                if nid in self._node_states:
                    self._node_states[nid] = shared_state

        # Build external (_is_external=True) callables; record tree_flatten child counts.
        self._tree_own_aux = []
        self._tree_own_treedef = []
        self._fn_dep = []
        self._fn_call = []
        self._ext_n_children = {}
        self._ext_call_kind = {}  # id(node) -> 'value' | 'none' | 'self' (External roots)
        for node in self.nodes:
            children_raw, aux = node.tree_flatten()
            children_flat, treedef = jax.tree_util.tree_flatten(children_raw)
            self._tree_own_aux.append(aux)
            self._tree_own_treedef.append(treedef)
            if node._is_external:
                self._ext_n_children[id(node)] = len(children_flat)
                node_state = self._node_states[id(node)]
                calc_deps = self._node_calc_deps[id(node)]
                call_return = ctx.call_returns[id(node)]
                fn_dep, fn_call, call_kind = _make_external_fn(node, self._node_var_deps[id(node)], calc_deps, call_return, node_state)
                self._fn_dep.append(fn_dep)
                self._fn_call.append(fn_call)
                self._ext_call_kind[id(node)] = call_kind
            else:
                self._fn_dep.append(None)
                self._fn_call.append(None)

        # Compute which params must be FD-differentiated (feed any external node
        # directly or transitively) vs which can use exact JAX auto-diff.
        downstream_of = defaultdict(list)
        for node in self.nodes:
            for dep in self._node_calc_deps[id(node)]:
                downstream_of[id(dep)].append(node)

        ext_reach = set()
        for node in reversed(self.nodes):
            if node._is_external:
                ext_reach.add(id(node))
            elif any(id(ds) in ext_reach for ds in downstream_of[id(node)]):
                ext_reach.add(id(node))

        fd_param_names = {p.name for node in self.nodes if id(node) in ext_reach for p in self._node_var_deps[id(node)]}
        self._fd_params  = [p for p in self.params if p.name in fd_param_names]
        self._jax_params = [p for p in self.params if p.name not in fd_param_names]
        self._fd_names = [p.name for p in self._fd_params]
        self._jax_names = [p.name for p in self._jax_params]

        self._call_fn = _build_graph_call_fn(self)

    @functools.cached_property
    def _jit_call_fn(self):
        """JIT-compiled version of ``_call_fn``; created once and cached on the graph."""
        return jax.jit(self._call_fn)

    def release(self):
        """Give up ownership of this graph's nodes, without invalidating this graph.

        Rarely needed: reconfiguring a node invalidates its owners by itself, and a graph nobody
        refers to is collected.  It is here for the case where something still holds this graph --
        a notebook's ``Out[7]``, a profiler -- and you want the nodes free of it.
        """
        for node in self.nodes:
            if node._owners is not None:
                node._owners.discard(self)

    def _check_not_stale(self):
        """Raise if this graph is no longer the current one for its nodes.

        One flag, set by whoever invalidated it, carrying the reason: another build claimed these
        nodes, or one of them was reconfigured.  The node objects are shared and a build writes
        their setup (`__post_init__`) onto them in place, so a graph that has been superseded
        would otherwise compute on someone else's configuration -- which it did, silently, before
        this check existed.
        """
        if self._invalid is not None:
            raise RuntimeError(
                f'{type(self.root).__name__} graph is no longer valid: {self._invalid}. '
                f'Build it again, or build the other one on `copy(...)` of the tree to keep both.')

    def _check_names(self, names):
        """Raise on any parameter name this graph does not have.

        A name that is not a parameter of the graph used to be dropped in silence, leaving the
        parameter at its default: the pipeline still runs and still returns a plausible number,
        just not the function the caller asked for.  Derived and solved names are accepted --
        they are Variables of the graph, and the marginalization machinery writes solved values
        back into the dict it passes on.

        Callers holding a superset of the graph's parameters (the prior sub-graph, a
        marginalization group, a component of a sum) must filter before calling, not rely on
        the graph to drop the extras.
        """
        unknown = [name for name in names if name not in self.params]
        if not unknown:
            return
        available = self.params.names()
        detail = []
        for name in unknown:
            close = difflib.get_close_matches(name, available, n=3, cutoff=0.5)
            detail.append(repr(name) + (' (did you mean {}?)'.format(close) if close else ''))
        raise ValueError('{} pipeline has no parameter {}. It would otherwise be silently '
                         'ignored and the parameter left at its default. Available: {}'.format(
                             type(self.root).__name__, '; '.join(detail), sorted(available)))

    def __call__(self, *args, return_derived=False, **kwargs):
        """
        Pure function: params → return_val, or ``(return_val, derived_dict)``
        when *return_derived* is ``True`` (keyword-only).
        Compatible with jax.jit, jax.vmap, jax.grad (dict form only for JAX transforms;
        wrap in a lambda/partial to fix ``return_derived`` before passing to jit/vmap).

        Calling conventions
        -------------------
        When no *input* callable was supplied to :func:`build`:

            pipe(params_dict)          # explicit {name: value} dict
            pipe(**overrides)          # override specific params; rest from defaults
            pipe()                     # all params from their current default values

        When an *input* callable was supplied:

            pipe(pytree)               # input(pytree) for side-effects; params from defaults + kwargs
            pipe(pytree, params_dict)  # input(pytree) for side-effects; explicit param dict
            pipe(pytree, **overrides)  # input(pytree) for side-effects; override specific params

        The *input* callable is invoked for its side-effects only (return value ignored).
        Kwargs form is a convenience for eager calls; missing params are filled from defaults.

        A name the graph does not have raises :exc:`ValueError` in either form, rather than
        being dropped and the parameter left at its default.  Callers that legitimately hold a
        superset (the prior sub-graph, a marginalization group, one component of a sum) filter
        with :func:`_restrict` first.

        After each eager call the parameters' ``.value`` attributes are restored
        to whatever they were on entry, so that finite-difference mutations
        inside ``pure_callback`` do not corrupt subsequent default-argument calls.
        """
        self._check_not_stale()
        if self.input is not None:
            self.input(args[0] if args else None)
            params = args[1] if len(args) > 1 else None
        else:
            params = args[0] if args else None
        # Snapshot current values; used as defaults and for post-call restoration.
        # What decides the restore is whether the call is traced, not whether a param is
        # derived: a traced value must not outlive its trace, whatever it is attached to, and an
        # eager one is the result the caller just asked for, whatever it is attached to.  See
        # the `finally` below.
        all_saved = {p.name: p._value for p in self.params}
        # Reject unknown names before the merge, in both calling conventions: after the merge
        # an unknown name is indistinguishable from a legitimate one.
        if params is not None:
            self._check_names(params)
        if kwargs:
            self._check_names(kwargs)
        if params is None:
            params = dict(all_saved)
            params.update(kwargs)
        else:
            missing = {n: v for n, v in all_saved.items() if n not in params}
            params = {**params, **missing}
            if kwargs:
                params.update(kwargs)
        fd_params_tuple = tuple(jnp.asarray(params[n]) for n in self._fd_names)
        jax_params_tuple = tuple(jnp.asarray(params[n]) for n in self._jax_names)
        # A tracer among the inputs means every value this call writes onto a node is a tracer.
        is_tracing = any(isinstance(value, jax.core.Tracer) for value in params.values())
        try:
            return_val, derived_dict, _ = self._call_fn(fd_params_tuple, jax_params_tuple)
        finally:
            if is_tracing:
                # Nothing survives a trace. A tracer left on a node escapes into the next eager
                # call, where using it raises `UnexpectedTracerError` far from here -- measured
                # across the profiler suite, from a derived Parameter a traced call had written.
                # The values themselves are not lost: they are in `derived_dict`.
                for p in self.params:
                    p._value = all_saved[p.name]
            # Eager: nothing is restored. The tree is left reflecting the evaluation that just
            # ran -- `get_params(graph)['b1'].value` is what was passed, and a node's computed
            # outputs (a derived Parameter, a `flattheory` Variable) are there to be read. An
            # operation that perturbs a parameter *internally* puts it back itself; see
            # `differentiate`.
        if return_derived:
            return return_val, derived_dict
        return return_val


def _iter_nodes(calc, level=None, exclude=None, filter=None):
    """Yield the nodes reachable from *calc*, walking its transitive Calculator dependencies
    (depth-first, cycle-safe).

    Sub-calculators are discovered (via :func:`_iter_node`) in both the constructor args
    (``_init``) and the public attributes.  Each calculator is yielded *before* its children are
    scanned, so a consumer that mutates a calculator (e.g. :func:`replace`) affects what is
    subsequently descended into.

    Parameters
    ----------
    level : int or None
        Maximal recursion depth (``None`` = unlimited, ``0`` = *calc* only, ``1`` =
        *calc* and its direct Calculator dependencies, ...).
    exclude : set of int or None
        Object ids never yielded nor descended into.
    filter : type, callable or None
        What to yield.  ``None`` (the default) yields the Calculators, which is the walk almost
        every caller wants.  A type or a predicate yields the nodes matching it instead:
        ``filter=Variable`` gives every Variable reachable from *calc*, at any depth, which is
        what :func:`share_params` needs.  The walk descends through Calculators either way.
    """
    seen = set(exclude or ())
    if filter is None:
        def match(node): return isinstance(node, Calculator)
    elif isinstance(filter, type):
        def match(node, _type=filter): return isinstance(node, _type)
    else:
        match = filter

    def _walk(current, depth):
        if id(current) in seen:
            return
        seen.add(id(current))
        if match(current):
            yield current
        if level is not None and depth >= level:
            return
        # a calculator's Node references live in its constructor args/kwargs and its public
        # attributes; both are scanned, so a dep reachable by either path is found
        args, kwargs = current._init
        sources = list(args) + list(kwargs.values())
        sources += [val for key, val in current.__dict__.items() if not key.startswith('_')]
        for src in sources:
            for dep in _iter_node(src):
                if isinstance(dep, Calculator):
                    if id(dep) not in seen:
                        yield from _walk(dep, depth + 1)
                elif match(dep):
                    yield dep

    yield from _walk(calc, 0)


def replace(node, old, new, level: int=None):
    """Replace, in *node* and its (transitive) Calculator dependencies, every Node
    matched by *old* with *new*.

    Parameters
    ----------
    node : Calculator
        Root calculator to rewrite in place.
    old : Node or callable
        Either a :class:`~desilike.parameter.Node` (matched by identity) or a predicate
        ``callable(Node) -> bool`` returning ``True`` for nodes that should be replaced.
    new : Node
        Replacement node.
    level : int or None
        Maximal dependency depth to descend into (see :func:`_iter_nodes`);
        ``None`` (default) is unlimited.

    Walks both the stored constructor arguments (``_init``) and the public attributes
    (``__dict__``), rebuilding nested containers.  Intended at construction time, before
    :func:`build` — e.g. to share a parameter across calculators::

        replace(bispectrum, bispectrum.b1, power_spectrum.b1)
        replace(bispectrum, lambda p: p.name == 'b1', power_spectrum.b1)

    Returns *node* (for chaining).
    """
    # Normalize *old* to a predicate.  A Node is itself callable, so test
    # ``isinstance(old, Node)`` before treating *old* as a predicate.
    if isinstance(old, Node):
        match = lambda candidate: candidate is old
    else:
        match = old
    # Never descend into (or substitute inside) the freshly-inserted replacement.
    exclude = {id(new)} if isinstance(new, Node) else None
    for calc in _iter_nodes(node, level=level, exclude=exclude):
        # Constructor args (so a later __post_init__ that reads _init stays consistent).
        args, kwargs = calc._init
        new_args = tuple(_replace_node(arg, match, new) for arg in args)
        new_kwargs = {key: _replace_node(val, match, new) for key, val in kwargs.items()}
        # Public attributes.
        extra_init_kwargs = {}
        for key, val in list(calc.__dict__.items()):
            if key.startswith('_'):
                continue
            new_val = _replace_node(val, match, new)
            if new_val is not val:
                setattr(calc, key, new_val)
                # Sync _init for direct Calculator-valued attributes so that a subsequent
                # update() re-uses the replacement rather than re-creating the old node.
                # Variables/Parameters are excluded (their identity in _init is fine to keep
                # as-is; auto-share replaces them in _init directly above).
                # Containers are excluded because _replace_node already updated their
                # counterparts in new_args/new_kwargs, and adding a container key here would
                # duplicate positional args already stored in _init.
                if isinstance(val, Calculator) and (key not in new_kwargs or new_kwargs[key] is not new_val):
                    extra_init_kwargs[key] = new_val
        calc._init = (new_args, {**new_kwargs, **extra_init_kwargs})
    return node


def copy(node, level=None):
    """Return an independent copy of *node* and its Calculator dependencies.

    Every Calculator in the tree (down to depth *level*, all of it by default) is
    re-instantiated -- the constructor is called again with a shallow copy of the stored
    ``_init`` arguments -- so no Calculator is shared.  That is what makes the copy the way to
    build a second graph over one tree: a build configures Calculators in place, so two graphs
    must not share any.

    Variables are shared, all of them: the ones the copy's constructors create are bound back
    by name to the original's, exactly as a build binds the ones ``__init__`` creates to the ones
    it was handed.  So ``get_params(copy)['b1'] is get_params(original)['b1']``, with whatever
    prior was set on it, and one sampled value feeds both graphs.

    The default is the whole tree because that is what every caller means by ``copy``.
    ``desi-clustering`` copies a likelihood to get a profiler's, a sampler's and an emulated
    one; with the old default of ``level=1`` those shared every theory, template and
    cosmology below the root and each build reconfigured the others' -- it only worked
    because the graphs were used one after another.  Pass *level* for a deliberately shallow
    copy.

    The *level* semantics follow :func:`_iter_nodes` and :func:`replace`:
    ``level=0`` copies only *node* itself; ``level=1`` copies *node* and its direct
    Calculator dependencies; ``None`` (default) copies the entire tree.

    Parameters
    ----------
    node : Calculator
        Root calculator to copy.
    level : int or None, default=None
        Maximum dependency depth to copy (``None`` = unlimited).

    Returns
    -------
    Calculator
        The newly-created root instance.
    """
    return _init_graph(node, level=level, fresh=True, bind=get_params(node))


def _init_graph(node, level=None, fresh=False, bind=None):
    """Run every Calculator's ``__init__`` again from its stored arguments -- on new objects
    (*fresh*, what :func:`copy` does) or in place (what a build does) -- and return the root.

    A calculator's arguments go first: a constructor is free to reconfigure the calculators it
    is handed (``self.template.update(k=...)`` is how every theory sizes its template), so they
    must be the reconstructed ones by then.  Copying in reversed traversal order once handed a
    ``pt`` the *original* template whenever it was reachable by two paths (through the theory's
    own arguments and through ``pt``), and the original template then swapped its cosmology
    under the graph built on it.  Recursing through ``_init`` visits every argument once,
    however many paths lead to it; calculators a constructor creates itself are created by it.

    In place, the build claims on the tree are lifted first, so that a graph built earlier over
    it is stale for one reason, this build, and not for each ``update()`` a constructor makes.

    *bind* is the collection every reconstructed node is rewired to, node by node as it is built
    rather than once at the end -- see the comment at the call.
    """
    to_visit = {id(calc) for calc in _iter_nodes(node, level=level)}
    done = {}

    def _remap(value):
        if isinstance(value, Calculator):
            if id(value) not in to_visit:
                return value
            if id(value) not in done:
                args, kwargs = value._init
                args = [_remap(arg) for arg in args]
                kwargs = {key: _remap(val) for key, val in kwargs.items()}
                if fresh:
                    new = done[id(value)] = type(value)(*args, **kwargs)
                else:
                    value.__init__(*args, **kwargs)
                    new = done[id(value)] = value
                if bind is not None:
                    # Here, not once after the whole traversal: a constructor is free to read its
                    # children's parameters -- `Posterior.__init__` calls `get_params` -- and one
                    # that does has to see settled objects. `__init__` makes fresh Parameters
                    # every time it runs, so between a child's reconstruction and a single bind at
                    # the end the tree holds two objects per name. Measured: a joint P+B build
                    # raised on `b1` from inside this traversal, before that bind could repair it.
                    #
                    # The whole subtree, so that a constructor which reconfigured what it was
                    # handed (`self.template.update(k=...)`) is repaired by its own parent's turn.
                    #
                    # By name, which is what leaves `update()` free to change the parameter set at
                    # all -- counterterms and band powers come and go with `k` and `ells`. Names
                    # that disappeared stay gone, names that appeared are left alone, and only the
                    # survivors are rewired to the object the tree already had.
                    _bind_variables(new, bind)
            return done[id(value)]
        if isinstance(value, (list, tuple)):
            return type(value)([_remap(item) for item in value])
        if isinstance(value, dict):
            return {key: _remap(val) for key, val in value.items()}
        return value

    return _remap(node)


def share_params(calculators, names=None, level: int=None):
    """Share Parameter objects across *calculators* so that same-named parameters
    become a single object — one prior and one value when they are compiled together.

    Parameters
    ----------
    calculators : sequence of Calculator
        Instances whose parameters should be unified.  For each shared name the
        **first** calculator (in list order) that defines it provides the canonical
        Parameter object; every other occurrence is rewired to it (via :func:`replace`).
    names : str, sequence of str, or None
        Parameter name(s) to share.  ``None`` (default) shares **every** name that
        appears, i.e. all same-named parameters across *calculators* are unified.
    level : int or None
        Maximal dependency depth to search/rewrite (see :func:`_iter_nodes`);
        ``None`` (default) is unlimited.

    Returns
    -------
    list of Calculator
        The same instances, modified in place.

    Examples
    --------
    >>> share_params([power_spectrum, bispectrum], names='b1')
    >>> share_params([power_spectrum, bispectrum])   # unify all same-named parameters
    """
    calculators = list(calculators)
    if isinstance(names, str):
        names = [names]
    names = None if names is None else set(names)
    # Canonical Parameter per name: first occurrence (in calculator order) wins.
    canonical = {}
    for calc in calculators:
        for var in _iter_nodes(calc, level=level, filter=Variable):
            if names is not None and var.name not in names:
                continue
            canonical.setdefault(var.name, var)
    # Rewire every matching parameter in every calculator to its canonical object.
    for name, canon in canonical.items():
        for calc in calculators:
            replace(calc, lambda node, _name=name: isinstance(node, Variable) and node.name == _name, canon, level=level)
    return calculators


def _trace_node(node: Calculator, ctx: _CompileContext) -> None:
    """DFS helper for _trace_graph: scan __dict__ for deps (all nodes were created in
    ``__init__`` at construction, so dependencies are present here before ``__post_init__``)."""
    ctx.traced.add(id(node))
    # Discover deps from public attributes set during construction (__init__/__post_init__).  _iter_node
    # walks arbitrarily nested standard containers (list/tuple/set/dict), stopping
    # at each Node, so deps held in e.g. a dict or tuple-of-tuples are all found.
    for key, val in node.__dict__.items():
        if key.startswith('_'):
            continue
        for dep in _iter_node(val):
            deps = ctx.node_deps.setdefault(id(node), [])
            if id(dep) not in {id(d) for d in deps}:
                deps.append(dep)
            if id(dep) not in ctx.traced and isinstance(dep, Calculator):
                _trace_node(dep, ctx)
    ctx.node_order.append(node)


def _variables_agree(one, other):
    """Whether two same-named :class:`Variable` objects carry the same metadata.

    Merging duplicates that agree loses nothing; merging duplicates that differ silently discards
    one of two answers, which is the failure this guards.  Compared through ``__getstate__`` so a
    subclass's own fields (a Parameter's prior, ref and fd) are included without listing them here.
    The value is left out: it is what a sample sets, not what the Variable declares, and two trees
    that met at different points still declare the same parameter.
    """
    if type(one) is not type(other):
        return False

    def equal(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(equal(*pair) for pair in zip(left, right))
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return np.array_equal(np.asarray(left), np.asarray(right))
        return left == right

    states = [dict(variable.__getstate__()) for variable in (one, other)]
    for state in states:
        state.pop('value')
    return equal(*states)


def _trace_graph(root: Calculator) -> _CompileContext:
    """Traverse root and all reachable Calculators; return the compilation context.

    All nodes were created in __init__ at construction, so this only discovers node
    dependencies (Variable, Parameter, Calculator) by scanning public attributes — without
    running __post_init__ or any __call__.
    Use get_params() or inspect ctx.node_deps / ctx.node_order to examine the graph.

    Same-named :class:`Variable` objects that appear as distinct instances across different
    nodes are automatically unified (first-seen identity wins), equivalent to calling
    :func:`share_params` before compilation.
    """
    outer_ctx = getattr(_compile_context, 'ctx', None)

    def _trace(r):
        c = _CompileContext()
        _compile_context.ctx = c
        try:
            _trace_node(r, c)
        finally:
            _compile_context.ctx = outer_ctx
        return c

    ctx = _trace(root)

    # Same-named Variables appearing as distinct objects are unified -- but only when they
    # agree.  Two calculators built independently and then combined routinely declare the same
    # parameter (two cosmologies in one likelihood, each with its own `h`), and unifying those is
    # the convenience that makes one sampled `h` feed both; measured across the suite, every
    # occurrence is of that kind.
    #
    # When they disagree, merging would silently discard one of two answers, and which one
    # survived was first-seen-by-traversal: a prior narrowed on an emulator's `h` came back as the
    # template's (0.1, 10.0) instead of the (0.66, 0.69) that had been set.  So that case raises
    # and says what differs, and `share_params` is there to make the choice explicit.
    canonical = {}
    needs_sharing = False
    disagreeing = {}
    for node in ctx.node_order:
        for dep in ctx.node_deps.get(id(node), []):
            if not isinstance(dep, Variable):
                continue
            if dep.name not in canonical:
                canonical[dep.name] = dep
            elif dep is not canonical[dep.name]:
                if _variables_agree(dep, canonical[dep.name]):
                    needs_sharing = True
                else:
                    disagreeing.setdefault(dep.name, []).append(type(node).__name__)
    if disagreeing:
        detail = '; '.join(f'{name!r} (differing copies, one of them owned by {sorted(set(owners))})'
                           for name, owners in sorted(disagreeing.items()))
        raise ValueError(
            f'{type(root).__name__} graph has same-named Variables that are distinct objects and '
            f'do not agree: {detail}. Merging them would silently drop one prior, value or '
            f'`fixed` flag in favour of the other. Pass the same instance to both, make them '
            f'agree, or call `share_params` to choose explicitly.')

    if needs_sharing:
        _bind_variables(root, canonical.values())
        ctx = _trace(root)

    return ctx


def _bind_variables(root, variables):
    """Rewire every Variable in *root*'s tree to the same-named one in *variables*."""
    for variable in variables:
        replace(root, lambda node, _v=variable: isinstance(node, Variable) and node.name == _v.name and node is not _v, variable)


def get_params(node_or_graph, level=None) -> VariableCollection:
    """Return the Variable/Parameter collection for a Calculator or CompiledGraph.

    For a :class:`CompiledGraph`, returns the already-collected params (*level* is ignored).
    For a :class:`Calculator`, traverses the constructed graph (no ``__call__``) and collects
    Variable deps up to depth *level* (``None`` = unlimited, ``1`` = root + direct deps only).

    Parameters
    ----------
    node_or_graph : Calculator, CompiledGraph
    level : int or None, default=None
        Maximum Calculator dependency depth to traverse.  Mirrors the *level* argument of
        :func:`copy` and :func:`replace`.  ``None`` collects from the full tree.

    Returns
    -------
    VariableCollection
    """
    if isinstance(node_or_graph, CompiledGraph):
        return node_or_graph.params
    ctx = _trace_graph(node_or_graph)
    nodes_in_scope = set(id(n) for n in _iter_nodes(node_or_graph, level=level))
    result = VariableCollection()
    seen_ids = set()
    for node in ctx.node_order:
        if id(node) not in nodes_in_scope:
            continue
        for dep in ctx.node_deps.get(id(node), []):
            if isinstance(dep, Variable) and id(dep) not in seen_ids:
                if dep.name in result:
                    raise ValueError(f'Variable {dep.name!r} appears as distinct objects in multiple nodes; pass the same instance')
                seen_ids.add(id(dep))
                result.set(dep)
    return result



def build(root: Calculator, output: Callable=None, input: Callable=None) -> CompiledGraph:
    """Trace root's dependency graph and return a CompiledGraph.

    Phase 1 (_trace_graph): discovers deps by scanning the constructed nodes' public attributes.
    Phase 2: runs __post_init__ on each node in dependency order (non-node setup).
    Phase 3: runs __call__ on each node in topological order; raises if __call__ introduces a new Calculator
    not declared at construction; prunes nodes not activated during __call__.

    Parameters
    ----------
    root : Calculator
    output : Callable, optional
        Custom output extractor called after the root node to produce the
        pipeline's return value.
    input : Callable, optional
        Side-effect callable invoked with the first positional argument of the compiled
        graph's ``__call__`` before the graph runs.  Its return value is ignored; it is
        called purely for side-effects (e.g. injecting pre-computed arrays into a shared
        Calculator).  When provided, the parameter dict is taken from the *second*
        positional argument (or defaults + kwargs when omitted).
        When ``None`` (default) the first positional argument is the parameter dict.

    Notes
    -----
    To get derived parameter values on a call, pass ``return_derived=True`` to the
    compiled graph's ``__call__``, e.g. ``val, derived = pipe(params, return_derived=True)``.
    """
    ctx = _build_graph(root)
    return CompiledGraph(root, ctx, output=output, input=input)


def _invalidate_owners(node, reason):
    """Mark every graph built over *node* invalid, with *reason* and the caller's line.

    The graphs are not at fault, so they are not touched further: each simply refuses the next
    time it is called, and says what invalidated it.  Reporting there rather than raising here is
    deliberate -- almost every reconfiguration is legitimate (a constructor sizing the child it
    was handed, an emulator being swapped in), and forbidding those to catch the rare accident
    costs more than it saves.
    """
    owners = node._owners
    if not owners:
        return
    import traceback
    frames = [frame for frame in traceback.extract_stack()[:-2] if 'desilike/base.py' not in frame.filename]
    where = f' at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}' if frames else ''
    for graph in list(owners):
        graph._invalid = f'{reason}{where}'
    owners.clear()


def _take_ownership(graph, nodes, context_id):
    """Give *graph* ownership of *nodes*, invalidating whoever held them for another build.

    Views over one context own their nodes together: `Posterior` builds several `CompiledGraph`
    over one `_CompileContext`, and one must not invalidate the next.
    """
    for node in nodes:
        owners = node._owners
        if owners is None:
            owners = node._owners = weakref.WeakSet()
        for other in list(owners):
            if other._context_id != context_id:
                other._invalid = f'superseded by a later build over {type(node).__name__}'
                owners.discard(other)
        owners.add(graph)


def _build_graph(root: Calculator) -> '_CompileContext':
    """Run _trace_graph + ``__post_init__`` + ``__call__`` for *root*; return the populated context.

    Separated from :func:`build` so that :class:`Posterior` can build the context once and
    create several :class:`CompiledGraph` views over it: views share one build, and one claim.

    A build does **not** restart the tree from its declaration.  Doing so -- re-running every
    ``__init__`` and binding the Variables back by name, which is what would let a rebuild clear
    the requirement registry other calculators wrote (see `_init_graph`, still used by
    :func:`copy`) -- was implemented and reverted: rebinding by name hands the new graph the
    *old* Parameter objects, and a derived one still holds the tracer its last traced call left
    on it, which then escapes into the next eager call (`UnexpectedTracerError`, measured across
    the profiler suite).  Re-init at build needs derived values to be reset, or threaded rather
    than stored on the node, first.
    """
    outer_ctx = getattr(_compile_context, 'ctx', None)
    # The flag stays set for this whole call, not just the reconstruct: a `__post_init__` may
    # build a graph of its own over these nodes (`_get_fiducial(fiducial, cosmo)` does), and that
    # a nested build must not restart the tree again -- it would clear the requirements the outer
    # calculators have just registered, and the cosmology would then be asked for a quantity it
    # was never told to compute (measured: `KeyError: ('params.A_s', ())` from every COMET
    # theory). One build restarts the tree once, at its start.
    init_graph = _compile_context.init_graph
    if not init_graph:
        inputs = get_params(root)
        _compile_context.init_graph = True
        _init_graph(root, bind=inputs)
    ctx = _trace_graph(root)
    _compile_context.ctx = ctx
    # This build claims every node it is about to configure. `__post_init__` writes setup onto
    # the node objects, and those objects are shared with any graph built over them before: a
    # second build over the same nodes reconfigures them under the first graph's feet. Measured:
    # deploying an emulator beside the theory it came from left the theory's own graph answering
    # every cosmological parameter with a zero response, silently. Claiming makes the older
    # graph refuse to run instead (see `CompiledGraph.__call__`); `copy()` is how to keep two.
    # Before the loop, and on the unpruned order: a `__post_init__` that raises part-way has
    # still reconfigured the nodes before it, so the graphs built over them are already wrong.
    # (Ownership itself is taken when the CompiledGraph is created, which is after this returns.)
    for node in ctx.node_order:
        _invalidate_owners(node, 'superseded by a later build')
    try:
        # Run __post_init__ (deferred non-node setup) in dependency order — node_order is
        # post-order DFS, so a node's deps run before it (e.g. a template's __post_init__
        # sets template.k before a theory's __post_init__ reads it). No new nodes here.
        for node in ctx.node_order:
            args, kwargs = node._init
            node.__post_init__(*args, **kwargs)
        ctx.phase = 'call'
        ctx.call_activated.add(id(root))
        # Seed solved parameters (derived='best'/'marg') with a zero placeholder so
        # that __call__ can be traced without a None-value error.  These parameters
        # have no prior and no explicit value, so _value is None at this point.
        seen_param_ids = set()
        for node in ctx.node_order:
            for dep in ctx.node_deps.get(id(node), []):
                if isinstance(dep, Variable) and id(dep) not in seen_param_ids:
                    seen_param_ids.add(id(dep))
                    if getattr(dep, 'solved', False) and dep._value is None:
                        dep._value = np.zeros(dep.shape) if dep.shape else 0.
        for node in ctx.node_order:
            ctx.stack.append(node)
            ret = node.__call__()
            ctx.call_returns[id(node)] = ret
            ctx.stack.pop()
        # Prune Calculator nodes not accessed during __call__
        ctx.node_order = [n for n in ctx.node_order if id(n) in ctx.call_activated]
        for nid in list(ctx.node_deps.keys()):
            ctx.node_deps[nid] = [d for d in ctx.node_deps[nid] if not isinstance(d, Calculator) or id(d) in ctx.call_activated]
    finally:
        _compile_context.ctx = outer_ctx
        _compile_context.init_graph = init_graph
    return ctx




def differentiate(graph, order, params=None, fd=None, fd_acc=None, fd_eps=None, fd_transform=False, jit=False):
    """
    Build a derivative callable for a compiled graph.

    Analogous to ``jax.jacfwd`` / ``jax.hessian`` on a dict input:
    ``differentiate(graph, order)`` returns a function; call that function on a
    *params* dict to evaluate the derivative(s) at that point.

    *order* selects what is computed:

    - ``dict[str | Variable, int]``: a **single mixed partial**.
      ``{'omega_m': 2, 'sigma8': 1}`` → d³/(dω_m² dσ₈).
    - ``int`` (1 or 2), with the *params* argument selecting the parameters:
      **all** partial derivatives of that total order, in one call.
      ``order=1`` returns ``{name: d/dname}`` (like ``jax.jacfwd`` on a dict
      input); ``order=2`` returns the Hessian as a nested dict
      ``{name1: {name2: d²/dname1 dname2}}`` (like ``jax.hessian`` on a dict
      input), with symmetric entries computed once.  See also the
      :func:`jacfwd` and :func:`hessian` shorthands.
    - sequence of dicts: a **batch of mixed partials** sharing setup and
      evaluations; the returned function yields a tuple aligned with *order*.

    Uses JAX forward-mode AD for parameters that feed only JAX calculators and
    *direct* finite-difference stencils for parameters that feed
    external (``_is_external=True``) nodes.  The FD stencil for order *k* and accuracy
    *fd_acc* costs ``k + fd_acc - 1`` graph evaluations — linear in *k* —
    versus the ``fd_acc^k`` cost of nested JVP calls.

    When JAX and FD parameters are mixed the JAX derivative is built first
    (cheaper inner function), and the FD stencil wraps it.  Mixed partials are
    correct because partial derivatives of smooth functions commute.

    Parameters
    ----------
    graph : CompiledGraph or Calculator
        A calculator is built on the spot.
    order : dict[str | Variable, int], int, or sequence of dict
        See above.
    params : sequence of str or Variable, optional
        Only with an ``int`` *order*: the parameters to differentiate with
        respect to.  Defaults to all varied, non-derived parameters of *graph*.
    fd : dict, optional
        ``dict(eps=..., acc=...)``, spelled as for ``Parameter(fd=...)``. The flat ``fd_acc`` /
        ``fd_eps`` below are the older spelling and win where both are given; per-parameter
        values go through those, e.g. ``fd_eps={'omega_m': 1e-4}``.
    fd_acc : int or dict[str | Variable, int], optional
        FD accuracy order; overrides ``param.fd.acc`` for each FD parameter
        in *order*.  A scalar value applies to all FD parameters; a dict
        gives per-parameter control.  Falls back to ``param.fd.acc``
        (default 2) when absent.
    fd_eps : float or dict[str | Variable, float], optional
        FD step size; overrides ``param.fd.eps`` for each FD parameter in
        *order*.  A scalar value applies to all FD parameters; a dict gives
        per-parameter control.  Falls back to ``param.fd.eps``
        (→ ``param.ref.std()`` → ``1e-5``) when absent.

    Returns
    -------
    callable
        ``(params: dict = None, return_derived: bool = False, **kwargs) -> derivatives``

        The derivative structure depends on the *order* form: a jax array
        (dict form), a (nested) dict keyed by parameter name (int form), or a
        tuple (sequence form).  When ``return_derived=True`` a tuple
        ``(d_return_val, d_derived)`` is returned instead, where each part
        carries that same structure (``d_derived`` is keyed by derived
        parameter name, e.g. ``d_derived[dname][name1][name2]`` for the
        Hessian form).

        *params* defaults to stored parameter values; *kwargs* are merged as
        overrides.  The returned function is compatible with ``jax.jit``.
        For graphs with only JAX parameters it can also be passed to
        ``jax.vmap``; for graphs with FD parameters the outer stencil loop
        runs in Python (vmap is used *internally* for array-valued FD
        parameters).

    Notes
    -----
    For an array-valued parameter of shape ``S``, each derivative level appends
    trailing axes: the Hessian entry ``[name1][name2]`` has trailing axes
    ``S_name1 + S_name2``, including the full cross-element block for
    ``[name][name]`` (matching ``jax.hessian``).  The legacy dict form
    ``{'x': 2}`` instead computes element-wise (diagonal) derivatives only.

    Examples
    --------
    First derivative, then call at default params::

        grad_omega = differentiate(graph, {'omega_m': 1})
        g = grad_omega()

    Override step and accuracy::

        d2 = differentiate(graph, {'omega_m': 2}, fd_acc=4, fd_eps=1e-4)
        v  = d2({'omega_m': 0.3})

    Mixed JAX + FD partial::

        cross = differentiate(graph, {'a': 1, 'x': 1})   # 'a' JAX, 'x' FD
        v = cross()

    Gradient and Hessian over several parameters in one call::

        jac = differentiate(graph, 1, params=['omega_m', 'sigma8'])
        jac()['omega_m']                       # d/dω_m

        hess = differentiate(graph, 2, params=['omega_m', 'sigma8'])
        hess()['omega_m']['sigma8']            # d²/(dω_m dσ₈)

    Batch of mixed partials sharing setup::

        d_fn = differentiate(graph, [{'a': 1}, {'a': 1, 'b': 1}, {'b': 2}])
        da, dab, dbb = d_fn(p0)

    Return value + derived params simultaneously::

        d_all = differentiate(graph, {'omega_m': 1})
        d_rv, d_derived = d_all(return_derived=True)
    """
    # `jacfwd` and `hessian` both route through here, so this serves all three. Building is not
    # free -- it runs every node's __post_init__ and __call__ -- so an already-built graph is
    # taken as is rather than rebuilt.
    graph = graph if isinstance(graph, CompiledGraph) else build(graph)
    # `fd=dict(eps=..., acc=...)`, the same spelling as `Parameter(fd=...)`; the flat `fd_acc=` /
    # `fd_eps=` kwargs still work and win where both are given
    fd = dict(fd or {})
    if fd_acc is None:
        fd_acc = fd.pop('acc', None)
    if fd_eps is None:
        fd_eps = fd.pop('eps', None)
    if fd:
        raise ValueError(f'unknown fd field(s) {sorted(fd)}; expected eps, acc')

    def _resolve_per_param(value, names):
        """Normalise a scalar-or-dict override to a ``{name: value}`` dict.

        A scalar *value* is broadcast to all *names*; a dict is key-normalised.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return {(k.name if isinstance(k, Variable) else str(k)): v for k, v in value.items()}
        return {n: value for n in names}

    def _to_name(key):
        return key.name if isinstance(key, Variable) else str(key)

    # ── normalise the *order* form ─────────────────────────────────────────────
    if isinstance(order, dict):
        form = 'single'
        order_dicts = [{_to_name(key): int(k) for key, k in order.items()}]
        names = list(order_dicts[0])
    elif isinstance(order, (int, np.integer)):
        form = 'tree'
        total_order = int(order)
        if total_order not in (1, 2):
            raise ValueError(f'int order must be 1 (jacobian) or 2 (hessian), got {total_order}; '
                             'pass explicit multi-index dicts for higher orders')
        if params is None:
            names = [p.name for p in graph.params if getattr(p, 'varied', True) and not p.derived]
        else:
            names = [_to_name(key) for key in params]
        if not names:
            raise ValueError('No varied parameter to differentiate with respect to; pass params=[...] explicitly')
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate parameter(s) in params: {names}')
        order_dicts = None
    elif isinstance(order, (list, tuple)):
        form = 'sequence'
        order_dicts = [{_to_name(key): int(k) for key, k in od.items()} for od in order]
        names = sorted({name for od in order_dicts for name in od})
    else:
        raise TypeError(f'order must be a dict, an int, or a sequence of dicts, got {type(order)}')

    if params is not None and form != 'tree':
        raise ValueError('params is only accepted together with an int order')

    # Maximum requested derivative order per parameter (collocation node counts).
    if order_dicts is not None:
        max_fd_order = {}
        for order_dict in order_dicts:
            for order_name, order_k in order_dict.items():
                max_fd_order[order_name] = max(max_fd_order.get(order_name, 0), order_k)
    else:
        max_fd_order = {name: total_order for name in names}

    # ── validate ──────────────────────────────────────────────────────────────
    known = set(graph._fd_names) | set(graph._jax_names)
    bad = set(names) - known
    if bad:
        raise ValueError(f'Unknown parameter(s): {sorted(bad)}')

    # ── split by differentiation strategy ─────────────────────────────────────
    # The graph classifies a parameter as FD only when it reaches a non-traceable node.  A caller
    # may still need FD for a traceable one: a jax-traceable graph can be undifferentiable in
    # *forward* mode -- a custom_vjp (e.g. the ACE cosmology network) defines a reverse rule only,
    # and jacfwd through it raises.  An explicit fd_acc / fd_eps for such a parameter moves it to
    # the FD set; FD is always valid for a traceable parameter, only slower, so this can never
    # make a correct call wrong.
    forced_fd = set()
    for override in (fd_acc, fd_eps):
        if isinstance(override, dict):
            forced_fd |= {name if isinstance(name, str) else getattr(name, 'name', None)
                          for name in override}
    fd_names_sel  = [n for n in names if n in graph._fd_names or n in forced_fd]
    jax_names_sel = [n for n in names if n in graph._jax_names and n not in forced_fd]

    # ── resolve fd_acc / fd_eps overrides for FD params ──────────────────────
    acc_ov = _resolve_per_param(fd_acc, fd_names_sel)
    eps_ov = _resolve_per_param(fd_eps, fd_names_sel)

    def _fd_spec(name, k):
        """Return the :class:`FDSpec` for the order-*k* stencil of FD param *name*."""
        param_obj = graph.params[name]
        fd = param_obj.fd
        # fd_transform=True honors param.fd.transform (derivatives then w.r.t. the
        # transformed variable, fd.eps / fd.center in transformed units); default False
        # keeps plain parameter-space derivatives.
        transform = fd.transform if fd_transform else None
        if fd.limits is not None:
            # Chebyshev collocation: order-n stencil = n + 1 Chebyshev-Lobatto nodes
            # spanning fd.limits (parameter units, mapped through the transform), every
            # derivative taken from the full node set -- the order-n Taylor is then
            # identically the degree-n Chebyshev interpolant over the range.
            lo, hi = fd.limits
            if transform is not None:
                fwd_map = _FD_TRANSFORMS[transform][0]
                lo, hi = float(fwd_map(lo)), float(fwd_map(hi))
            n_nodes = max(max_fd_order.get(name, k), k) + 1
            nodes = chebyshev_lobatto_nodes(n_nodes, limits=(lo, hi))
            return FDSpec(None, None, None, None, transform, nodes)
        eps = eps_ov.get(name, fd.eps)
        if eps is None and getattr(param_obj, 'ref', None) is not None:
            eps = param_obj.ref.std()
        if eps is None or (np.ndim(eps) == 0 and not np.isfinite(eps)):
            eps = 1e-5
        acc = acc_ov.get(name, fd.acc)
        offsets, coeffs = _fd_stencil(k, acc)
        prior_limits = param_obj.prior.limits if param_obj.prior is not None else None
        return FDSpec(offsets, coeffs, eps, prior_limits, transform, None)

    # ── shared evaluation core ─────────────────────────────────────────────────
    # _return_derived is a one-element mutable box shared between _eval and
    # _derivative.  _eval reads it at trace time so that when return_derived=False
    # (the common case) the derivative chain only differentiates val, not
    # derived_dict.  jax.jacfwd re-traces on every call so changing the flag
    # between calls is safe.
    _return_derived = [False]
    _call_fn = graph._jit_call_fn if jit else graph._call_fn

    def _eval(p_dict):
        fd_t  = tuple(jnp.asarray(p_dict[n]) for n in graph._fd_names)
        jax_t = tuple(jnp.asarray(p_dict[n]) for n in graph._jax_names)
        val, derived_dict, _ = _call_fn(fd_t, jax_t)
        if _return_derived[0]:
            return val, derived_dict
        return val

    # The strategy split, not the graph's own classification: a parameter the caller forced to FD
    # must nest an FD stencil here too, otherwise this path silently rebuilds the jacfwd chain the
    # override exists to avoid.
    fd_set, jax_set = set(fd_names_sel), set(jax_names_sel)

    def _build_chain(order_dict):
        """One mixed partial: JAX jacfwd nests inner, direct FD stencils outer."""
        fn = _eval
        for name, k in order_dict.items():
            if name in jax_set and k > 0:
                # one `jax.jacfwd` pass per order: nesting k times gives the k-th derivative
                for _ in range(k):
                    def fn(p_dict, _fn=fn, _name=name):
                        return jax.jacfwd(lambda value: _fn({**p_dict, _name: value}))(p_dict[_name])
        for name, k in order_dict.items():
            if name in fd_set and k > 0:
                fn = _fd_wrap(fn, name, _fd_spec(name, k), k)
        return fn

    # ── build the derivative function chain(s) once ────────────────────────────
    if form in ('single', 'sequence'):
        signatures = [tuple(od.items()) for od in order_dicts]
        chains = {}
        for signature, order_dict in zip(signatures, order_dicts):
            if signature not in chains:
                chains[signature] = _build_chain(order_dict)

        def _run(p0):
            results = {signature: chain(p0) for signature, chain in chains.items()}
            if form == 'single':
                return results[signatures[0]]
            entries = tuple(results[signature] for signature in signatures)
            if _return_derived[0]:
                return tuple(entry[0] for entry in entries), tuple(entry[1] for entry in entries)
            return entries

    else:  # form == 'tree': all partials of total order 1 (jacobian) or 2 (hessian)
        param_ndims = {name: len(graph.params[name].shape or ()) for name in names}

        def _permute_param_axes(entry, src_names, dst_names):
            """Reorder the trailing parameter axis blocks of every leaf of *entry*
            from *src_names* order to *dst_names* order (a permutation of it)."""
            if list(src_names) == list(dst_names) or all(param_ndims[name] == 0 for name in src_names):
                return entry
            total_axes = sum(param_ndims[name] for name in src_names)
            starts = {}
            position = 0
            for name in src_names:
                starts[name] = position
                position += param_ndims[name]

            def _per_leaf(leaf):
                lead = leaf.ndim - total_axes
                perm = list(range(lead))
                for name in dst_names:
                    perm.extend(range(lead + starts[name], lead + starts[name] + param_ndims[name]))
                return jnp.transpose(leaf, perm)

            return jax.tree_util.tree_map(_per_leaf, entry)

        def _extract(res, getter):
            """Apply *getter* (indexing into the ``{name: ...}`` dicts grafted by
            jacfwd) to both the return-value part and each derived entry of a
            chain result."""
            if _return_derived[0]:
                d_val, d_derived = res
                return getter(d_val), {dn: getter(d) for dn, d in d_derived.items()}
            return getter(res)

        # Pure-JAX block: one (nested) jacfwd pass over the dict of JAX params —
        # a single trace yields the whole gradient / Hessian block at once.
        jacfwd_inner = _jacfwd_dict_wrap(_eval, jax_names_sel) if jax_names_sel else None
        chain_jax = None
        if jax_names_sel:
            chain_jax = jacfwd_inner if total_order == 1 else _jacfwd_dict_wrap(jacfwd_inner, jax_names_sel)

        fd_chains = {}     # order 1: fd name -> chain
        cross_chains = {}  # order 2: fd name -> FD stencil over the JAX gradient dict (all (jax, fd) cross terms at once)
        fdfd_chains = {}   # order 2: (p, q) with p <= q in `names` order -> chain; axes appended (q, p)

        if total_order == 1:
            for name in fd_names_sel:
                fd_chains[name] = _fd_wrap(_eval, name, _fd_spec(name, 1), 1)
        else:
            for name in fd_names_sel:
                if jax_names_sel:
                    cross_chains[name] = _fd_wrap(jacfwd_inner, name, _fd_spec(name, 1), 1)
            for idx_p, name_p in enumerate(fd_names_sel):
                for name_q in fd_names_sel[idx_p:]:
                    if name_p == name_q and not graph.params[name_p].shape:
                        # Scalar diagonal: direct order-2 stencil (fewer evaluations).
                        fdfd_chains[(name_p, name_q)] = _fd_wrap(_eval, name_p, _fd_spec(name_p, 2), 2)
                    elif name_p == name_q:
                        # Array diagonal: nested order-1 stencils give the full
                        # cross-element block.  The inner stencil gets no prior
                        # limits and the outer stencil's limits are shrunk by the
                        # inner reach, so all nested points stay in bounds and the
                        # inner base is never shifted (a shift there would distort
                        # the outer stencil grid).
                        spec = _fd_spec(name_p, 1)
                        inner = _fd_wrap(_eval, name_p, spec, 1, prior_limits=None)
                        eps_below, eps_above, _ = _fd_parse_eps(spec.eps)
                        margin = (len(spec.offsets) // 2) * max(eps_below, eps_above)
                        outer_limits = None
                        if spec.prior_limits is not None:
                            lo, hi = spec.prior_limits
                            outer_limits = (lo + margin if np.isfinite(lo) else lo,
                                            hi - margin if np.isfinite(hi) else hi)
                        fdfd_chains[(name_p, name_q)] = _fd_wrap(inner, name_p, spec, 1, prior_limits=outer_limits)
                    else:
                        inner = _fd_wrap(_eval, name_q, _fd_spec(name_q, 1), 1)
                        fdfd_chains[(name_p, name_q)] = _fd_wrap(inner, name_p, _fd_spec(name_p, 1), 1)

        def _run(p0):
            entries = {}  # ordered name tuple (n1,) or (n1, n2) -> derivative entry, axes in tuple order
            if total_order == 1:
                if chain_jax is not None:
                    res = chain_jax(p0)
                    for name in jax_names_sel:
                        entries[(name,)] = _extract(res, lambda t, a=name: t[a])
                for name, chain in fd_chains.items():
                    entries[(name,)] = chain(p0)
            else:
                if chain_jax is not None:
                    res = chain_jax(p0)
                    for name_1 in jax_names_sel:
                        for name_2 in jax_names_sel:
                            entries[(name_1, name_2)] = _extract(res, lambda t, a=name_1, b=name_2: t[a][b])
                for name_q, chain in cross_chains.items():
                    res = chain(p0)  # axes: (jax name, name_q)
                    for name in jax_names_sel:
                        entry = _extract(res, lambda t, a=name: t[a])
                        entries[(name, name_q)] = entry
                        entries[(name_q, name)] = _permute_param_axes(entry, [name, name_q], [name_q, name])
                for (name_p, name_q), chain in fdfd_chains.items():
                    entry = chain(p0)  # axes: (name_q, name_p)
                    entries[(name_q, name_p)] = entry
                    if name_p != name_q:
                        entries[(name_p, name_q)] = _permute_param_axes(entry, [name_q, name_p], [name_p, name_q])

            def _nest(getter):
                if total_order == 1:
                    return {name: getter(entries[(name,)]) for name in names}
                return {name_1: {name_2: getter(entries[(name_1, name_2)]) for name_2 in names} for name_1 in names}

            if _return_derived[0]:
                derived_names = list(next(iter(entries.values()))[1])
                return _nest(lambda entry: entry[0]), {dn: _nest(lambda entry, a=dn: entry[1][a]) for dn in derived_names}
            return _nest(lambda entry: entry)

    # ── returned callable ──────────────────────────────────────────────────────
    def _derivative(params=None, return_derived=False, **kwargs):
        # Use current param values as defaults (supports in-place mutation after
        # build).  After the call, restore only *input* param values so that
        # FD mutations from pure_callback do not corrupt subsequent default calls.
        all_saved = {p.name: p._value for p in graph.params}

        p0 = dict(all_saved)
        # Same hole as the graph's own __call__: an unknown name would be added to p0 and never
        # read again, leaving the parameter at its default.
        if params is not None:
            graph._check_names(params)
            p0.update(params)
        if kwargs:
            graph._check_names(kwargs)
        p0.update(kwargs)
        _return_derived[0] = return_derived
        try:
            return _run(p0)
        finally:
            # Every param, not just the inputs, and whether or not this was traced: a stencil
            # steps the parameter it differentiates (and a traced one leaves tracers), so a
            # derivative has to leave the point it was asked about. Without this a bare
            # `graph()` afterwards would evaluate at the last stencil node instead of at p0.
            for p in graph.params:
                p._value = all_saved[p.name]

    return _derivative


def jacfwd(graph, params=None, fd=None, fd_acc=None, fd_eps=None, jit=False):
    """First derivatives of *graph* w.r.t. several parameters in one call.

    Analogous to ``jax.jacfwd`` on a dict input: returns a function whose value
    is ``{name: d graph / d name}``.  Shorthand for
    ``differentiate(graph, 1, params=params, ...)``; see :func:`differentiate`
    for the argument and return-value semantics.

    Examples
    --------
    ::

        jac = jacfwd(graph, params=['omega_m', 'sigma8'])
        jac()['omega_m']            # d graph / dω_m at default params
    """
    return differentiate(graph, 1, params=params, fd=fd, fd_acc=fd_acc, fd_eps=fd_eps, jit=jit)


def hessian(graph, params=None, fd=None, fd_acc=None, fd_eps=None, jit=False):
    """Second derivatives (Hessian) of *graph* w.r.t. several parameters in one call.

    Analogous to ``jax.hessian`` on a dict input: returns a function whose value
    is the nested dict ``{name1: {name2: d² graph / dname1 dname2}}``, with
    symmetric entries computed once.  Shorthand for
    ``differentiate(graph, 2, params=params, ...)``; see :func:`differentiate`
    for the argument and return-value semantics.

    Examples
    --------
    ::

        hess = hessian(graph, params=['omega_m', 'sigma8'])
        hess()['omega_m']['sigma8']   # d² graph / (dω_m dσ₈) at default params
    """
    return differentiate(graph, 2, params=params, fd=fd, fd_acc=fd_acc, fd_eps=fd_eps, jit=jit)


@default_mpicomm
def pmap(fn, backend='mpi_and_jax', mpicomm=None):
    """Return a batched version of *fn*, distributed across MPI ranks and/or local JAX devices.

    Like :func:`jax.vmap`, ``pmap(fn)`` returns ``mapped(*args)`` that maps *fn* over the
    **leading (batch) axis 0** of every array leaf of its pytree arguments, and returns
    *fn*'s output as a pytree with the same leading batch axis.  On top of ``vmap`` it adds
    distribution: the batch is split across MPI ranks (outer) and local JAX devices (inner).

    * ``'jax'``         — shard the full batch across local JAX devices via
      :func:`jax.experimental.shard_map`; each device runs :func:`jax.vmap` on its slice.
    * ``'mpi'``         — split the batch across MPI ranks; each rank runs ``jax.vmap`` on a
      single device, then outputs are gathered (Allgatherv) so every rank holds the full result.
    * ``'mpi_and_jax'`` *(default)* — MPI outer loop + JAX inner loop: each rank takes a
      contiguous slice and fans it out across its local devices via ``shard_map``.

    Sub-batches whose size is not a multiple of the local device count are zero-padded before
    sharding; the padding is stripped from the output, so it never affects the result.  On a
    single-device, single-rank machine all three backends reduce to a plain ``vmap``.

    Parameters
    ----------
    fn : callable
        Function ``fn(*unbatched_args) -> output``.  Both inputs and output may be arbitrary
        pytrees; every array leaf of ``*args`` must share the same leading batch size ``N``.
    backend : {'jax', 'mpi', 'mpi_and_jax'}, optional
        Parallelism strategy.  Default is ``'mpi_and_jax'``.
    mpicomm : MPI communicator, optional
        Communicator for the ``'mpi'`` / ``'mpi_and_jax'`` backends.  Defaults to
        :func:`desilike.distributed.get_mpicomm`.

    Returns
    -------
    callable
        ``mapped(*args) -> output`` where every array leaf of ``args`` is batched on axis 0,
        and the returned pytree carries the batch on axis 0.  For the MPI backends the result
        is identical on all ranks (Allgatherv semantics).

    Examples
    --------
    ::

        eval_batch = pmap(lambda x: {'sq': x**2, 'sum': jnp.sum(x)})
        out = eval_batch(jnp.arange(1000.))   # out['sq'].shape == (1000,), out['sum'].shape == (1000,)

        # Vectorise a compiled likelihood over a batch of parameter dicts:
        pipe = build(my_likelihood)
        logpdf = pmap(pipe)({'omega_m': jnp.linspace(0.2, 0.4, 1000)})
    """
    _VALID_BACKENDS = ('jax', 'mpi', 'mpi_and_jax')
    if backend not in _VALID_BACKENDS:
        raise ValueError(f'pmap(): backend must be one of {_VALID_BACKENDS}, got {backend!r}')

    try:  # jax >= 0.8.0
        from jax import shard_map
        _shard_map_kwargs = {'check_vma': False}
    except ImportError:  # older jax
        from jax.experimental.shard_map import shard_map
        _shard_map_kwargs = {'check_rep': False}
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    from jax import tree_util as jtu

    devices     = jax.local_devices()
    ndevices    = len(devices)
    mesh        = Mesh(np.array(devices), ('batch',))
    batch_shard = NamedSharding(mesh, P('batch'))

    vfn     = jax.vmap(fn)
    vfn_jit = jax.jit(vfn)

    # Cache the (jitted shard_map'd fn, output treedef) keyed on the per-element input
    # structure (treedef + each leaf's non-batch shape and dtype) so the wrappers are built
    # once and reused across calls / batch sizes (jax.jit then caches compilation by shape).
    _sharded_cache = {}

    def _get_sharded(sig):
        if sig not in _sharded_cache:
            treedef, leafsig = sig
            dummy = [jax.ShapeDtypeStruct((1,) + shape, np.dtype(dt)) for shape, dt in leafsig]
            dummy_args = jtu.tree_unflatten(treedef, dummy)
            out_struct = jax.eval_shape(vfn, *dummy_args)
            in_specs  = jtu.tree_unflatten(treedef, [P('batch')] * len(leafsig))
            out_specs = jtu.tree_map(lambda _: P('batch'), out_struct)
            sharded = jax.jit(shard_map(vfn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **_shard_map_kwargs))
            _sharded_cache[sig] = (sharded, jtu.tree_structure(out_struct))
        return _sharded_cache[sig]

    def _run_sharded(sub_args, sub_n, sig):
        """Run *sub_args* (batch ``sub_n``) through shard_map, zero-padding to a device multiple."""
        if sub_n == 0:
            return vfn_jit(*sub_args)   # empty batch: plain vmap preserves the output structure
        sharded, _ = _get_sharded(sig)
        remainder = sub_n % ndevices
        if remainder:
            pad_n = ndevices - remainder
            sub_args = jtu.tree_map(
                lambda x: jnp.concatenate([x, jnp.zeros((pad_n,) + x.shape[1:], dtype=x.dtype)], axis=0),
                sub_args)
        sub_args = jtu.tree_map(lambda x: jax.device_put(x, batch_shard), sub_args)
        out = sharded(*sub_args) if isinstance(sub_args, tuple) else sharded(sub_args)
        return jtu.tree_map(lambda x: x[:sub_n], out)

    def mapped(*args):
        leaves, in_treedef = jtu.tree_flatten(args)
        if not leaves:
            raise ValueError('pmap(): no array arguments to map over')
        leaves = [jnp.asarray(leaf) for leaf in leaves]
        batch_size = int(leaves[0].shape[0])
        if any(int(leaf.shape[0]) != batch_size for leaf in leaves):
            raise ValueError('pmap(): all batched leaves must share the same leading (batch) axis size')
        args = jtu.tree_unflatten(in_treedef, leaves)
        sig = (in_treedef, tuple((tuple(leaf.shape[1:]), leaf.dtype.str) for leaf in leaves))

        if backend == 'jax':
            return _run_sharded(args, batch_size, sig)

        # 'mpi' / 'mpi_and_jax': split the batch across ranks, run locally, allgather.
        rank, nranks = mpicomm.rank, mpicomm.size
        local_start = rank * batch_size // nranks
        local_stop  = (rank + 1) * batch_size // nranks
        local_n     = local_stop - local_start
        local_args  = jtu.tree_map(lambda x: x[local_start:local_stop], args)

        if backend == 'mpi_and_jax':
            local_out = _run_sharded(local_args, local_n, sig)
        else:  # 'mpi'
            local_out = vfn_jit(*local_args)

        out_leaves, out_treedef = jtu.tree_flatten(local_out)
        gathered = [jnp.asarray(_mpi_gather(np.asarray(leaf), mpiroot=Ellipsis, mpicomm=mpicomm))
                    for leaf in out_leaves]
        return jtu.tree_unflatten(out_treedef, gathered)

    return mapped
