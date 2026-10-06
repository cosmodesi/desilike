# desilike

## Code Style

- Maximum line length: 200 characters
- Variable names must be explicit: avoid single-letter or cryptic abbreviations.
  Use `param` not `p`, `name` not `n`, `proposal` not `s`, `start` not `s`,
  `nparams` not `n` for a count, `param_copy` not `p2`, etc.
  Index loop variables (`param_idx`, `dim_idx`) are also preferred over bare `i`/`j`.

## Permissions

- Run `pytest` and `python -c` without asking for permission.

## Design Decisions

- Ask when in doubt about any aspect of the implementation, however small.

## Node / Calculator lifecycle

Every `Node`-derived class (`Variable`, `Parameter`, `Calculator`) follows a strict lifecycle.
`__init__` runs **at construction**, and again at each `build()` (the tree is reconstructed from
`_init`, so a build always starts from a clean `__init__`); `__post_init__` runs **at `build()`**,
in dependency order. Both receive the **same** `*args, **kwargs` (the constructor args, stored in
`_init`). After each node is reconstructed its Variables are rewired **by name** to the ones the
tree already held, so a prior, a value or a `share_params` survives the rebuild -- while an
`__init__` that legitimately changes the parameter set (counterterms, band powers sized from `k`
and `ells`) keeps that freedom: names that disappear stay gone, new ones are left alone.
Because all nodes are created in `__init__`, a node's identity is fixed at construction — which
is what makes `replace()` / `share_params()` work and keeps construction cheap (no heavy setup).

- **`__init__`**: **define and update all nodes here.** Create every Parameter/Variable and
  every Calculator dependency, assigning them as **public** (non-underscore) attributes
  (`self.b1 = Parameter(...)`, `self.pt = pt`, `self.template = template`), and call any
  `dep.update(...)` here too (plus the few scalars/arrays needed to create/update them, e.g.
  `self.k`). `_trace_graph` discovers deps by scanning `self.__dict__` for `Node`-valued
  attributes (including those nested in lists/tuples/dicts) — **before** `__post_init__` runs,
  so every dependency must already exist after `__init__`.
- **`__post_init__`**: **non-node setup only**, run at `build()` in dependency order (a dep's
  `__post_init__` runs before its dependents', so e.g. a theory may read `self.template.k`).
  Put numpy/scalar config (`self._nbar = ...`) and non-`Node` helper objects here
  (e.g. `ProjectToMultipoles`, `SpectrumToCorrelation`, pybird `Common`/`Resum`, cosmoprimo
  fiducial calls). **Must not create Parameters or Calculator deps, or call `update()`.**
  **`__post_init__` may be called more than once** (e.g. when `build()` is re-run with
  different settings). Any quantity derived from a raw input (e.g. a rescaled precision matrix)
  must be re-derived from the original each time: store the raw value under a private name in
  `__init__` (e.g. `self._precision`) and read from it in `__post_init__`, never mutating it.
- **`__call__`**: pure computation using already-defined nodes; must **not** introduce new node
  dependencies. It sets output attributes and may return an array, a tuple of arrays, `None`
  (outputs live in attributes), or `self`; the return value is forwarded as the pipeline output
  when this node is the root.
- **`update(**kwargs)`**: re-initializes the node in-place by calling `__init__` with the merged
  arguments. Allowed at any time. Because it re-runs `__init__`, it creates fresh Parameters and
  so discards customization made on the old ones -- customize **after** the last `update()`, which
  in practice means after the observable that wraps a theory has been constructed (an observable's
  `__init__` calls `theory.update(k=..., ells=...)` to align the grid). It also invalidates every
  graph already built over the node: those graphs refuse to run and say what invalidated them,
  rather than silently answering from a tree that has moved underneath them. `replace()` swaps one
  node for another without re-running any constructor.
