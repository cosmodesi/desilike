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
`__init__` runs **at construction**; `__post_init__` runs **at `compile()`** (in dependency
order). Both receive the **same** `*args, **kwargs` (the constructor args, stored in `_init`).
Because all nodes are created in `__init__`, a node's identity is fixed at construction — which
is what makes `replace()` / `share_params()` work and keeps construction cheap (no heavy setup).

- **`__init__`**: **define and update all nodes here.** Create every Parameter/Variable and
  every Calculator dependency, assigning them as **public** (non-underscore) attributes
  (`self.b1 = Parameter(...)`, `self.pt = pt`, `self.template = template`), and call any
  `dep.update(...)` here too (plus the few scalars/arrays needed to create/update them, e.g.
  `self.k`). `build_graph` discovers deps by scanning `self.__dict__` for `Node`-valued
  attributes (including those nested in lists/tuples/dicts) — **before** `__post_init__` runs,
  so every dependency must already exist after `__init__`.
- **`__post_init__`**: **non-node setup only**, run at `compile()` in dependency order (a dep's
  `__post_init__` runs before its dependents', so e.g. a theory may read `self.template.k`).
  Put numpy/scalar config (`self._nbar = ...`) and non-`Node` helper objects here
  (e.g. `ProjectToMultipoles`, `SpectrumToCorrelation`, pybird `Common`/`Resum`, cosmoprimo
  fiducial calls). **Must not create Parameters or Calculator deps, or call `update()`.**
- **`__call__`**: pure computation using already-defined nodes; must **not** introduce new node
  dependencies. It sets output attributes and may return an array, a tuple of arrays, `None`
  (outputs live in attributes), or `self`; the return value is forwarded as the pipeline output
  when this node is the root.
- **`update(**kwargs)`**: re-initializes the node in-place by calling `__init__` with the merged
  arguments. **Only allowed during construction** (i.e. from within a parent's `__init__`);
  calling it on an already-constructed node raises. To change a constructed node, reconstruct it
  or use `replace()`.
