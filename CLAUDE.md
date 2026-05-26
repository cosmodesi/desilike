# desilike

## Code Style

- Maximum line length: 200 characters

## Permissions

- Run `pytest` and `python -c` without asking for permission.

## Design Decisions

- Ask before making any non-trivial design choice. When in doubt about complexity, ask first instead of over-engineering.
- Ask when in doubt about any aspect of the implementation, however small.

## Node / Calculator lifecycle

Every `Node`-derived class (`Variable`, `Parameter`, `Calculator`) follows a strict lifecycle:

- **`__init__`**: stores args/kwargs only; no computation, no dependency wiring.
- **`__post_init__`**: called at `compile()` time; must define **all** node dependencies by assigning them as **public** (non-underscore) attributes (`self.cosmo = cosmo`, `self.omega_m = omega_m`). After `__post_init__` returns, `build_graph` scans `self.__dict__` for `Node`-valued attributes to discover deps. No new Calculator dependencies may be introduced after `__post_init__` returns.
- **`__call__`**: executes the computation using already-defined nodes; must **not** introduce new node dependencies.
- **`update(**kwargs)`**: re-initializes the node in-place by calling `__init__` with the merged arguments, then sets `_updated = True`. Any `update()` call on a node that belongs to a compiled pipeline marks that pipeline as stale; `compile()` must be called again before the pipeline can execute.
