# desilike

## Code Style

- Maximum line length: 200 characters

## Design Decisions

- Ask before making any non-trivial design choice. When in doubt about complexity, ask first instead of over-engineering.
- Ask when in doubt about any aspect of the implementation, however small.
- Calculator dependencies are set exclusively inside `init()`, never in `__init__`. `__init__` only stores args/kwargs for lazy initialization. (`init` : `call` :: `__init__` : `__call__`)
