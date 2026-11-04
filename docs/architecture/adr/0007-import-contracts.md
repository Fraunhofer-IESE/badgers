# ADR-0007: Enforce layering with import contracts

- **Date:** 2026-11-02
- **Status:** Accepted

## Context

The architecture documentation described a layering (`core` below `generators`, generators
backend-blind, `benchmarks/` a pure consumer) that nothing verified. The previous
`docs/architecture.md` had already drifted from the code: it listed flat modules that had since
become packages, omitted `core/dgp`, `core/selectors`, `core/sampling` and
`generators/tabular_data/causal`, and referenced a `docs/superpowers/` directory that no longer
existed.

Prose cannot keep a layering true. Either the layering is checked, or it is a wish.

## Decision

Express the layering as `import-linter` contracts in `pyproject.toml` and run `lint-imports` in
CI alongside the tests.

```toml
[tool.importlinter]
root_package = "badgers"
include_external_packages = true

[[tool.importlinter.contracts]]
name = "core must not depend on generators"
type = "forbidden"
source_modules = ["badgers.core"]
forbidden_modules = ["badgers.generators"]

[[tool.importlinter.contracts]]
name = "generators must not import DGP backends"
type = "forbidden"
source_modules = ["badgers.generators"]
forbidden_modules = ["badgers.core.dgp.backends"]

[[tool.importlinter.contracts]]
name = "core.dgp must not import generators"
type = "forbidden"
source_modules = ["badgers.core.dgp"]
forbidden_modules = ["badgers.generators"]

[[tool.importlinter.contracts]]
name = "the library must not import the benchmarks package"
type = "forbidden"
source_modules = ["badgers"]
forbidden_modules = ["benchmarks"]

[[tool.importlinter.contracts]]
name = "Layered architecture: generators -> dgp -> selectors -> utils"
type = "layers"
layers = [
    "badgers.generators",
    "badgers.core.dgp",
    "badgers.core.selectors",
    "badgers.core.utils",
]
```

## Consequences

- The structural claims in [Building blocks](../building-blocks.md) cannot silently rot: a
  violating import fails the build.
- The documentation is allowed to be short, because the structural half is machine-checked.
- One new dev dependency (`import-linter`) and one CI step.
- The `layers` contract is coarser than the real design: `badgers.core.dgp.backends` cannot be
  its own layer because `import-linter` rejects a layer whose modules are also descendants of
  another layer's module. The backend rule is therefore expressed as a separate `forbidden`
  contract instead.
- Import hygiene for the *optional* heavy dependencies (`pymc`, `pytensor`, `dowhy`) is not
  expressible as a contract and stays a test
  (`tests/core/test_import_hygiene.py`).
