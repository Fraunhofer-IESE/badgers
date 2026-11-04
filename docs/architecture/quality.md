# Quality goals

Architecture is only useful if "good" is testable. These are the quality scenarios this
project optimises for, each with the mechanism that keeps it true.

| ID | Scenario | Mechanism |
|---|---|---|
| **Q1** | Adding a generator touches at most its own package plus one `__init__.py` — never `badgers/core/`. | Review; the four contracts on [Building blocks](building-blocks.md) |
| **Q2** | A new out-of-distribution strategy is a new *sampler*, not a new generator class. | [ADR-6](adr/0006-shared-ood-samplers.md) |
| **Q3** | No generator imports a DGP backend. | `import-linter` contract |
| **Q4** | `badgers.core` never imports `badgers.generators`. | `import-linter` contract |
| **Q5** | `badgers` never imports `benchmarks`. | `import-linter` contract |
| **Q6** | Importing `badgers.core.dgp` never pulls `pymc`, `pytensor` or `dowhy`. | `tests/core/test_import_hygiene.py` |
| **Q7** | Every generator accepts numpy and pandas, 1D and 2D where applicable. | Parametrized fixtures in `tests/generators/*/conftest.py` |
| **Q8** | Generators are vectorized: no Python loop over `n_samples`. | [ADR-4](adr/0004-vectorized-numpy.md) + benchmarks |
| **Q9** | A performance regression is caught before merge. | [ADR-5](adr/0005-baseline-regression-detection.md) |
| **Q10** | Correctness is tested in exactly one place. | [ADR-1](adr/0001-tests-vs-benchmarks.md) |

## Non-goals

Explicitly *not* optimised for, so that nobody "fixes" them:

- **scikit-learn API compatibility.** `Pipeline` is not an `sklearn` transformer; `generate`
  is not `fit`/`transform`.
- **Streaming / out-of-core data.** Everything is in-memory numpy.
- **Determinism without a seed.** Generators are deterministic *given* an RNG, never by
  default.
- **Backend parity.** The DoWhy and PyMC backends expose the same protocol, not the same
  numerical results.

## Trade-offs we accepted

| Trade-off | Why |
|---|---|
| `generate(X, **params)` returns one object instead of `(X, y)` | Keeps `Pipeline` modality-agnostic; targets live inside `X`. |
| Selectors have no base class and no registry | The contract is the call signature; a registry would add indirection for no gain. |
| DGPs are immutable and rngless | Makes cascades reproducible and prevents hidden state; costs a `pick_parameters` hook. |
| Structure edits are free functions, not DGP methods | Keeps the DGP protocol to one operation (`fit_node`); costs a slightly unusual API. |
| `benchmarks/` lives outside the package | Guarantees the library can never depend on it (enforced by contract). |
