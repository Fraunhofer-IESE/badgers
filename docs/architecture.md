# Architecture

This section describes how `badgers` is structured, why it is structured that way, and which of
those claims are enforced automatically.

It follows a trimmed [arc42](https://arc42.org/) outline — only the sections that carry
information for a single-process Python library — plus a [C4](https://c4model.com/) level-1
context diagram and a log of architecture decision records (ADRs).

## How to read this section

| Page | Question it answers | Changes how often |
|---|---|---|
| [System context](architecture/context.md) | Who uses badgers, and what does it depend on? | Rarely |
| [Building blocks](architecture/building-blocks.md) | Which modules exist, and which may depend on which? | When a package is added |
| [Runtime scenarios](architecture/runtime.md) | What happens during a `generate()` call? | When a flow changes |
| [Quality goals](architecture/quality.md) | What does "good architecture" mean here, testably? | Rarely |
| [Causal defect family](architecture/causal-design.md) | How do the causal generators, DGPs and backends fit together? | When the causal API changes |
| [Decisions](architecture/adr/index.md) | Why is it like this, and what was rejected? | Append-only |

## The one-paragraph version

`badgers` is a library of **generators**: objects exposing `generate(X, **params) -> Xt` that
augment data with quality deficits (outliers, missing values, noise, drift, imbalance, typos,
…). Everything modality-specific (e.g., tabular_data, time.series, text, etc.) lives under `badgers.generators`; everything reusable lives
under `badgers.core`. The core is split into four independent concerns — the generator
contract, the *selector* contract (which rows get a deficit), the *sampler* contract (what
values replace them), and the *DGP* contract (how a causal model propagates a deficit). A
generator composes these; it never reimplements them.

## What is enforced, not just documented

Architecture documents rot. The structural claims on these pages are therefore split into two
halves: a narrative half (this section) and an **enforced half** — import contracts checked by
[`import-linter`](https://import-linter.readthedocs.io/) in CI. If the code ever contradicts the
[building blocks](architecture/building-blocks.md) page, the build fails.

```bash
# run the contracts locally
lint-imports
```

The contracts live in `pyproject.toml` under `[tool.importlinter]`; see
[ADR-0007](architecture/adr/0007-import-contracts.md) for the full list and the reasoning.

| Contract | Meaning |
|---|---|
| `core must not depend on generators` | The foundation never imports a concrete generator. |
| `generators must not import DGP backends` | Generators are backend-blind; backends are injected. |
| `core.dgp must not import generators` | The DGP protocol does not know about deficits. |
| `the library must not import the benchmarks package` | `benchmarks/` is a consumer, never a dependency. |
| `Layered architecture: generators -> dgp -> selectors -> utils` | Dependency direction is one-way. |

Import hygiene is additionally pinned by `tests/core/test_import_hygiene.py`, which asserts that
importing `badgers.core.dgp` never pulls `pymc`, `pytensor` or `dowhy` into `sys.modules`.

## Conventions used on these pages

- **Module paths** are written as `badgers.core.selectors`, not as file paths.
- **Diagrams** are Mermaid and are rendered by mkdocs-material; they are hand-written and
  deliberately few, because hand-written diagrams are the part that rots fastest.
- **API-level detail** is *not* duplicated here — it is generated from docstrings into the
  [API reference](reference/badgers/index.md). If you want to know what a class does, read
  the API reference; if you want to know why it exists, read this section.
