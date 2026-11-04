# Architecture decision records

Each record captures one decision, the context that forced it, and what it costs. Records are
**append-only**: a decision is never edited, it is superseded by a new record.

Format is MADR-lite — Decision / Context / Consequences. No status lifecycle beyond
`Accepted` and `Superseded by NNNN`.

## Index

| ADR | Title | Date | Status |
|---|---|---|---|
| [0001](0001-tests-vs-benchmarks.md) | Strict separation of tests and benchmarks | 2026-06-23 | Accepted |
| [0002](0002-per-data-type-fixtures.md) | Per-data-type pytest fixture architecture | 2026-06-23 | Accepted |
| [0003](0003-benchmark-auto-registration.md) | Benchmark registration via auto-discovered modules | 2026-06-24 | Accepted |
| [0004](0004-vectorized-numpy.md) | Vectorized NumPy over Python loops | 2026-06-26 | Accepted |
| [0005](0005-baseline-regression-detection.md) | Baseline-based regression detection | 2026-06-24 | Accepted |
| [0006](0006-shared-ood-samplers.md) | Outlier value synthesis owned by shared OOD samplers | 2026-10-25 | Accepted |
| [0007](0007-import-contracts.md) | Enforce layering with import contracts | 2026-11-02 | Accepted |

## Writing a new record

1. Copy the shape of an existing file; number it `NNNN-kebab-title.md`.
2. Add a row to the index above.
3. Keep it under ~40 lines. If it needs more, it is a design document, not a decision.
4. If it replaces an earlier decision, mark the old one `Superseded by NNNN` and link forward.
