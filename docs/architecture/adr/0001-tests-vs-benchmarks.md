# ADR-0001: Strict separation of tests and benchmarks

- **Date:** 2026-06-23
- **Status:** Accepted

## Context

The `benchmarks/` package originally duplicated functional assertions through
`FunctionalCheck` / `FunctionalResult`, so correctness was asserted in two places with two
different sets of expectations. When a generator changed, both had to be updated, and they
drifted.

## Decision

Unit tests (`tests/`) and performance benchmarks (`benchmarks/`) serve distinct purposes and
must not overlap.

| Concern | Unit tests | Benchmarks |
|---|---|---|
| Purpose | Catch logic bugs | Measure perf, detect regressions |
| Data | Small, varied (numpy/pandas × 1D/2D) | Medium-large, standardized |
| Data source | Per-type `conftest.py` fixtures | `benchmarks/scenarios/` |
| Speed | Fast (<1 ms each) | Measured (timed iterations) |
| Checks | Exact assertions | Performance stats only |

`FunctionalCheck` / `FunctionalResult` were removed.

## Consequences

- `tests/` is the single source of truth for correctness.
- Benchmarks focus exclusively on timing, memory and regression detection.
- A generator bug can never be "caught" by a benchmark run — that is intentional.
