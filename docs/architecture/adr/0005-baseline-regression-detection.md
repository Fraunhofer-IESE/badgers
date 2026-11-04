# ADR-0005: Baseline-based regression detection

- **Date:** 2026-06-24
- **Status:** Accepted

## Context

Performance work (see [ADR-0004](0004-vectorized-numpy.md)) needs a way to prove that an
optimization did not regress another generator. Absolute timings are machine-dependent, so a
fixed threshold is meaningless.

## Decision

Benchmarks support a baseline workflow with defined thresholds:

```bash
python -m benchmarks run --iterations 10
python -m benchmarks baseline save --name v0.0.13
python -m benchmarks compare --baseline v0.0.13
```

**Regression thresholds:** a >20 % time increase or a >30 % memory increase flags a regression
in `comparator.py`.

## Consequences

- Baselines are versioned snapshots of generator performance, enabling automated detection of
  regressions before merging.
- Provides a quantitative gate for optimization work.
- Baselines are local-only (`.benchmarks/baselines/` is gitignored), so the gate is advisory
  rather than enforced in CI.
