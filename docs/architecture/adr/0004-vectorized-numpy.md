# ADR-0004: Vectorized NumPy over Python loops

- **Date:** 2026-06-26
- **Status:** Accepted

## Context

Generators often produce many samples at once. Per-row RNG calls and per-column pandas `.iloc`
accesses inside Python loops were orders of magnitude slower than batched NumPy operations. A
concrete case: `random_spherical_coordinate` used an O(d²) `np.prod(sin_phis[:i])` pattern.

## Decision

All generator implementations must prefer batched/vectorized NumPy operations over Python-level
`for` loops, especially when `n_samples` can be large.

Three canonical patterns:

1. **Per-row RNG → batched RNG.** Replace `[rng.exponential(size=d) for _ in range(n)]` with
   `rng.exponential(size=(n, d))`.
2. **Per-column `.iloc` / `.choice` → integer array indexing.** Replace
   `np.stack([rng.choice(X.iloc[:, i], ...) for i in range(d)])` with
   `X.values[row_idx, np.arange(d)]`.
3. **Batch variants for utilities.** When a utility operates on a single sample (e.g.
   `random_spherical_coordinate`), also provide a batch version
   (`random_spherical_coordinates`) accepting arrays with `axis=` parameters.

**When not to vectorize:** small `n_samples` (≤ 10), and algorithms requiring sequential state
(e.g. rejection sampling with an unknown iteration count).

## Consequences

- Large speedups on the common path.
- Two APIs to maintain for some utilities (single-sample and batch).
- `DataFrame.iloc[:, i]` inside a loop is treated as a code smell in review.
- The O(d²) spherical-coordinate pattern was fixed to O(d) via `np.cumprod`.
