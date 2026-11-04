# ADR-0006: Outlier value synthesis owned by shared OOD samplers

- **Date:** 2026-10-25
- **Status:** Accepted

## Context

Outlier value synthesis was implemented inside the outlier generators. The causal outlier
generator needed the same algorithms, and users wanted to reuse them in custom pipelines.
Duplicating the algorithms per generator meant a new strategy required a new generator class in
two places.

## Decision

All non-causal outlier value synthesis is delegated to the reusable OOD samplers in
`badgers.core.sampling`. One generic `OutlierSamplingGenerator` selects a sampler at
`generate()` time; a string registry name is resolved via `create_out_of_distribution_sampler`.

| Strategy | Sampler (`badgers.core.sampling`) | Registry name |
|---|---|---|
| z-score (±(3 + exponential)·σ per column) | `ZScoreSampler(scale)` | `"zscore"` (default) |
| hypersphere shell (radius ≥ 3σ) | `HypersphereSampler(scale)` | `"hypersphere"` |
| uniform around the observed range | `UniformOutOfDistributionSampler(expansion)` | `"uniform"` |
| per-column low-density histogram bins | `IndependentHistogramSampler(bins)` | `"independent_histogram"` |
| joint low-density histogram bins (≤ 5 columns) | `JointHistogramSampler(bins, threshold_low_density)` | `"histogram"` |
| grid-CDF KDE: scores a bounded regular grid, categorical over low-density cells, exact count, no rejection | `KernelDensitySampler(threshold_low_density, grid_budget)` | `"low_density"` |
| adaptive kd partition (median splits), axis-aligned Voronoi cells extended across the full gap to the data-box boundary on every axis, uncovered box regions added as always-qualifying count-0 cells, categorical over cells with extended-territory density ≤ threshold × mean density, exact count | `TreeCellSampler(threshold_low_density, leaf_size)` | `"tree"` |

The generator keeps its responsibilities outside synthesis: the required `selector` owns the
outlier count and the perturbed targets; the generator copies the seed rows, replaces the
selected cells with the sampler's values, appends the rows, and records
`get_outliers_labels()`.

`CausalOutlierPropagationGenerator` consumes the same samplers for its perturbation nodes, so
both outlier families share one strategy set and one interface
(`sample(rng, X, columns, n)`).

## Consequences

- The sampling algorithms are usable beyond the outlier generators.
- A new strategy is a new sampler, not a new generator class — for both the non-causal and the
  causal generator.
- `badgers.core.sampling` becomes a public surface that must stay stable.
