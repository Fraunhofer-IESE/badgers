# Changelog

See https://github.com/Fraunhofer-IESE/badgers/releases

## dev_0_0_15 (unreleased)

### Breaking changes

- **`PyMCDGP` and the old `badgers.core.dgp` module are removed.** The causal
  defect generators now fit a **per-node data generation process** behind the
  new `DGP` protocol (`badgers.core.dgp` is a package). The default backend is
  a deterministic pure-NumPy least-squares DGP (`OlsDGP`) — no extra
  dependency needed. Custom backends are injected as objects (`dgp=...` /
  `use_dgp`); PyMC-specific knobs (`n_samples_mcmc`, `fit_dgp`) no longer
  appear in `fit(...)`. PyMC- and DoWhy-GCM-based backends are planned as
  follow-ups (the `causal` extra still installs PyMC + PyTensor for writing
  them).
- **`fit` signature change** for the causal defect family:
  `fit(X, graph, column_mapping, dgp=None)` — `n_samples_mcmc` and
  `fit_dgp` are gone.
- **`CausalEdgePerturbationGenerator.generate` drops the dead
  `mechanism`/`mechanism_cause`/`mechanism_rate` parameters** — they were
  accepted but ignored (the edge generator's gate is always "all rows").
  The signature is now
  `generate(X, y, perturbation_nodes=None, edge=None, weight_factor=2.0)`.
- **`CausalImbalanceGenerator` no longer requires a DGP** (breaking): the
  generator is a pure row-subset (quantile-range truncation); the kept
  rows' parents are unchanged, so no structural equation is re-evaluated.
  The dead `mechanism`/`mechanism_cause`/`mechanism_rate` parameters, the
  on-demand OLS fit, and the no-op propagate pass are removed — the
  signature is now
  `generate(X, y, perturbation_nodes=None, quantile_range=(0.5, 1.0))`.
  This is the first generator in the causal family whose `generate` needs
  neither a DGP nor any random numbers (selection-side only).

### Added

- **Per-node DGP protocol** (`badgers.core.dgp` package): `DGP` protocol and
  `DGPMixin` (topological `fit`/`sample` walk) — backends implement exactly
  one operation, `fit_node(node, parents, data)`, and expose
  `response(node, parent_values)` and `draw_noise(node, rng, n)`;
  `sample(rng, n)` is tier 2 and no generator depends on it.
- **`do()` intervention primitive** (`badgers.core.dgp.algebra`): overwrite
  node columns and cascade prediction differences to descendants
  (`mode="delta"` — every node's own exogenous noise is preserved exactly) or
  redraw their noise (`mode="recompute"`); backend-blind.
- **`ScaledEdgeDGP`** view for edge reweighting without coefficient access
  (calibrate on the fitted base DGP, then `response` + `draw_noise`).
- **Causal defect family** (five generators sharing `CausalGenerator`, an
  explicit `fit(X, graph, column_mapping)` phase followed by `generate(...)`
  with mechanism gating `mechanism`/`mechanism_cause`/`mechanism_rate`):
  `CausalDriftGenerator` (single systematic shift, propagated as a delta),
  `CausalNoiseGenerator` (per-edge noise injection, `noise_std`),
  `CausalImbalanceGenerator` (quantile-range truncation — pure subset,
  descendants untouched), `CausalMissingnessGenerator` (m-graph missingness
  with `any`/`all` propagation, optional observation-graph derivation),
  `CausalEdgePerturbationGenerator` (marginal-preserving edge-weight
  perturbation; child recomputed with the new coefficient, descendants
  delta-propagated).
- **`causal` optional extra**: PyMC + PyTensor (`pip install
  "badgers[causal]"`) — only needed for writing custom PyMC-based backends;
  the default OLS path is pure NumPy.

- **`TopQuantileSelector`** preset (`badgers.core.selectors`): fires rows in
  the top ``rate`` fraction of a column — self mode (own column) or driver
  mode (a driver column, optionally broadcast to all gated variables).
  Deterministic (no rng); thresholds computed on the called rows or on a
  separate ``stats`` row set (for generators gating expanded rows against
  the original columns' statistics). All four mechanism knobs of the causal
  defect family are now backed by selector presets (`BernoulliSelector`,
  `TopQuantileSelector`, or the inline all-True rule); the legacy per-row
  `_should_defect` API is removed.

## dev_0_0_14 (unreleased)

### Added

- **`CausalOutlierPropagationGenerator`**: New tabular generator that propagates outliers along causal-graph edges. First generator to rely on the causal-graph API.
- **Causal-graph utilities**: `validate_dag`, `topological_order`, and related helpers for working with causal graphs.
- **Complete tabular generator benchmark coverage**: All 16 tabular generators are now registered in the benchmark framework (up from 7). Added registrations for `HistogramSamplingGenerator`, `ZScoreSamplingGenerator`, `LowDensitySamplingGenerator`, `HypersphereSamplingGenerator`, `GaussianNoiseClassesGenerator`, `RandomShiftClassesGenerator`, `RandomSamplingClassesGenerator`, and `RandomSamplingTargetsGenerator`.

### Changed

- **Generators now output NumPy arrays** (breaking): Tabular and time-series generators return NumPy arrays instead of pandas DataFrames/Series. Input handling still accepts pandas DataFrames/Series.
- **Vectorized generators**: Replaced per-row/per-column `.iloc` loops with batched NumPy operations for improved performance.
- **Two-tier sampler API**: The `sampler` parameter is split into `within_distribution_sampler` and `out_of_distribution_sampler`.
- **Explicit `column_mapping` parameter**: The causal-graph API now requires an explicit `column_mapping` parameter.
- **Benchmark outputs moved to `.benchmarks/`**: Results and baselines are now stored in `.benchmarks/results/` and `.benchmarks/baselines/` (gitignored) instead of `benchmarks/results/` and `benchmarks/baselines/`.

### Removed

- `node_to_index` parameter from the causal-graph API (replaced by `column_mapping`).

## dev_0_0_13 (released)

### Changed

- **Separated tests from benchmarks**: Generator correctness tests now live in `tests/generators/` as standalone pytest functions with fixtures, independent of the benchmark framework. The `benchmarks/checks/` directory and `FunctionalCheck` infrastructure have been removed. Benchmarks now focus exclusively on performance measurement (time/memory). (9 commits: `943e230`..`56fee5d`)

### Added

- Pytest fixtures for each data type: `tabular_small`/`tabular_large`, `time_series_sine`/`time_series_walk`, `graph_erdos_renyi`, `text_word_list`
- Test conftest files in `tests/generators/{tabular_data,time_series,graph,text}/`

### Removed

- `benchmarks/checks/` directory and all `FunctionalCheck` infrastructure
- `--type functional` option from benchmark CLI
- `functional_checks` parameter from `GeneratorBenchmark` model