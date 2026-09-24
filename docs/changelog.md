# Changelog

See https://github.com/Fraunhofer-IESE/badgers/releases

## dev_0_0_15 (unreleased)

### Added

- **Causal defect family**: A new family of tabular generators that produce bad data by perturbing the *structure* of the data (causal graph + data generation process) and propagating the induced **delta** through descendants: each descendant moves by `beta @ (parents_new − parents_orig)` on top of its original value, so every node's own exogenous noise is preserved exactly. All generators share an explicit `fit(X, graph, column_mapping)` phase followed by `generate(...)` with a mechanism-gating API (`mechanism`, `mechanism_cause`, `mechanism_rate`) that selects which rows receive the defect.
  - **`PyMCDGP`** (`badgers.core.dgp`): PyMC-based data generation process. Fits a Bayesian linear structural model (latent nodes + observation layer) and exposes `fit`, `forward_pass`, and `sample` for structural perturbation.
  - **`CausalGenerator`** (`badgers.generators.tabular_data.causal`): shared base class with DAG/column-mapping/perturbation validation, DGP fitting, mechanism gating, delta propagation, and label building.
  - **`CausalMissingnessGenerator`** (`badgers.generators.tabular_data.missingness`): m-graph mechanism-based missingness with `any`/`all` descendant propagation (absorbing NaN mask pass).
  - **`CausalNoiseGenerator`** (`badgers.generators.tabular_data.noise`): per-edge noise injection with configurable `noise_std`.
  - **`CausalDriftGenerator`** (`badgers.generators.tabular_data.drift`): single systematic shift (drawn once) on perturbation nodes, propagated as a delta to their descendants.
  - **`CausalImbalanceGenerator`** (`badgers.generators.tabular_data.imbalance`): truncates the dataset to a quantile range of the perturbation node — a pure subset (unchanged parents ⇒ delta 0 ⇒ descendants untouched).
  - **`CausalEdgePerturbationGenerator`** (`badgers.generators.tabular_data.relationship`): marginal-preserving edge-weight perturbation (relationship-level defect). The child is recomputed with the new coefficient (its equation changed); its descendants are delta-propagated.
- **`causal` optional extra**: PyMC + PyTensor are now an optional dependency (`pip install "badgers[causal]"`). The classic defect packages do not import PyMC; only the `causal` subpackage requires the extra.

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