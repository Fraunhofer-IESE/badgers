# Architecture and key principles

## Project Structure

```
badgers/
├── badgers/                          # main library package
│   ├── __init__.py
│   ├── core/                         # foundation: base classes, pipeline, utils
│   │   ├── __init__.py
│   │   ├── base.py                   # GeneratorMixin abstract base class
│   │   ├── pipeline.py               # Pipeline for chaining generators
│   │   ├── utils.py                  # probability normalization, RNG helpers
│   │   └── decorators/               # input preprocessing decorators
│   │       ├── __init__.py
│   │       ├── tabular_data.py
│   │       └── time_series.py
│   └── generators/                   # data transformation implementations
│       ├── __init__.py
│       ├── tabular_data/             # outliers, drift, imbalance, missingness, noise
│       │   ├── outliers/
│       │   │   ├── distribution_sampling.py
│       │   │   ├── instance_sampling.py
│       │   │   └── low_density_sampling.py
│       │   ├── drift.py
│       │   ├── imbalance.py
│       │   ├── missingness.py
│       │   └── noise.py
│       ├── time_series/              # changepoints, seasons, trends, errors
│       │   ├── changepoints.py
│       │   ├── missingness.py
│       │   ├── noise.py
│       │   ├── outliers.py
│       │   ├── patterns.py
│       │   ├── seasons.py
│       │   ├── transmission_errors.py
│       │   ├── trends.py
│       │   └── utils.py
│       ├── graph/                    # graph manipulations
│       │   └── missingness.py
│       ├── text/                     # text transformations
│       │   └── typos.py
│       ├── image/                    # image processing (stub)
│       └── geolocated_data/          # geospatial (stub)
│
├── benchmarks/                       # performance benchmarking framework
│   ├── __init__.py
│   ├── __main__.py                   # entry point: python -m benchmarks
│   ├── models.py                     # Scenario, GeneratorBenchmark, BenchmarkResult
│   ├── registry.py                   # auto-discovers _*.py registrations
│   ├── runner.py                     # run_performance() with time/memory measurement
│   ├── cli.py                        # run, baseline, compare subcommands
│   ├── comparator.py                 # regression detection (>20% time, >30% memory)
│   ├── scenarios/                    # reusable data factories
│   │   ├── tabular.py
│   │   ├── time_series.py
│   │   ├── graph.py
│   │   └── text.py
│   └── generators/                   # per-generator benchmark registrations
│       ├── tabular_data/
│       │   ├── _drift.py
│       │   ├── _imbalance.py
│       │   ├── _missingness.py
│       │   ├── _noise.py
│       │   └── _outliers.py
│       ├── time_series/
│       │   ├── _changepoints.py
│       │   ├── _missingness.py
│       │   ├── _noise.py
│       │   ├── _outliers.py
│       │   ├── _patterns.py
│       │   ├── _seasons.py
│       │   ├── _transmission_errors.py
│       │   └── _trends.py
│       ├── graph/
│       │   └── _missingness.py
│       └── text/
│           └── _typos.py
│
├── tests/                            # pytest test suite (148 tests)
│   ├── conftest.py                   # shared fixtures (rng)
│   ├── core/
│   │   ├── test_pipelines.py
│   │   └── test_utils.py
│   ├── benchmarks/
│   │   ├── test_cli.py
│   │   ├── test_comparator.py
│   │   ├── test_models.py
│   │   ├── test_registry.py
│   │   ├── test_runner.py
│   │   └── test_scenarios.py
│   └── generators/
│       ├── tabular_data/
│       │   ├── conftest.py           # tabular_small, tabular_large fixtures
│       │   ├── test_drift.py
│       │   ├── test_imbalance.py
│       │   ├── test_missingness.py
│       │   ├── test_noise.py
│       │   └── test_outliers.py
│       ├── time_series/
│       │   ├── conftest.py           # time_series_sine, time_series_walk fixtures
│       │   ├── test_changepoints.py
│       │   ├── test_missingness.py
│       │   ├── test_noise.py
│       │   ├── test_outliers.py
│       │   ├── test_patterns.py
│       │   ├── test_seasons.py
│       │   ├── test_transmission_errors.py
│       │   ├── test_trends.py
│       │   └── test_utils.py
│       ├── graph/
│       │   ├── conftest.py           # graph_erdos_renyi fixture
│       │   └── test_missingness.py
│       └── text/
│           ├── conftest.py           # text_word_list fixture
│           └── test_typos.py
│
├── docs/                             # mkdocs documentation
│   ├── index.md
│   ├── architecture.md
│   ├── benchmarking.md
│   ├── changelog.md
│   ├── getting-started.md
│   ├── tutorials/                    # Jupyter notebook tutorials
│   └── superpowers/                  # design specs and implementation plans
│
├── experiments/                      # ad-hoc experiments and benchmark results
├── pyproject.toml                    # project metadata and build config
├── requirements.txt
├── tox.ini                           # multi-version test matrix (py38–py314)
└── mkdocs.yml                        # documentation site config
```

## Core Module

The `core` module serves as the foundation of the Badgers framework, providing essential building blocks and infrastructure that other components rely on.

### Main Responsibilities:

1. **Base Classes**: Defines the fundamental `GeneratorMixin` abstract base class that all generators must inherit from, ensuring a consistent interface across the entire system.

2. **Standardized Interface**: Enforces a uniform `generate(X, y, **params)` method signature that returns transformed data `(Xt, yt)` for all generators.

3. **Input Preprocessing**: Provides decorator functions (`preprocess_inputs`) that automatically validate and convert input data to standardized formats (numpy arrays).

4. **Pipeline Infrastructure**: Implements the `Pipeline` class that enables chaining multiple generators together in sequential workflows.

5. **Utility Functions**: Offers helper functions for common operations like probability normalization and random number generation.

## Generators Module

The `generators` module contains the actual implementation of various data transformation algorithms, organized by data type categories.

### Main Responsibilities:

1. **Data Transformation Implementation**: Houses concrete implementations of various data generation techniques across different data domains:
   - Tabular data transformations (outliers, drift, imbalance, missingness, noise)
   - Time series modifications (changepoints, seasons, trends, transmission errors)
   - Graph-based manipulations
   - Image processing generators
   - Text transformation tools

2. **Domain-Specific Organization**: Structures generators by data type categories, making it easy to find and use appropriate transformations for specific data modalities.

3. **Extensibility**: Provides a plug-and-play architecture where new generators can be easily added by following the established `GeneratorMixin` interface.

## Benchmarks Module

The `benchmarks` package provides a systematic framework for measuring performance (time/memory) of all generators. It lives outside the main `badgers` source to avoid coupling.

### Architecture (3 Layers)

1. **Registration Layer** (`models.py`, `registry.py`, `scenarios/`, `generators/`):
   - `Scenario` dataclasses define reusable data factories for each data type (tabular, time series, graph, text).
   - `GeneratorBenchmark` ties a generator class to its scenarios.
   - The `Registry` auto-discovers `_*.py` registration modules from `benchmarks/generators/`.

2. **Execution Layer** (`runner.py`, `cli.py`):
   - `run_performance()` measures wall-clock time and peak memory over multiple iterations.
   - CLI provides `run`, `baseline`, and `compare` subcommands.

3. **Analysis Layer** (`comparator.py`):
   - `compare_results()` diffs two result sets and flags regressions (>20% time, >30% memory).

### Usage

```bash
# Run all benchmarks (performance)
python -m benchmarks run

# Filter by generator category
python -m benchmarks run --generators tabular_data.outliers

# Save a baseline for regression detection
python -m benchmarks baseline save --name v1.0

# Compare latest results against a baseline
python -m benchmarks compare --baseline v1.0
```

### Adding a New Generator to Benchmarks

Create a `_<name>.py` file in the appropriate `benchmarks/generators/<category>/` directory:

```python
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator

register(GeneratorBenchmark(
    generator_cls=GaussianNoiseGenerator,
    name="GaussianNoise",
    module_path="tabular_data.noise",
    default_params={"noise_std": 0.5},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
```

The registry auto-discovers all `_*.py` files — no other changes needed.

## Tests

Tests live under `tests/` and use **pytest** with function-based tests (no `unittest.TestCase` classes). The test structure mirrors the source layout:

```
tests/
├── conftest.py                  # shared fixtures (e.g., rng)
├── core/                        # tests for badgers.core
├── benchmarks/                  # tests for the benchmarks package
└── generators/
    ├── tabular_data/
    │   ├── conftest.py          # tabular-specific fixtures
    │   ├── test_outliers.py
    │   ├── test_noise.py
    │   └── ...
    ├── time_series/
    │   ├── conftest.py          # time-series fixtures (sine, random walk)
    │   └── ...
    ├── graph/
    │   ├── conftest.py          # graph fixtures (Erdős-Rényi)
    │   └── ...
    └── text/
        ├── conftest.py          # text fixtures (word list)
        └── ...
```

### Key Conventions

- **Fixtures over classes**: Each data type has a `conftest.py` with reusable fixtures (e.g., `tabular_small`, `time_series_sine`, `graph_erdos_renyi`, `text_word_list`). Fixtures return `(X, y)` tuples with deterministic random seeds.
- **Flat test functions**: Tests are plain `test_<subject>__<behavior>` functions (double underscore separator), not methods on `TestCase` classes.
- **Generator tests are independent**: Generator correctness is tested directly via pytest, not through the benchmark framework's functional checks. This keeps tests fast, focused, and free of benchmark infrastructure coupling.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only generator tests
pytest tests/generators/

# Run a specific category
pytest tests/generators/tabular_data/ -v
```

## Architectural Decisions

This section records significant architectural decisions made during the project's evolution. Each decision includes the rationale and date of adoption.

### ADR-1: Strict Separation of Tests and Benchmarks (2026-06-23)

**Decision:** Unit tests (`tests/`) and performance benchmarks (`benchmarks/`) serve distinct purposes and must not overlap.

| Concern | Unit Tests | Benchmarks |
|---|---|---|
| **Purpose** | Catch logic bugs | Measure perf, detect regressions |
| **Data** | Small, varied (numpy/pandas × 1D/2D) | Medium-large, standardized |
| **Data source** | Per-type `conftest.py` fixtures | `benchmarks/scenarios/` |
| **Speed** | Fast (<1ms each) | Measured (timed iterations) |
| **Checks** | Exact assertions | Performance stats only |

**Rationale:** The benchmarks module originally duplicated functional assertions via `FunctionalCheck`/`FunctionalResult`. These were removed so that `tests/` is the single source of truth for correctness and benchmarks focus exclusively on timing, memory, and regression detection. Generator correctness is tested directly via pytest, not through the benchmark framework.

### ADR-2: Per-Data-Type Pytest Fixture Architecture (2026-06-23)

**Decision:** Tests use per-data-type `conftest.py` files with parametrized pytest fixtures instead of `unittest.TestCase` classes or a monolithic fixture file.

```
tests/
  conftest.py                          # shared: rng fixture only
  generators/
    tabular_data/
      conftest.py                      # tabular_data, tabular_data_labeled fixtures
    time_series/
      conftest.py                      # time_series_sine, time_series_walk fixtures
    graph/
      conftest.py                      # graph_erdos_renyi fixture
    text/
      conftest.py                      # text_word_list fixture
```

**Naming conventions:**
- Fixtures: prefixed with data type (`tabular_data`, `time_series_sine`, `graph_erdos_renyi`, `text_word_list`)
- Test functions: `test_<subject>__<behavior>` with double underscore separator (e.g., `test_gaussian_noise__increases_variance`)
- No `unittest.TestCase` classes — flat pytest functions only
- Only numpy and pandas input types (no `list_1D`/`list_2D`)

**Rationale:** Fixtures are auto-discovered by pytest's `conftest.py` mechanism, scoped to the nearest directory. This keeps test data close to the tests that use it, avoids a single monolithic fixture file, and enables pytest's built-in parametrization for testing across input types.

### ADR-3: Benchmark Registration via Auto-Discovered Modules (2026-06-24)

**Decision:** Each generator category has a `_*.py` registration module in `benchmarks/generators/<data_type>/` that imports generator classes and calls `register(GeneratorBenchmark(...))`. The registry auto-discovers these modules.

**Key constraint:** Density-based generators (e.g., `HistogramSamplingGenerator`) must only use `SCENARIO_SMALL_BLOBS` (2D) because they raise on >5D input.

**Rationale:** Decouples benchmark definitions from the benchmark runner. Adding a new generator to benchmarks is a single `register()` call in the appropriate `_*.py` file — no other changes needed.

### ADR-4: Vectorized NumPy Over Python Loops (2026-06-26)

**Decision:** All generator implementations must prefer batched/vectorized NumPy operations over Python-level `for` loops, especially when `n_samples` can be large.

**Three canonical patterns:**

1. **Per-row RNG → batched RNG:** Replace `[rng.exponential(size=d) for _ in range(n)]` with `rng.exponential(size=(n, d))`
2. **Per-column `.iloc`/`.choice` → integer array indexing:** Replace `np.stack([rng.choice(X.iloc[:, i], ...) for i in range(d)])` with `X.values[row_idx, np.arange(d)]`
3. **Batch variants for utilities:** When a utility operates on a single sample (e.g., `random_spherical_coordinate`), also provide a batch version (`random_spherical_coordinates`) accepting arrays with `axis=` parameters

**When NOT to vectorize:** small `n_samples` (≤10), algorithms requiring sequential state (e.g., rejection sampling with unknown iteration count).

**Rationale:** Badgers generators often produce many samples at once. Python loops over RNG calls or pandas `.iloc` accesses are orders of magnitude slower than batched NumPy operations. The O(d²) `np.prod(sin_phis[:i])` pattern in `random_spherical_coordinate` was also fixed to O(d) via `np.cumprod`.

### ADR-5: Baseline-Based Regression Detection (2026-06-24)

**Decision:** Benchmarks support a baseline workflow for detecting performance regressions with defined thresholds:

```bash
python -m benchmarks run --iterations 10
python -m benchmarks baseline save --name v0.0.13
python -m benchmarks compare --baseline v0.0.13
```

**Regression thresholds:** >20% time increase or >30% memory increase triggers a flag in `comparator.py`.

**Rationale:** Baselines are versioned snapshots of generator performance. They enable automated detection of performance regressions before merging, providing a quantitative gate for optimization work.