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

3. **Input Preprocessing**: Provides decorator functions (`preprocess_inputs`) that automatically validate and convert input data to standardized formats (pandas DataFrames/Series).

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
from benchmarks.models import Scenario, GeneratorBenchmark
from benchmarks.registry import register

register(GeneratorBenchmark(
    class_name="MyNewGenerator",
    module="badgers.generators.tabular_data.noise",
    scenarios={"small": Scenario(...), "large": Scenario(...)},
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