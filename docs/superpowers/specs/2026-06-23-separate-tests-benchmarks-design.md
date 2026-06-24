# Design: Separate functional tests from performance benchmarks

**Date:** 2026-06-23
**Status:** approved

## Motivation

The `benchmarks/` module was recently introduced with its own functional check system (`FunctionalCheck`, `benchmarks/checks/`, `run_functional()`). This duplicates the functional assertions already present in `tests/generators/`. We want:

1. **One source of truth for functional correctness** — the unit tests in `tests/`
2. **Benchmarks focused on performance** — timing, memory, regression detection only
3. **Cleaner test configuration** — replace manual input-type loops with pytest fixtures

## Design

### 1. Remove functional checks from benchmarks

**Deleted:**
- `benchmarks/checks/` — entire directory
- `tests/benchmarks/test_checks.py`

**Removed from `benchmarks/models.py`:**
- `FunctionalCheck` dataclass
- `FunctionalResult` dataclass
- `functional` field from `BenchmarkResult`
- `functional_checks` field from `GeneratorBenchmark`

**Removed from `benchmarks/runner.py`:**
- `run_functional()` function
- `run_all()` simplified to just call `run_performance()`

**Changed in `benchmarks/cli.py`:**
- Remove `--type functional` CLI option
- Remove `"functional"` key from serialized JSON output
- `cmd_run()` only calls `run_performance()`

**Changed in benchmark registrations (`benchmarks/generators/**/_*.py`):**
- Remove `functional_checks=[...]` from every `GeneratorBenchmark(...)` call
- Remove imports of check constants

**Updated tests:**
- `tests/benchmarks/test_runner.py` — remove `TestRunFunctional`, keep `TestRunPerformance`
- `tests/benchmarks/test_models.py` — remove `TestFunctionalCheck`, `TestFunctionalResult`
- `tests/benchmarks/test_cli.py` — update to not reference functional checks

### 2. Refactor test data into pytest fixtures

Replace `tests/generators/tabular_data/__init__.py` helper functions with per-data-type `conftest.py` files using pytest fixtures.

**New file structure:**
```
tests/
  conftest.py                          # shared: rng fixture only
  generators/
    tabular_data/
      conftest.py                      # tabular_data, tabular_data_labeled fixtures
      test_noise.py
      test_outliers.py
      test_drift.py
      test_imbalance.py
      test_missingness.py
    time_series/
      conftest.py                      # time_series_sine, time_series_walk fixtures
      test_noise.py
      test_outliers.py
      test_changepoints.py
      test_missingness.py
      test_patterns.py
      test_seasons.py
      test_transmission_errors.py
      test_trends.py
      test_utils.py
    graph/
      conftest.py                      # graph_erdos_renyi fixture
      test_missingness.py
    text/
      conftest.py                      # text_word_list fixture
      test_typos.py
```

**Fixture naming convention:** Prefix with data type — `tabular_data`, `tabular_data_labeled`, `time_series_sine`, `graph_erdos_renyi`, `text_word_list`.

**Tabular fixtures** (`tests/generators/tabular_data/conftest.py`):
- `tabular_data` — parametrized over `numpy_1D`, `numpy_2D`, `pandas_1D`, `pandas_2D`. Yields `(X, y=None)`.
- `tabular_data_labeled` — same parametrization. Yields `(X, y)` with 5-class classification labels.
- List types (`list_1D`, `list_2D`) are dropped — generators only use numpy/pandas.

**Time series fixtures** (`tests/generators/time_series/conftest.py`):
- `time_series_sine` — 200-point sine wave
- `time_series_walk` — 200-point random walk

**Graph fixtures** (`tests/generators/graph/conftest.py`):
- `graph_erdos_renyi` — 100-node Erdős-Rényi graph

**Text fixtures** (`tests/generators/text/conftest.py`):
- `text_word_list` — list of 20 technical words

**Shared fixture** (`tests/conftest.py`):
- `rng` — `np.random.default_rng(0)`

### 3. Convert tests from unittest.TestCase to pytest functions

**Naming convention:** `test_<subject>__<behavior>`

Examples:
```python
def test_gaussian_noise__increases_variance(tabular_data): ...
def test_gaussian_noise__preserves_shape(tabular_data): ...
def test_random_shift__preserves_shape_scalar_std(tabular_data): ...
def test_random_shift__preserves_shape_array_std(tabular_data): ...
def test_outliers__correct_shape_and_labels(tabular_data): ...
def test_missingness__correct_nan_count(tabular_data): ...
```

**Pattern:** Double underscore `__` separates subject from behavior. Single underscore within each part.

**No more TestCase classes.** Drop `setUp`/`tearDown` — fixtures handle setup. If a group of tests shares a generator instance, use a fixture:
```python
@pytest.fixture
def generator():
    return GaussianNoiseGenerator()
```

**Edge cases stay as plain functions** — tests for error paths, internal state, or hand-crafted data don't use parametrized fixtures.

### 4. What stays unchanged

- `benchmarks/scenarios/` — used by performance benchmarks only
- `benchmarks/generators/**/_*.py` — only change is removing `functional_checks=[...]`
- `tests/core/` — no changes needed
- `tests/benchmarks/test_comparator.py` — untouched
- `tests/benchmarks/test_registry.py` — untouched
- `tests/benchmarks/test_scenarios.py` — untouched

### 5. Separation of concerns

| | Unit tests | Benchmarks |
|---|---|---|
| **Purpose** | Catch logic bugs | Measure perf, detect regressions |
| **Data** | Small, varied (numpy/pandas × 1D/2D) | Medium-large, standardized |
| **Data source** | Per-type conftest fixtures | `benchmarks/scenarios/` |
| **Speed** | Fast (<1ms each) | Measured (timed iterations) |
| **Checks** | Exact assertions | Performance stats only |
