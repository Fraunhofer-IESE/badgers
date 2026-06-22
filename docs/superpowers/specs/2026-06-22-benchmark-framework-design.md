# Benchmark Framework Design

**Date:** 2026-06-22
**Status:** Draft
**Author:** Julien Siebert

## 1. Motivation

Badgers needs a systematic performance regression and functional correctness benchmarking framework. When refactoring or optimizing generators, developers need to know whether a change improves or degrades performance, and whether it breaks expected behavior. The framework must be extensible — adding a new generator to the benchmark suite should be as simple as adding one registration, mirroring badgers' core philosophy.

The framework separates two concerns:
- **Functional correctness:** does the generator produce the right kind of bad data? (shape, statistical properties, behavioral invariants)
- **Non-functional performance:** how fast and memory-efficient is it? (wall-clock time, peak memory)

## 2. Architecture Overview

The framework lives in a new top-level `benchmarks/` package, separate from both `tests/` and `experiments/`. It has three layers:

- **Registration Layer:** `Scenario` (data factories), `GeneratorBenchmark` (generator + scenarios + checks), and a `Registry` that auto-discovers registrations.
- **Execution Layer:** A CLI runner that can run functional checks, performance measurements, or both, with filtering by generator category.
- **Analysis Layer:** A comparison command that diffs results against a stored baseline and produces a human-readable regression report.

```
benchmarks/
├── __init__.py
├── __main__.py
├── cli.py
├── registry.py
├── models.py
├── runner.py
├── comparator.py
├── scenarios/
│   ├── __init__.py
│   ├── tabular.py
│   ├── time_series.py
│   ├── graph.py
│   └── text.py
├── checks/
│   ├── __init__.py
│   ├── common.py
│   ├── tabular.py
│   └── time_series.py
├── generators/
│   ├── __init__.py
│   ├── tabular_data/
│   │   ├── __init__.py
│   │   ├── _noise.py
│   │   ├── _outliers.py
│   │   ├── _missingness.py
│   │   ├── _drift.py
│   │   └── _imbalance.py
│   ├── time_series/
│   │   └── ...
│   ├── graph/
│   │   └── ...
│   └── text/
│       └── ...
└── results/
    └── .gitkeep
```

## 3. CLI Interface

```bash
# Run only functional checks
python -m benchmarks run --type functional

# Run only performance measurements
python -m benchmarks run --type performance

# Run both (default)
python -m benchmarks run --type all

# Filter by generator category
python -m benchmarks run --generators tabular_data.outliers
python -m benchmarks run --generators time_series

# Combine filters
python -m benchmarks run --type performance --generators tabular_data

# Save current results as the new baseline
python -m benchmarks baseline save

# Compare current results against a named baseline
python -m benchmarks compare --baseline v0.0.13

# Compare two specific runs
python -m benchmarks compare --baseline results/run_abc123.json --target results/run_def456.json
```

The `--type` flag controls which test protocol runs. The `--generators` flag filters by module path (dot-separated, matches prefix). Both are optional — defaults run everything.

## 4. Data Model

### 4.1 Scenario

A named data factory that produces `(X, y)` for a given generator type. Replaces the ad-hoc `generate_test_data_*` helpers currently in `tests/generators/tabular_data/__init__.py`.

```python
@dataclass
class Scenario:
    name: str                              # e.g. "small_blobs_2d"
    data_type: str                         # "tabular", "time_series", "graph", "text"
    factory: Callable[[Generator], Tuple]  # (X, y) factory, receives a numpy Generator
    tags: List[str] = field(default_factory=list)  # e.g. ["classification", "1D", "large"]
```

Predefined scenarios ship with the framework (small/medium/large for each data type). Generators can reference these or define custom ones in their registration module.

### 4.2 FunctionalCheck

A single assertion about the generator's output.

```python
@dataclass
class FunctionalCheck:
    name: str         # e.g. "output_has_correct_shape"
    description: str  # human-readable
    check: Callable[[Any, Any, Any, Any, Dict], bool]
    # check(original_X, original_y, transformed_X, transformed_y, params) -> bool
```

Predefined checks live in `benchmarks/checks/`:
- `common.py`: `CHECK_SAME_SHAPE`, `CHECK_NO_NANS`, `CHECK_INDEX_PRESERVED`
- `tabular.py`: `CHECK_INCREASED_VARIANCE`, `CHECK_OUTLIER_EXTREMITY`, `CHECK_MISSING_COUNT`
- `time_series.py`: `CHECK_PATTERN_COUNT`, `CHECK_CHANGEPOINT_COUNT`

### 4.3 GeneratorBenchmark

The central registration unit. One per generator class.

```python
@dataclass
class GeneratorBenchmark:
    generator_cls: Type                    # the generator class
    name: str                              # short name, e.g. "GaussianNoise"
    module_path: str                       # e.g. "tabular_data.noise"
    default_params: Dict                   # kwargs for generate()
    scenarios: List[Scenario]              # which scenarios to test
    functional_checks: List[FunctionalCheck]  # what to assert
```

### 4.4 Registry

A module-level list populated at import time. The runner discovers registrations by walking `benchmarks/generators/` and importing each `_*.py` module.

```python
# benchmarks/registry.py
_registry: List[GeneratorBenchmark] = []

def register(benchmark: GeneratorBenchmark):
    _registry.append(benchmark)

def discover() -> List[GeneratorBenchmark]:
    # walks benchmarks/generators/, imports _*.py modules
    ...
```

### 4.5 Registration Example

```python
# benchmarks/generators/tabular_data/_noise.py
from benchmarks import Scenario, FunctionalCheck, GeneratorBenchmark, register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from benchmarks.checks.common import CHECK_SAME_SHAPE
from benchmarks.checks.tabular import CHECK_INCREASED_VARIANCE
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator

register(GeneratorBenchmark(
    generator_cls=GaussianNoiseGenerator,
    name="GaussianNoise",
    module_path="tabular_data.noise",
    default_params={"noise_std": 0.5},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
    functional_checks=[CHECK_SAME_SHAPE, CHECK_INCREASED_VARIANCE],
))
```

## 5. Performance Measurement

### 5.1 Metrics

Two metrics per `(generator, scenario, params)` combination:
- **Wall-clock time** (milliseconds)
- **Peak memory** (megabytes)

### 5.2 Protocol

For each combination:
1. Create fresh generator instance with fixed seed
2. Generate `(X, y)` from the scenario factory
3. Warmup: one `generate()` call (not measured)
4. Run N iterations (configurable, default 5):
   - Time: `time.perf_counter()` around `generate()`
   - Memory: `tracemalloc` peak delta during `generate()`
5. Collect min, max, mean, median, stddev across iterations

No external dependencies — uses stdlib `time` and `tracemalloc`, plus `numpy` (already a project dependency) for statistics.

### 5.3 Result Schema

Results stored as JSON, one file per run, named with timestamp and git branch:

```json
{
  "meta": {
    "timestamp": "2026-06-22T14:30:00",
    "git_commit": "abc1234",
    "git_branch": "optimize-histogram",
    "python_version": "3.12.4",
    "platform": "Windows-10"
  },
  "results": [
    {
      "generator": "tabular_data.outliers.HistogramSampling",
      "scenario": "medium_blobs_5d",
      "params": {"n_outliers": 50, "bins": 10},
      "functional": {
        "passed": 4,
        "failed": 0,
        "checks": [
          {"name": "correct_shape", "passed": true},
          {"name": "outliers_are_extreme", "passed": true}
        ]
      },
      "performance": {
        "time_ms": {
          "min": 142.3, "max": 158.7, "mean": 148.2,
          "median": 146.9, "stddev": 6.1, "iterations": 5
        },
        "memory_mb": {
          "min": 2.1, "max": 3.4, "mean": 2.6,
          "median": 2.5, "stddev": 0.5, "iterations": 5
        }
      }
    }
  ]
}
```

## 6. Comparison and Regression Detection

### 6.1 Baseline Management

Results are saved to `benchmarks/results/` (gitignored). The `baseline` subcommand manages named baselines:

```bash
python -m benchmarks baseline save          # saves as "latest"
python -m benchmarks baseline save --name v0.0.13
python -m benchmarks baseline list
```

### 6.2 Comparison

The `compare` command diffs two result files:

```bash
python -m benchmarks compare                           # latest vs saved baseline
python -m benchmarks compare --baseline v0.0.13        # latest vs named baseline
python -m benchmarks compare --baseline a.json --target b.json  # two specific files
```

### 6.3 Regression Thresholds

Configurable via CLI or config file. Defaults:
- **Time regression:** >20% increase
- **Memory regression:** >30% increase
- **Improvement:** >threshold decrease

A result is flagged as regression (🔴), improvement (🟢), or unchanged (⚪).

### 6.4 Report Format

```
Generator                          Scenario          Time (baseline → current)       Memory
──────────────────────────────────────────────────────────────────────────────────────────────
tabular_data.outliers.Histogram    medium_blobs_5d   148ms → 312ms 🔴 +110%          2.6MB → 2.5MB 🟢 -4%
tabular_data.outliers.LowDensity   medium_blobs_5d   46ms → 44ms  🟢 -4%            0.9MB → 0.9MB ⚪
tabular_data.noise.GaussianNoise   small_blobs_2d    1.2ms → 1.1ms 🟢 -8%           0.3MB → 0.3MB ⚪

Summary: 1 regression, 2 improvements, 3 unchanged
```

## 7. Integration with Existing Code

- **`tests/`** — unchanged. Existing unit tests stay as they are. The benchmark framework is complementary, not a replacement.
- **`experiments/`** — the old `test_generators_bench.py`, `run_benchmarks.bat`, and CSV stats can be retired once the new framework covers the same generators. The visualization and summarization scripts can be adapted to read the new JSON format.
- **`badgers/`** — zero changes to the source package. The framework is entirely external.
- **`pyproject.toml`** — no new dependencies. Everything uses stdlib (`tracemalloc`, `time`, `argparse`, `importlib`, `json`, `dataclasses`) or existing deps (`numpy` for statistics).

## 8. Error Handling

- **Generator raises `NotImplementedError`:** skip that scenario for that generator, log a warning.
- **Functional check raises an exception:** mark the check as failed with the exception message.
- **Performance measurement times out:** configurable timeout per scenario (default 60s). Mark as timed out.
- **Missing baseline for comparison:** error with instructions on how to create one.
- **Empty registry (no generators discovered):** error suggesting to check registration files exist.

## 9. Testing the Framework Itself

The framework should have its own tests in `tests/benchmarks/`:
- `test_models.py` — Scenario, FunctionalCheck, GeneratorBenchmark creation and validation
- `test_registry.py` — registration, discovery, duplicate handling
- `test_runner.py` — functional check execution, performance measurement with mock generators
- `test_comparator.py` — regression detection with known result pairs
- `test_cli.py` — CLI argument parsing and integration

## 10. Future Extensions (Out of Scope)

These are explicitly deferred:
- Golden-file / snapshot testing (exact output comparison with fixed seeds)
- Historical trend tracking and visualization over multiple commits
- CI integration (GitHub Actions workflow)
- Parallel execution of independent benchmark scenarios
- Custom threshold configuration per generator
