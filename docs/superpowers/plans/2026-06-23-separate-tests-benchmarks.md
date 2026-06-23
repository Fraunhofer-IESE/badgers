# Separate Tests from Benchmarks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove duplicated functional checks from benchmarks, refactor tests to use pytest fixtures, and keep benchmarks performance-only.

**Architecture:** Benchmarks become performance-only (timing, memory, regression detection). Unit tests move from `unittest.TestCase` classes to pytest functions parametrized by per-data-type fixtures in `conftest.py` files. Benchmark scenarios and test fixtures remain separate — benchmarks use standardized medium/large data, tests use small varied data.

**Tech Stack:** Python 3.12+, pytest, numpy, pandas, scikit-learn, networkx

## Global Constraints

- No `list_1D`/`list_2D` input types — only numpy and pandas
- Test naming: `test_<subject>__<behavior>` with double underscore separator
- Fixture naming: prefixed with data type (`tabular_data`, `time_series_sine`, etc.)
- No `unittest.TestCase` classes — flat pytest functions only
- Per-data-type `conftest.py` files (not one monolithic file)

---

### Task 1: Remove functional checks from benchmark models and runner

**Files:**
- Modify: `benchmarks/models.py`
- Modify: `benchmarks/runner.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `GeneratorBenchmark` without `functional_checks` field; `BenchmarkResult` without `functional` field; `run_performance()` unchanged; `run_all()` simplified

- [ ] **Step 1: Remove `FunctionalCheck` and `FunctionalResult` from models, and `functional`/`functional_checks` fields**

In `benchmarks/models.py`, remove the `FunctionalCheck` dataclass, `FunctionalResult` dataclass, the `functional` field from `BenchmarkResult`, and the `functional_checks` field from `GeneratorBenchmark`. Also remove the `Optional` import if no longer needed elsewhere.

```python
# benchmarks/models.py — after changes

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np


@dataclass
class Scenario:
    """A named data factory that produces (X, y) for benchmarking."""
    name: str
    data_type: str  # "tabular", "time_series", "graph", "text"
    factory: Callable[[np.random.Generator], Tuple[Any, Any]]
    tags: List[str] = field(default_factory=list)


@dataclass
class GeneratorBenchmark:
    """Registration unit for one generator class."""
    generator_cls: Type
    name: str
    module_path: str  # e.g. "tabular_data.noise"
    default_params: Dict
    scenarios: List[Scenario]


@dataclass
class PerformanceStats:
    """Aggregate statistics from multiple measurement iterations."""
    min: float
    max: float
    mean: float
    median: float
    stddev: float
    iterations: int


@dataclass
class BenchmarkResult:
    """Complete result for one (generator, scenario) combination."""
    generator: str
    scenario: str
    params: Dict
    performance: Dict[str, PerformanceStats] = field(default_factory=dict)
    # performance keys: "time_ms", "memory_mb"


@dataclass
class RunMeta:
    """Metadata about a benchmark run."""
    timestamp: str
    git_commit: str
    git_branch: str
    python_version: str
    platform: str
```

- [ ] **Step 2: Remove `run_functional()` and simplify `run_all()` in runner**

In `benchmarks/runner.py`, delete the entire `run_functional()` function. Simplify `run_all()` to just delegate to `run_performance()`. Remove unused imports (`FunctionalResult`).

```python
# benchmarks/runner.py — after changes

"""Benchmark runner: executes performance measurements."""
import time
import tracemalloc
import warnings
from typing import List, Optional

import numpy as np

from benchmarks.models import (
    GeneratorBenchmark, BenchmarkResult, PerformanceStats,
)


def _compute_stats(values: List[float]) -> PerformanceStats:
    """Compute aggregate statistics from a list of measurement values."""
    arr = np.array(values)
    return PerformanceStats(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        stddev=float(np.std(arr)),
        iterations=len(values),
    )


def run_performance(
    benchmarks: List[GeneratorBenchmark],
    filter_path: Optional[str] = None,
    iterations: int = 5,
    timeout: float = 60.0,
) -> List[BenchmarkResult]:
    """Run performance measurements for all registered benchmarks.

    Args:
        benchmarks: List of GeneratorBenchmark registrations.
        filter_path: Optional module path prefix to filter generators.
        iterations: Number of measurement iterations per scenario.
        timeout: Maximum seconds per scenario (not enforced in MVP).

    Returns:
        List of BenchmarkResult with performance results populated.
    """
    results: List[BenchmarkResult] = []

    for gb in benchmarks:
        if filter_path and not gb.module_path.startswith(filter_path):
            continue

        for scenario in gb.scenarios:
            rng = np.random.default_rng(0)
            try:
                X, y = scenario.factory(rng)
            except Exception as e:
                warnings.warn(
                    f"Scenario {scenario.name} factory failed for {gb.name}: {e}"
                )
                continue

            generator = gb.generator_cls(random_generator=np.random.default_rng(0))

            # Warmup
            try:
                generator.generate(X, y, **gb.default_params)
            except NotImplementedError:
                warnings.warn(
                    f"Generator {gb.name} does not support scenario {scenario.name}"
                )
                continue
            except Exception:
                continue

            time_values = []
            memory_values = []

            for _ in range(iterations):
                gen = gb.generator_cls(random_generator=np.random.default_rng(0))
                tracemalloc.start()
                t_start = time.perf_counter()
                gen.generate(X, y, **gb.default_params)
                t_end = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                time_values.append((t_end - t_start) * 1000)
                memory_values.append(peak / (1024 * 1024))

            results.append(BenchmarkResult(
                generator=f"{gb.module_path}.{gb.name}",
                scenario=scenario.name,
                params=gb.default_params,
                performance={
                    "time_ms": _compute_stats(time_values),
                    "memory_mb": _compute_stats(memory_values),
                },
            ))

    return results


def run_all(
    benchmarks: List[GeneratorBenchmark],
    filter_path: Optional[str] = None,
    iterations: int = 5,
    timeout: float = 60.0,
) -> List[BenchmarkResult]:
    """Run performance benchmarks (convenience alias for run_performance)."""
    return run_performance(benchmarks, filter_path, iterations, timeout)
```

- [ ] **Step 3: Run existing benchmark tests to verify nothing is broken yet**

Run: `pytest tests/benchmarks/ -v`
Expected: some failures (tests still reference removed types — that's fine, we fix them in later tasks)

- [ ] **Step 4: Commit**

```bash
git add benchmarks/models.py benchmarks/runner.py
git commit -m "refactor: remove functional checks from benchmark models and runner"
```

---

### Task 2: Remove functional checks from benchmark CLI

**Files:**
- Modify: `benchmarks/cli.py`

**Interfaces:**
- Consumes: updated `BenchmarkResult` (no `functional` field), `run_performance()` / `run_all()`
- Produces: CLI with `run` (performance-only), `baseline`, `compare` subcommands

- [ ] **Step 1: Update CLI to remove functional type and serialization**

In `benchmarks/cli.py`, remove the `--type` argument from the `run` subcommand, remove `run_functional` import, update `cmd_run()` to only call `run_performance()` or `run_all()`, and remove the `"functional"` key from `_serialize_results()`.

```python
# benchmarks/cli.py — key changes

# In build_parser(), change the 'run' subparser:
# Remove: parser_run.add_argument("--type", ...)
# Keep only: --generators, --iterations, --timeout

def build_parser():
    parser = argparse.ArgumentParser(description="Badgers benchmark framework")
    subparsers = parser.add_subparsers(dest="command")

    # run
    parser_run = subparsers.add_parser("run", help="Run benchmarks")
    parser_run.add_argument(
        "--generators", type=str, default=None,
        help="Filter generators by module path prefix (e.g. 'tabular_data')",
    )
    parser_run.add_argument(
        "--iterations", type=int, default=5,
        help="Number of performance measurement iterations",
    )
    parser_run.add_argument(
        "--timeout", type=float, default=60.0,
        help="Timeout per scenario in seconds",
    )

    # baseline
    parser_baseline = subparsers.add_parser("baseline", help="Manage baselines")
    baseline_sub = parser_baseline.add_subparsers(dest="baseline_command")
    parser_baseline_save = baseline_sub.add_parser("save", help="Save latest results as baseline")
    parser_baseline_save.add_argument("--name", type=str, default="latest")
    baseline_sub.add_parser("list", help="List saved baselines")

    # compare
    parser_compare = subparsers.add_parser("compare", help="Compare results against baseline")
    parser_compare.add_argument("--baseline", type=str, default=None)
    parser_compare.add_argument("--target", type=str, default=None)

    return parser


# In cmd_run(), remove the if/elif for functional/performance/all:
def cmd_run(args):
    """Execute the 'run' subcommand."""
    benchmarks = discover()
    if not benchmarks:
        print("No benchmarks discovered. Check that benchmark registration files exist.")
        sys.exit(1)

    results = run_all(benchmarks, args.generators, args.iterations, args.timeout)

    meta = _make_meta()
    data = _serialize_results(results, meta)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{meta.git_branch}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {filepath}")
    print(f"  Generators: {len(set(r.generator for r in results))}")
    print(f"  Scenarios:  {len(results)}")


# In _serialize_results(), remove the "functional" key:
def _serialize_results(results: List[BenchmarkResult], meta: RunMeta) -> dict:
    """Convert results to JSON-serializable dict."""
    return {
        "meta": {
            "timestamp": meta.timestamp,
            "git_commit": meta.git_commit,
            "git_branch": meta.git_branch,
            "python_version": meta.python_version,
            "platform": meta.platform,
        },
        "results": [
            {
                "generator": r.generator,
                "scenario": r.scenario,
                "params": _make_json_safe(r.params),
                "performance": {
                    key: {
                        "min": ps.min, "max": ps.max, "mean": ps.mean,
                        "median": ps.median, "stddev": ps.stddev,
                        "iterations": ps.iterations,
                    }
                    for key, ps in r.performance.items()
                } if r.performance else None,
            }
            for r in results
        ],
    }
```

Also remove the `run_functional` import at the top of the file:
```python
# Change:
from benchmarks.runner import run_functional, run_performance, run_all
# To:
from benchmarks.runner import run_performance, run_all
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/cli.py
git commit -m "refactor: remove functional type from benchmark CLI"
```

---

### Task 3: Remove functional_checks from benchmark registrations

**Files:**
- Modify: `benchmarks/generators/tabular_data/_noise.py`
- Modify: `benchmarks/generators/tabular_data/_outliers.py`
- Modify: `benchmarks/generators/tabular_data/_missingness.py`
- Modify: `benchmarks/generators/tabular_data/_drift.py`
- Modify: `benchmarks/generators/tabular_data/_imbalance.py`
- Modify: `benchmarks/generators/time_series/_noise.py`
- Modify: `benchmarks/generators/time_series/_outliers.py`
- Modify: `benchmarks/generators/time_series/_patterns.py`
- Modify: `benchmarks/generators/time_series/_missingness.py`
- Modify: `benchmarks/generators/time_series/_changepoints.py`
- Modify: `benchmarks/generators/time_series/_seasons.py`
- Modify: `benchmarks/generators/time_series/_trends.py`
- Modify: `benchmarks/generators/time_series/_transmission_errors.py`
- Modify: `benchmarks/generators/graph/_missingness.py`
- Modify: `benchmarks/generators/text/_typos.py`

**Interfaces:**
- Consumes: `GeneratorBenchmark` without `functional_checks` field
- Produces: clean registrations with only `generator_cls`, `name`, `module_path`, `default_params`, `scenarios`

- [ ] **Step 1: Remove functional_checks and check imports from all registration files**

For each file, remove the `functional_checks=[...]` argument from every `GeneratorBenchmark(...)` call and remove any imports of check constants. Example for `_noise.py`:

```python
# benchmarks/generators/tabular_data/_noise.py — after
"""Benchmark registrations for tabular noise generators."""
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

Apply the same pattern to all 15 files — remove `functional_checks=[...]` and any `from benchmarks.checks...` imports.

- [ ] **Step 2: Verify registrations still import cleanly**

Run: `python -c "from benchmarks.registry import discover; print(len(discover()))"`
Expected: prints the number of registered benchmarks (should match previous count)

- [ ] **Step 3: Commit**

```bash
git add benchmarks/generators/
git commit -m "refactor: remove functional_checks from all benchmark registrations"
```

---

### Task 4: Delete benchmarks/checks/ directory and update benchmark tests

**Files:**
- Delete: `benchmarks/checks/__init__.py`
- Delete: `benchmarks/checks/common.py`
- Delete: `benchmarks/checks/tabular.py`
- Delete: `benchmarks/checks/time_series.py`
- Delete: `tests/benchmarks/test_checks.py`
- Modify: `tests/benchmarks/test_runner.py`
- Modify: `tests/benchmarks/test_models.py`
- Modify: `tests/benchmarks/test_cli.py`

**Interfaces:**
- Consumes: updated models and runner from Tasks 1-3
- Produces: passing benchmark test suite

- [ ] **Step 1: Delete the checks directory and test_checks.py**

```bash
rm -r benchmarks/checks/
rm tests/benchmarks/test_checks.py
```

- [ ] **Step 2: Update test_runner.py — remove TestRunFunctional, keep TestRunPerformance**

```python
# tests/benchmarks/test_runner.py — after
import unittest
import numpy as np
from benchmarks.models import Scenario, GeneratorBenchmark
from benchmarks.runner import run_performance, run_all


class FakeGenerator:
    def __init__(self, random_generator=None):
        self.random_generator = random_generator

    def generate(self, X, y, **params):
        return X, y


class TestRunPerformance(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=lambda rng: (np.zeros((100, 2)), None),
        )

    def test_measures_time_and_memory(self):
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
        )
        results = run_performance([gb], iterations=3)
        self.assertEqual(len(results), 1)
        perf = results[0].performance
        self.assertIn("time_ms", perf)
        self.assertIn("memory_mb", perf)
        self.assertEqual(perf["time_ms"].iterations, 3)
        self.assertGreaterEqual(perf["time_ms"].min, 0)

    def test_run_all_delegates_to_performance(self):
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
        )
        results = run_all([gb], iterations=2)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].performance)
        self.assertIn("time_ms", results[0].performance)
```

- [ ] **Step 3: Update test_models.py — remove FunctionalCheck and FunctionalResult tests**

Remove `TestFunctionalCheck` and `TestFunctionalResult` classes. Update `TestBenchmarkResult` to not reference `functional`:

```python
# tests/benchmarks/test_models.py — after (showing only changed parts)

import unittest
import numpy as np
from benchmarks.models import (
    Scenario, GeneratorBenchmark,
    PerformanceStats, BenchmarkResult, RunMeta
)


class TestScenario(unittest.TestCase):
    def test_create_scenario(self):
        def factory(rng):
            return np.zeros((10, 2)), None

        s = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=factory,
            tags=["small", "2D"],
        )
        self.assertEqual(s.name, "test_scenario")
        self.assertEqual(s.data_type, "tabular")
        self.assertEqual(s.tags, ["small", "2D"])
        X, y = s.factory(np.random.default_rng(0))
        self.assertEqual(X.shape, (10, 2))
        self.assertIsNone(y)

    def test_scenario_default_tags(self):
        s = Scenario(
            name="test",
            data_type="tabular",
            factory=lambda rng: (None, None),
        )
        self.assertEqual(s.tags, [])


class TestGeneratorBenchmark(unittest.TestCase):
    def test_create_benchmark(self):
        s = Scenario("s1", "tabular", lambda rng: (None, None))

        gb = GeneratorBenchmark(
            generator_cls=type("FakeGen", (), {}),
            name="FakeGen",
            module_path="tabular_data.test",
            default_params={"p": 1},
            scenarios=[s],
        )
        self.assertEqual(gb.name, "FakeGen")
        self.assertEqual(gb.module_path, "tabular_data.test")
        self.assertEqual(len(gb.scenarios), 1)


class TestPerformanceStats(unittest.TestCase):
    def test_create_stats(self):
        ps = PerformanceStats(
            min=1.0, max=5.0, mean=3.0, median=2.5, stddev=1.5, iterations=5
        )
        self.assertEqual(ps.min, 1.0)
        self.assertEqual(ps.iterations, 5)


class TestBenchmarkResult(unittest.TestCase):
    def test_create_result(self):
        ps = PerformanceStats(1, 2, 1.5, 1.5, 0.5, 5)
        br = BenchmarkResult(
            generator="tabular_data.test.FakeGen",
            scenario="test_scenario",
            params={"p": 1},
            performance={"time_ms": ps, "memory_mb": ps},
        )
        self.assertEqual(br.generator, "tabular_data.test.FakeGen")
        self.assertEqual(br.performance["time_ms"].mean, 1.5)


class TestRunMeta(unittest.TestCase):
    def test_create_meta(self):
        rm = RunMeta(
            timestamp="2026-06-22T14:30:00",
            git_commit="abc1234",
            git_branch="main",
            python_version="3.12.4",
            platform="Windows-10",
        )
        self.assertEqual(rm.git_branch, "main")
```

- [ ] **Step 4: Update test_cli.py — remove functional references**

```python
# tests/benchmarks/test_cli.py — after

import unittest
import json
import tempfile
import pathlib
from unittest.mock import patch, MagicMock
from benchmarks.cli import build_parser, cmd_run


class TestCLIParsing(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_run_defaults(self):
        args = self.parser.parse_args(["run"])
        self.assertIsNone(args.generators)

    def test_run_with_filter(self):
        args = self.parser.parse_args(["run", "--generators", "tabular_data"])
        self.assertEqual(args.generators, "tabular_data")

    def test_baseline_save(self):
        args = self.parser.parse_args(["baseline", "save"])
        self.assertEqual(args.name, "latest")

    def test_baseline_save_named(self):
        args = self.parser.parse_args(["baseline", "save", "--name", "v1.0"])
        self.assertEqual(args.name, "v1.0")

    def test_baseline_list(self):
        args = self.parser.parse_args(["baseline", "list"])
        self.assertEqual(args.baseline_command, "list")

    def test_compare_default(self):
        args = self.parser.parse_args(["compare"])
        self.assertIsNone(args.baseline)
        self.assertIsNone(args.target)

    def test_compare_with_baseline(self):
        args = self.parser.parse_args(["compare", "--baseline", "v0.0.13"])
        self.assertEqual(args.baseline, "v0.0.13")


class TestCmdRun(unittest.TestCase):
    @patch("benchmarks.cli.discover")
    @patch("benchmarks.cli.run_all")
    def test_cmd_run_saves_results(self, mock_run_all, mock_discover):
        mock_discover.return_value = [MagicMock()]
        mock_run_all.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("benchmarks.cli.RESULTS_DIR", pathlib.Path(tmpdir)):
                args = MagicMock()
                args.generators = None
                args.iterations = 5
                args.timeout = 60
                cmd_run(args)

                json_files = list(pathlib.Path(tmpdir).glob("*.json"))
                self.assertEqual(len(json_files), 1)
```

- [ ] **Step 5: Run benchmark tests to verify all pass**

Run: `pytest tests/benchmarks/ -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add benchmarks/checks/ tests/benchmarks/
git commit -m "refactor: delete checks directory, update benchmark tests"
```

---

### Task 5: Create shared pytest fixtures (conftest.py files)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/generators/tabular_data/conftest.py`
- Create: `tests/generators/time_series/conftest.py`
- Create: `tests/generators/graph/conftest.py`
- Create: `tests/generators/text/conftest.py`
- Delete: `tests/generators/tabular_data/__init__.py` (old helper module)

**Interfaces:**
- Consumes: nothing
- Produces: `rng`, `tabular_data`, `tabular_data_labeled`, `time_series_sine`, `time_series_walk`, `graph_erdos_renyi`, `text_word_list` fixtures

- [ ] **Step 1: Create tests/conftest.py with shared rng fixture**

```python
# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
```

- [ ] **Step 2: Create tests/generators/tabular_data/conftest.py**

```python
# tests/generators/tabular_data/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["numpy_1D", "numpy_2D", "pandas_1D", "pandas_2D"])
def tabular_data(request, rng):
    """Parametrized fixture yielding (X, y=None) for each input type."""
    if request.param == "numpy_1D":
        return rng.normal(size=100).reshape(-1, 1), None
    elif request.param == "numpy_2D":
        return rng.normal(size=(100, 10)), None
    elif request.param == "pandas_1D":
        return pd.DataFrame(
            rng.normal(size=100).reshape(-1, 1), columns=["col"]
        ), None
    elif request.param == "pandas_2D":
        return pd.DataFrame(
            rng.normal(size=(100, 10)),
            columns=[f"col{i}" for i in range(10)],
        ), None


@pytest.fixture(params=["numpy_1D", "numpy_2D", "pandas_1D", "pandas_2D"])
def tabular_data_labeled(request, rng):
    """Parametrized fixture yielding (X, y) with 5-class classification labels."""
    if request.param == "numpy_1D":
        return (
            rng.normal(size=100).reshape(-1, 1),
            np.array([i % 5 for i in range(100)]),
        )
    elif request.param == "numpy_2D":
        return (
            rng.normal(size=(100, 10)),
            np.array([0, 1, 2, 3, 4] * 20),
        )
    elif request.param == "pandas_1D":
        return (
            pd.DataFrame(
                rng.normal(size=100).reshape(-1, 1), columns=["col"]
            ),
            pd.Series([i % 5 for i in range(100)]),
        )
    elif request.param == "pandas_2D":
        return (
            pd.DataFrame(
                rng.normal(size=(100, 10)),
                columns=[f"col{i}" for i in range(10)],
            ),
            pd.Series([0, 1, 2, 3, 4] * 20),
        )
```

- [ ] **Step 3: Create tests/generators/time_series/conftest.py**

```python
# tests/generators/time_series/conftest.py
import numpy as np
import pytest


@pytest.fixture
def time_series_sine():
    """200-point sine wave as (X, None)."""
    t = np.linspace(0, 4 * np.pi, 200)
    X = np.sin(t).reshape(-1, 1)
    return X, None


@pytest.fixture
def time_series_walk(rng):
    """200-point random walk as (X, None)."""
    steps = rng.normal(0, 1, size=(200, 1))
    X = np.cumsum(steps, axis=0)
    return X, None
```

- [ ] **Step 4: Create tests/generators/graph/conftest.py**

```python
# tests/generators/graph/conftest.py
import networkx as nx
import pytest


@pytest.fixture
def graph_erdos_renyi():
    """100-node Erdős-Rényi graph as (G, None)."""
    G = nx.erdos_renyi_graph(n=100, p=0.25, seed=0, directed=False)
    return G, None
```

- [ ] **Step 5: Create tests/generators/text/conftest.py**

```python
# tests/generators/text/conftest.py
import pytest


_WORDS = [
    "algorithm", "benchmark", "computation", "database", "experiment",
    "framework", "generator", "hypothesis", "implementation", "kernel",
    "library", "machine", "network", "optimization", "pipeline",
    "quantum", "regression", "statistics", "transformer", "validation",
]


@pytest.fixture
def text_word_list():
    """List of 20 technical words as (words, None)."""
    return list(_WORDS), None
```

- [ ] **Step 6: Delete old helper module**

```bash
rm tests/generators/tabular_data/__init__.py
```

- [ ] **Step 7: Verify fixtures are discoverable**

Run: `pytest --collect-only tests/generators/tabular_data/ 2>&1 | head -5`
Expected: pytest discovers tests (they'll fail since tests still use old patterns, but fixtures should be found)

- [ ] **Step 8: Commit**

```bash
git add tests/conftest.py tests/generators/tabular_data/conftest.py tests/generators/time_series/conftest.py tests/generators/graph/conftest.py tests/generators/text/conftest.py
git rm tests/generators/tabular_data/__init__.py
git commit -m "feat: add pytest fixtures for test data generation"
```

---

### Task 6: Convert tabular data tests to pytest functions

**Files:**
- Modify: `tests/generators/tabular_data/test_noise.py`
- Modify: `tests/generators/tabular_data/test_outliers.py`
- Modify: `tests/generators/tabular_data/test_drift.py`
- Modify: `tests/generators/tabular_data/test_imbalance.py`
- Modify: `tests/generators/tabular_data/test_missingness.py`

**Interfaces:**
- Consumes: `tabular_data`, `tabular_data_labeled` fixtures
- Produces: passing tabular generator tests

- [ ] **Step 1: Convert test_noise.py**

```python
# tests/generators/tabular_data/test_noise.py
import numpy as np
from badgers.generators.tabular_data.noise import (
    GaussianNoiseGenerator, GaussianNoiseClassesGenerator,
)


def test_gaussian_noise__preserves_shape(tabular_data):
    X, y = tabular_data
    generator = GaussianNoiseGenerator()
    Xt, _ = generator.generate(X.copy(), y=None, noise_std=1)
    assert len(X) == len(Xt)


def test_gaussian_noise__increases_variance(tabular_data):
    X, y = tabular_data
    generator = GaussianNoiseGenerator()
    Xt, _ = generator.generate(X.copy(), y=None, noise_std=1)
    assert (np.var(Xt, axis=0) > np.var(X, axis=0)).all()


def test_gaussian_noise_classes__preserves_shape(tabular_data_labeled):
    X, y = tabular_data_labeled
    noise_std_per_class = {label: 0.1 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator()
    Xt, yt = generator.generate(X.copy(), y, noise_std_per_class=noise_std_per_class)
    assert len(X) == len(Xt)
    assert len(y) == len(yt)
```

- [ ] **Step 2: Convert test_outliers.py**

```python
# tests/generators/tabular_data/test_outliers.py
import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.generators.tabular_data.outliers.distribution_sampling import (
    ZScoreSamplingGenerator, HypersphereSamplingGenerator, HyperCubeSampling,
)
from badgers.generators.tabular_data.outliers.low_density_sampling import (
    HistogramSamplingGenerator, LowDensitySamplingGenerator,
    IndependentHistogramsGenerator,
)

COMMON_GENERATORS = [
    ("hyper_cube", HyperCubeSampling, {"expansion": 0.0}),
    ("z_score", ZScoreSamplingGenerator, {"scale": 1.0}),
    ("hypersphere", HypersphereSamplingGenerator, {"scale": 1.0}),
    ("histogram", HistogramSamplingGenerator, {"bins": 3, "threshold_low_density": 0.5}),
    ("low_density", LowDensitySamplingGenerator, {}),
    ("independent_histograms", IndependentHistogramsGenerator, {"bins": 3}),
]


def test_outliers__correct_shape_and_labels(tabular_data):
    """For each generator and input type: outliers shape is correct, yt has right length."""
    n_outliers = 10
    X, y = tabular_data

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        generator = gen_class(random_generator=default_rng(0))
        X_original = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)

        try:
            outliers, yt = generator.generate(
                X.copy(), y, n_outliers=n_outliers, **default_kwargs
            )
        except NotImplementedError:
            continue

        # yt has correct length
        assert len(yt) == len(outliers)

        # outliers have correct shape
        if isinstance(X, pd.DataFrame):
            assert outliers.shape[1] == X.shape[1]
        else:
            assert outliers.shape[1] == (X.shape[1] if X.ndim > 1 else 1)

        assert outliers.shape[0] == n_outliers
```

- [ ] **Step 3: Convert test_drift.py**

```python
# tests/generators/tabular_data/test_drift.py
import numpy as np
from badgers.generators.tabular_data.drift import (
    RandomShiftGenerator, RandomShiftClassesGenerator,
)


def test_random_shift__preserves_shape_scalar_std(tabular_data):
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=np.random.default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=0.1)
    assert len(X) == len(Xt)


def test_random_shift__preserves_shape_array_std(tabular_data):
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=np.random.default_rng(0))
    if X.ndim == 1 or X.shape[1] == 1:
        shift_std = np.array([0.1])
    else:
        shift_std = np.linspace(0.1, 1, X.shape[1])
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)


def test_random_shift_classes__preserves_shape_scalar_std(tabular_data_labeled):
    X, y = tabular_data_labeled
    generator = RandomShiftClassesGenerator(random_generator=np.random.default_rng(0))
    Xt, yt = generator.generate(X.copy(), y, shift_std=0.1)
    assert len(X) == len(Xt)


def test_random_shift_classes__preserves_shape_array_std(tabular_data_labeled):
    X, y = tabular_data_labeled
    generator = RandomShiftClassesGenerator(random_generator=np.random.default_rng(0))
    if X.ndim == 1 or X.shape[1] == 1:
        shift_std = 0.1
    else:
        shift_std = np.linspace(0.1, 1, 5)
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)


def test_random_shift_classes__preserves_shape_2d_array_std():
    """Only for 2D pandas input with per-class per-feature shift_std."""
    import pandas as pd
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(100, 10)),
        columns=[f"col{i}" for i in range(10)],
    )
    y = pd.Series([0, 1, 2, 3, 4] * 20)
    shift_std = np.array([[i] * 10 for i in np.linspace(0.1, 1, 5)])
    generator = RandomShiftClassesGenerator(random_generator=np.random.default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)
```

- [ ] **Step 4: Convert test_imbalance.py**

```python
# tests/generators/tabular_data/test_imbalance.py
import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba
from badgers.generators.tabular_data.imbalance import (
    RandomSamplingFeaturesGenerator, RandomSamplingClassesGenerator,
)


def test_random_sampling_classes__preserves_structure(tabular_data_labeled):
    X, y = tabular_data_labeled
    unique_labels = np.unique(y)
    proportion_classes = {label: 1.0 / len(unique_labels) for label in unique_labels}
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(X.copy(), y, proportion_classes=proportion_classes)
    assert Xt.shape[0] == len(yt)
    # pandas columns preserved
    if hasattr(X, "columns") and hasattr(Xt, "columns"):
        assert list(X.columns) == list(Xt.columns)


def test_random_sampling_features__preserves_structure(tabular_data_labeled):
    X, y = tabular_data_labeled

    def proba_func(X_in):
        feature = X_in.iloc[:, 0] if hasattr(X_in, "iloc") else X_in[:, 0]
        return normalize_proba(
            (np.max(feature) - feature) / (np.max(feature) - np.min(feature))
        )

    generator = RandomSamplingFeaturesGenerator()
    Xt, yt = generator.generate(X.copy(), y, sampling_proba_func=proba_func)
    assert Xt.shape[0] == len(yt)
    if hasattr(X, "columns") and hasattr(Xt, "columns"):
        assert list(X.columns) == list(Xt.columns)
```

- [ ] **Step 5: Convert test_missingness.py**

```python
# tests/generators/tabular_data/test_missingness.py
import numpy as np
from badgers.generators.tabular_data.missingness import MissingValueGenerator


def test_missingness__correct_nan_count(tabular_data):
    X, y = tabular_data
    percentage_missing = 0.1

    for cls in MissingValueGenerator.__subclasses__():
        transformer = cls()
        Xt, _ = transformer.generate(X.copy(), y, percentage_missing=percentage_missing)

        # shape preserved
        assert Xt.shape == X.shape

        # correct number of NaNs
        total_cells = X.shape[0] * (X.shape[1] if X.ndim > 1 else 1)
        expected_nans = int(percentage_missing * total_cells)
        assert np.sum(np.isnan(Xt).sum()) == expected_nans
```

- [ ] **Step 6: Run tabular tests to verify all pass**

Run: `pytest tests/generators/tabular_data/ -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add tests/generators/tabular_data/
git commit -m "refactor: convert tabular data tests to pytest functions with fixtures"
```

---

### Task 7: Convert time series tests to pytest functions

**Files:**
- Modify: `tests/generators/time_series/test_noise.py`
- Modify: `tests/generators/time_series/test_outliers.py`
- Modify: `tests/generators/time_series/test_changepoints.py`
- Modify: `tests/generators/time_series/test_missingness.py`
- Modify: `tests/generators/time_series/test_patterns.py`
- Modify: `tests/generators/time_series/test_seasons.py`
- Modify: `tests/generators/time_series/test_trends.py`
- Modify: `tests/generators/time_series/test_transmission_errors.py`
- Modify: `tests/generators/time_series/test_utils.py`

**Interfaces:**
- Consumes: `time_series_sine`, `time_series_walk` fixtures
- Produces: passing time series generator tests

- [ ] **Step 1: Convert test_noise.py**

```python
# tests/generators/time_series/test_noise.py
from badgers.generators.time_series.noise import (
    LocalGaussianNoiseGenerator, GlobalGaussianNoiseGenerator,
)


def test_local_gaussian_noise__generates_output(time_series_sine):
    X, y = time_series_sine
    generator = LocalGaussianNoiseGenerator()
    Xt, _ = generator.generate(X, None)
    assert Xt.shape == X.shape


def test_global_gaussian_noise__generates_output(time_series_sine):
    X, y = time_series_sine
    generator = GlobalGaussianNoiseGenerator()
    Xt, _ = generator.generate(X, None)
    assert Xt.shape == X.shape
```

- [ ] **Step 2: Convert test_outliers.py**

```python
# tests/generators/time_series/test_outliers.py
import numpy as np
from badgers.generators.time_series.outliers import (
    RandomZerosGenerator, LocalZScoreGenerator,
)


def test_random_zeros__correct_count(time_series_sine):
    X, y = time_series_sine
    n_outliers = 10
    generator = RandomZerosGenerator()
    Xt, _ = generator.generate(X=X, y=None, n_outliers=n_outliers)
    assert Xt.shape == X.shape
    assert len(generator.outliers_indices_) == n_outliers


def test_local_zscore__correct_count(time_series_sine):
    X, y = time_series_sine
    n_outliers = 10
    generator = LocalZScoreGenerator()
    Xt, _ = generator.generate(X=X, y=None, n_outliers=n_outliers)
    assert Xt.shape == X.shape
    assert len(generator.outliers_indices_) == n_outliers
    assert not Xt.isna().any()[0]
```

- [ ] **Step 3: Convert test_changepoints.py**

```python
# tests/generators/time_series/test_changepoints.py
import numpy as np
import pandas as pd
from numpy.random import default_rng
from badgers.generators.time_series.changepoints import RandomChangeInMeanGenerator


def test_random_change_in_mean__modifies_data():
    generator = RandomChangeInMeanGenerator(random_generator=default_rng(seed=0))
    X = pd.DataFrame(data=np.zeros(100), columns=["dimension_0"], dtype=float)
    Xt, _ = generator.generate(
        X.copy(), None, n_changepoints=10, min_change=-5, max_change=5,
    )
    assert (X != Xt).any().any()
```

- [ ] **Step 4: Convert test_missingness.py**

```python
# tests/generators/time_series/test_missingness.py
from badgers.generators.time_series.missingness import MissingAtRandomGenerator


def test_missing_at_random__correct_count(time_series_sine):
    X, y = time_series_sine
    n_missing = 10
    generator = MissingAtRandomGenerator()
    Xt, _ = generator.generate(X=X, y=None, n_missing=n_missing)
    assert Xt.shape == X.shape
    assert len(generator.missing_indices_) == n_missing
    assert Xt.isna().sum()[0] == n_missing
```

- [ ] **Step 5: Convert test_patterns.py**

```python
# tests/generators/time_series/test_patterns.py
import numpy as np
import pandas as pd
from badgers.generators.time_series.patterns import (
    Pattern, add_offset, add_linear_trend, scale,
    RandomlySpacedPatterns, RandomlySpacedConstantPatterns,
    RandomlySpacedLinearPatterns,
)


class TestPattern:
    def test_resample(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        resampled = pattern.resample(10)
        assert len(resampled) == 10
        assert abs(resampled[0] - 1) < 1e-10
        assert abs(resampled[-1] - 5) < 1e-10

    def test_add_offset(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        expected = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
        result = add_offset(pattern.values, 2)
        assert result.tolist() == expected.tolist()

    def test_add_linear_trend(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        result = add_linear_trend(pattern.values, start_value=0, end_value=-1)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == -1

    def test_scale(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        expected = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
        result = scale(pattern.values, scaling_factor=2)
        assert result.tolist() == expected.tolist()


def test_randomly_spaced_patterns__1d(time_series_sine):
    X, y = time_series_sine
    n_patterns = 3
    pattern = Pattern(values=np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]))
    generator = RandomlySpacedPatterns()
    Xt, _ = generator.generate(
        X=X, y=None, n_patterns=n_patterns,
        min_width_pattern=5, max_width_patterns=10, pattern=pattern,
    )
    assert len(generator.patterns_indices_) == n_patterns
    for i in range(n_patterns - 1):
        assert generator.patterns_indices_[i][1] < generator.patterns_indices_[i + 1][0]


def test_randomly_spaced_patterns__2d():
    X = np.zeros(shape=(100, 2))
    n_patterns = 3
    pattern = Pattern(values=np.array([
        [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
        [0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0],
    ]).T)
    generator = RandomlySpacedPatterns()
    Xt, _ = generator.generate(
        X=X, y=None, n_patterns=n_patterns,
        min_width_pattern=5, max_width_patterns=10, pattern=pattern,
    )
    assert len(generator.patterns_indices_) == n_patterns
    for i in range(n_patterns - 1):
        assert generator.patterns_indices_[i][1] < generator.patterns_indices_[i + 1][0]


def test_randomly_spaced_constant_patterns__all_zeros():
    t = np.linspace(1, 10, 101)
    X = pd.DataFrame(data=(np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1))
    n_patterns = 3
    generator = RandomlySpacedConstantPatterns()
    Xt, _ = generator.generate(
        X=X, y=None, n_patterns=n_patterns,
        min_width_pattern=5, max_width_patterns=10, constant_value=0,
    )
    assert len(generator.patterns_indices_) == n_patterns
    for start, end in generator.patterns_indices_:
        assert Xt.iloc[start:end, :].values.tolist() == np.zeros((end - start, X.shape[1])).tolist()


def test_randomly_spaced_linear_patterns__linear_segments():
    t = np.linspace(1, 10, 101)
    X = pd.DataFrame(data=(np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1))
    generator = RandomlySpacedLinearPatterns()
    Xt, _ = generator.generate(
        X=X, y=None, n_patterns=3,
        min_width_pattern=5, max_width_patterns=10,
    )
    for start, end in generator.patterns_indices_:
        for col in range(X.shape[1]):
            expected = np.linspace(
                X.iloc[start, col], X.iloc[end, col], end - start,
            )
            assert Xt.iloc[start:end, col].tolist() == expected.tolist()
```

- [ ] **Step 6: Convert test_seasons.py**

```python
# tests/generators/time_series/test_seasons.py
import numpy as np
from numpy.random import default_rng
from badgers.generators.time_series.seasons import GlobalAdditiveSinusoidalSeasonGenerator


def test_sinusoidal_season__matches_expected():
    X = np.zeros(shape=100)
    period = 10
    generator = GlobalAdditiveSinusoidalSeasonGenerator(random_generator=default_rng(seed=0))
    Xt, yt = generator.generate(X=X, y=None, period=period)

    t = np.arange(100)
    season = np.sin(t * 2 * np.pi / period)
    expected = (X + season).reshape(-1, 1)

    assert Xt.values.tolist() == expected.tolist()
    assert yt is None
```

- [ ] **Step 7: Convert test_trends.py**

```python
# tests/generators/time_series/test_trends.py
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas._testing import assert_frame_equal
from badgers.generators.time_series.trends import (
    GlobalAdditiveLinearTrendGenerator, AdditiveLinearTrendGenerator,
    RandomlySpacedLinearTrends,
)


def test_global_additive_linear_trend__matches_expected():
    X = pd.DataFrame(
        data=np.zeros(shape=(10, 4)),
        columns=[f"col{i}" for i in range(4)],
    )
    slope = np.array([1, 2, 3, 4])
    generator = GlobalAdditiveLinearTrendGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X, None, slope=slope)

    expected = pd.DataFrame(
        data=np.array([np.linspace(0, len(X) * s, len(X)) for s in slope]).T,
        columns=X.columns, index=X.index,
    )
    assert_frame_equal(Xt, expected)


def test_additive_linear_trend__matches_expected():
    X = pd.DataFrame(
        data=np.zeros(shape=(10, 4)),
        columns=[f"col{i}" for i in range(4)],
    )
    slope = np.array([0, 0.5, 1, 2])
    generator = AdditiveLinearTrendGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X, None, slope=slope, start=3, end=7)

    expected = pd.DataFrame(
        data=np.array([
            [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
            [0., 0., 0., 0.], [0., 2./3., 4./3., 8./3.],
            [0., 4./3., 8./3., 16./3.], [0., 2., 4., 8.],
            [0., 2., 4., 8.], [0., 2., 4., 8.], [0., 2., 4., 8.],
        ]),
        columns=X.columns, index=X.index,
    )
    assert_frame_equal(Xt, expected)


def test_randomly_spaced_linear_trends__gaps_are_constant():
    X = pd.DataFrame(
        data=np.zeros(shape=(100, 4)),
        columns=[f"col{i}" for i in range(4)],
    )
    generator = RandomlySpacedLinearTrends(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(
        X, None, n_patterns=5, min_width_pattern=5, max_width_patterns=10,
    )
    for i in range(1, len(generator.patterns_indices_)):
        s = generator.patterns_indices_[i - 1][1]
        e = generator.patterns_indices_[i][0]
        assert Xt[s:e].diff().dropna().sum().sum() == 0
```

- [ ] **Step 8: Convert test_transmission_errors.py**

```python
# tests/generators/time_series/test_transmission_errors.py
import pandas as pd
from numpy.random import default_rng
from badgers.generators.time_series.transmission_errors import (
    RandomTimeSwitchGenerator, RandomRepeatGenerator,
    RandomDropGenerator, LocalRegionsRandomDropGenerator,
    LocalRegionsRandomRepeatGenerator,
)


def test_random_time_switch__no_switch_raises():
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
    try:
        generator.generate(X.copy(), y=None, n_switches=0)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_random_time_switch__single_switch():
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_switches=1)
    assert set(X[0]) == set(Xt[0])
    assert (X != Xt).sum().values[0] == 2


def test_random_time_switch__single_switch_frame():
    X = pd.DataFrame(data=[
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
        [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
    ])
    generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_switches=1)
    assert set(X[0]) == set(Xt[0])
    assert (X != Xt).sum().values[0] == 2


def test_random_repeat__no_repeat_raises():
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
    try:
        generator.generate(X.copy(), y=None, n_repeats=0, min_nb_repeats=2, max_nb_repeats=3)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_random_repeat__two_repeats():
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_repeats=2, min_nb_repeats=2, max_nb_repeats=3)
    assert Xt.shape[0] == X.shape[0] + 2 * 2
    assert set(X[0]) == set(Xt[0])


def test_local_regions_random_repeat__single():
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(
        X.copy(), y=None, n_repeats=1, min_nb_repeats=2, max_nb_repeats=3,
        n_regions=1, min_width_regions=3, max_width_regions=7,
    )
    assert Xt.shape[0] == X.shape[0] + 2
    assert set(X[0]) == set(Xt[0])


def test_local_regions_random_repeat__many():
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(
        X.copy(), y=None, n_repeats=4, min_nb_repeats=2, max_nb_repeats=3,
        n_regions=2, min_width_regions=3, max_width_regions=5,
    )
    assert Xt.shape[0] == X.shape[0] + 2 * 4
    assert set(X[0]) == set(Xt[0])
```

- [ ] **Step 9: Convert test_utils.py**

```python
# tests/generators/time_series/test_utils.py
import numpy as np
from badgers.generators.time_series.utils import generate_random_patterns_indices


def test_generate_random_patterns_indices__no_patterns_raises():
    rng = np.random.default_rng(0)
    try:
        generate_random_patterns_indices(
            random_generator=rng, signal_size=100,
            n_patterns=0, min_width_pattern=5, max_width_patterns=10,
        )
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_generate_random_patterns_indices__single_pattern():
    rng = np.random.default_rng(0)
    signal_size = 100
    indices = generate_random_patterns_indices(
        random_generator=rng, signal_size=signal_size,
        n_patterns=1, min_width_pattern=5, max_width_patterns=10,
    )
    assert len(indices) == 1
    start, end = indices[0]
    assert start < end
    assert end - start >= 5
    assert end - start < 10
    assert 0 <= start < signal_size
    assert 0 <= end < signal_size


def test_generate_random_patterns_indices__multiple():
    rng = np.random.default_rng(0)
    signal_size = 100
    n_patterns = 5
    indices = generate_random_patterns_indices(
        random_generator=rng, signal_size=signal_size,
        n_patterns=n_patterns, min_width_pattern=5, max_width_patterns=10,
    )
    assert len(indices) == n_patterns
    for start, end in indices:
        assert start < end
        assert end - start >= 5
        assert end - start < 10
        assert 0 <= start < signal_size
        assert 0 <= end < signal_size
```

- [ ] **Step 10: Run time series tests to verify all pass**

Run: `pytest tests/generators/time_series/ -v`
Expected: all tests PASS

- [ ] **Step 11: Commit**

```bash
git add tests/generators/time_series/
git commit -m "refactor: convert time series tests to pytest functions with fixtures"
```

---

### Task 8: Convert graph and text tests to pytest functions

**Files:**
- Modify: `tests/generators/graph/test_missingness.py`
- Modify: `tests/generators/text/test_typos.py`

**Interfaces:**
- Consumes: `graph_erdos_renyi`, `text_word_list` fixtures
- Produces: passing graph and text generator tests

- [ ] **Step 1: Convert test_missingness.py (graph)**

```python
# tests/generators/graph/test_missingness.py
import numpy as np
from badgers.generators.graph.missingness import (
    NodesMissingCompletelyAtRandom, EdgesMissingCompletelyAtRandom,
)


def test_nodes_mcar__correct_count(graph_erdos_renyi):
    G, y = graph_erdos_renyi
    percentage_missing = 0.1
    generator = NodesMissingCompletelyAtRandom(random_generator=np.random.default_rng(0))
    Xt, _ = generator.generate(X=G, y=None, percentage_missing=percentage_missing)
    assert len(Xt) == len(G) - 10


def test_nodes_mcar__with_labels(graph_erdos_renyi):
    G, _ = graph_erdos_renyi
    percentage_missing = 0.1
    generator = NodesMissingCompletelyAtRandom(random_generator=np.random.default_rng(0))
    Xt, yt = generator.generate(G, [0] * len(G), percentage_missing=percentage_missing)
    assert len(Xt) == len(G) - 10
    assert len(yt) == len(G) - 10


def test_edges_mcar__correct_count(graph_erdos_renyi):
    G, y = graph_erdos_renyi
    percentage_missing = 0.1
    generator = EdgesMissingCompletelyAtRandom(random_generator=np.random.default_rng(0))
    Xt, _ = generator.generate(X=G, y=None, percentage_missing=percentage_missing)
    assert Xt.number_of_nodes() == G.number_of_nodes()
    expected_edges = len(G.edges()) - int(G.number_of_edges() * percentage_missing)
    assert Xt.number_of_edges() == expected_edges
```

- [ ] **Step 2: Convert test_typos.py (text)**

```python
# tests/generators/text/test_typos.py
from copy import deepcopy
from numpy.random import default_rng
from badgers.generators.text.typos import (
    SwapLettersGenerator, LeetSpeakGenerator, SwapCaseGenerator,
)


def test_swap_letters__long_words_modified(text_word_list):
    X, y = text_word_list
    trf = SwapLettersGenerator(random_generator=default_rng(0))
    Xt, _ = trf.generate(X=deepcopy(X), y=None, swap_proba=1)

    for original, transformed in zip(X, Xt):
        if len(original) > 3:
            assert transformed != original
            assert transformed[0] == original[0]
            assert transformed[-1] == original[-1]
            assert len(transformed) == len(original)
            assert set(transformed) == set(original)
        else:
            assert transformed == original


def test_swap_letters__short_words_unchanged():
    X = ["abc", "ab", "a"]
    trf = SwapLettersGenerator(random_generator=default_rng(0))
    Xt, _ = trf.generate(X=deepcopy(X), y=None, swap_proba=1)
    assert Xt == X


def test_leet_speak__preserves_length():
    X = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "fox", " <> "]
    trf = LeetSpeakGenerator()
    Xt, _ = trf.generate(X, None)
    assert len(X) == len(Xt)


def test_swap_case__all_uppercase():
    X = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "fox", " <> "]
    trf = SwapCaseGenerator()
    Xt, _ = trf.generate(X, None, swapcase_proba=1.0)
    assert len(X) == len(Xt)
    for w1, w2 in zip(X, Xt):
        assert w1.upper() == w2
```

- [ ] **Step 3: Run graph and text tests to verify all pass**

Run: `pytest tests/generators/graph/ tests/generators/text/ -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/generators/graph/ tests/generators/text/
git commit -m "refactor: convert graph and text tests to pytest functions with fixtures"
```

---

### Task 9: Final verification — run full test suite

**Files:** none (verification only)

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Run benchmarks to verify they still work**

Run: `python -m benchmarks run --iterations 2`
Expected: benchmarks run, results saved to `benchmarks/results/`

- [ ] **Step 3: Commit any final cleanup**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: final verification, all tests and benchmarks pass"
```
