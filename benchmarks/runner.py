"""Benchmark runner: executes functional checks and performance measurements."""
import time
import tracemalloc
import warnings
from typing import List, Optional

import numpy as np

from benchmarks.models import (
    GeneratorBenchmark, BenchmarkResult, FunctionalResult, PerformanceStats,
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


def run_functional(
    benchmarks: List[GeneratorBenchmark],
    filter_path: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Run functional checks for all registered benchmarks.

    Args:
        benchmarks: List of GeneratorBenchmark registrations.
        filter_path: Optional module path prefix to filter generators.

    Returns:
        List of BenchmarkResult with functional results populated.
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
            try:
                Xt, yt = generator.generate(X, y, **gb.default_params)
            except NotImplementedError:
                warnings.warn(
                    f"Generator {gb.name} does not support scenario {scenario.name}"
                )
                continue
            except Exception as e:
                warnings.warn(
                    f"Generator {gb.name} raised {type(e).__name__}: {e}"
                )
                continue

            check_results = []
            passed = 0
            failed = 0
            for check in gb.functional_checks:
                try:
                    ok = check.check(X, y, Xt, yt, gb.default_params)
                except Exception:
                    ok = False
                if ok:
                    passed += 1
                else:
                    failed += 1
                check_results.append({"name": check.name, "passed": ok})

            results.append(BenchmarkResult(
                generator=f"{gb.module_path}.{gb.name}",
                scenario=scenario.name,
                params=gb.default_params,
                functional=FunctionalResult(
                    passed=passed,
                    failed=failed,
                    checks=check_results,
                ),
            ))

    return results


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

                time_values.append((t_end - t_start) * 1000)  # ms
                memory_values.append(peak / (1024 * 1024))     # MB

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
    """Run both functional checks and performance measurements.

    Returns combined results with both functional and performance data.
    """
    func_results = run_functional(benchmarks, filter_path)
    perf_results = run_performance(benchmarks, filter_path, iterations, timeout)

    # Merge by (generator, scenario) key
    perf_map = {(r.generator, r.scenario): r for r in perf_results}
    merged = []
    for fr in func_results:
        key = (fr.generator, fr.scenario)
        if key in perf_map:
            fr.performance = perf_map[key].performance
        merged.append(fr)

    # Add perf-only results (no functional checks defined)
    func_keys = {(r.generator, r.scenario) for r in func_results}
    for pr in perf_results:
        key = (pr.generator, pr.scenario)
        if key not in func_keys:
            merged.append(pr)

    return merged