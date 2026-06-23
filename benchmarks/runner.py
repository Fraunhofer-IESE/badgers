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
    """Run performance benchmarks (convenience alias for run_performance)."""
    return run_performance(benchmarks, filter_path, iterations, timeout)