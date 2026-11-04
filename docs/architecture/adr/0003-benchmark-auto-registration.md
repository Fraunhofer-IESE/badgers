# ADR-0003: Benchmark registration via auto-discovered modules

- **Date:** 2026-06-24
- **Status:** Accepted

## Context

Benchmark definitions were coupled to the benchmark runner: adding a generator to the benchmark
suite meant editing a central list, which conflicted with the goal of keeping `benchmarks/` a
pure consumer of the library.

## Decision

Each generator category has a `_*.py` registration module in
`benchmarks/generators/<data_type>/` that imports generator classes and calls
`register(GeneratorBenchmark(...))`. The registry auto-discovers these modules.

```python
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from badgers.generators.tabular_data.noise import AdditiveNoiseGenerator

register(GeneratorBenchmark(
    generator_cls=AdditiveNoiseGenerator,
    name="AdditiveNoise",
    module_path="tabular_data.noise",
    default_params={"distribution": scipy.stats.norm(scale=0.5)},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
```

**Key constraint:** `OutlierSamplingGenerator` benchmarks using `JointHistogramSampler`
(registry name `"histogram"`) must only use `SCENARIO_SMALL_BLOBS` (2D), because joint histogram
sampling supports at most 5 columns.

## Consequences

- Adding a generator to benchmarks is a single `register()` call in the appropriate `_*.py`
  file — no other changes needed.
- Benchmark definitions are decoupled from the runner.
- Discovery is implicit: a typo in a filename silently drops the benchmark. Mitigated by
  `tests/benchmarks/test_registry.py`.
