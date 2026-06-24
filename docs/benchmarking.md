# Benchmarking

Badgers includes a benchmarking framework to measure performance (time/memory) of all generators. It helps detect regressions when refactoring or optimizing code.

## Quick Start

```bash
# Run all benchmarks (performance)
python -m benchmarks run

# Run only performance measurements
python -m benchmarks run --type performance
```

## CLI Reference

### `run` — Execute Benchmarks

```
python -m benchmarks run [--type {performance,all}] [--generators FILTER] [--iterations N] [--timeout S]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `all` | `performance` (time/memory), or `all` |
| `--generators` | *(all)* | Filter by module path prefix, e.g. `tabular_data.outliers` |
| `--iterations` | `5` | Number of iterations for performance measurement |
| `--timeout` | `60` | Timeout in seconds per scenario |

Results are saved as JSON to `benchmarks/results/run_<branch>_<timestamp>.json`.

### `baseline` — Manage Baselines

```
python -m benchmarks baseline save [--name NAME]
python -m benchmarks baseline list
```

Baselines are snapshots of results used for regression detection. Save a baseline after verifying all checks pass on a known-good commit.

### `compare` — Detect Regressions

```
python -m benchmarks compare [--baseline NAME] [--target PATH]
```

Compares the latest (or specified) results against a baseline and reports regressions:

- **Time regression**: >20% increase in mean execution time
- **Memory regression**: >30% increase in peak memory usage

## What Gets Measured

### Performance Metrics

- **Wall-clock time**: min, max, mean, median, stddev over N iterations
- **Peak memory**: measured via `tracemalloc`

## Adding a New Generator

Create a registration file `_<name>.py` in `benchmarks/generators/<category>/`:

```python
from benchmarks.models import Scenario, GeneratorBenchmark
from benchmarks.registry import register

register(GeneratorBenchmark(
    class_name="MyGenerator",
    module="badgers.generators.tabular_data.noise",
    scenarios={
        "small": Scenario(
            factory="benchmarks.scenarios.tabular:make_blobs_small",
            params={},
        ),
    },
))
```

The registry auto-discovers all `_*.py` files — no other registration needed.

## Testing Generators

Generator correctness is tested separately from benchmarks using **pytest**. See the [Architecture](architecture.md#tests) page for the test structure and conventions.

## Architecture

See the [Architecture](architecture.md#benchmarks-module) page for the 3-layer design (Registration → Execution → Analysis).
