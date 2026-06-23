# Architecture and key principles

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

The `benchmarks` package provides a systematic framework for measuring functional correctness and performance (time/memory) of all generators. It lives outside the main `badgers` source to avoid coupling.

### Architecture (3 Layers)

1. **Registration Layer** (`models.py`, `registry.py`, `scenarios/`, `checks/`, `generators/`):
   - `Scenario` dataclasses define reusable data factories for each data type (tabular, time series, graph, text).
   - `FunctionalCheck` callables assert expected properties (e.g., same shape, no NaNs, increased variance).
   - `GeneratorBenchmark` ties a generator class to its scenarios and checks.
   - The `Registry` auto-discovers `_*.py` registration modules from `benchmarks/generators/`.

2. **Execution Layer** (`runner.py`, `cli.py`):
   - `run_functional()` runs all functional checks and reports pass/fail per check.
   - `run_performance()` measures wall-clock time and peak memory over multiple iterations.
   - CLI provides `run`, `baseline`, and `compare` subcommands.

3. **Analysis Layer** (`comparator.py`):
   - `compare_results()` diffs two result sets and flags regressions (>20% time, >30% memory).

### Usage

```bash
# Run all benchmarks (functional + performance)
python -m benchmarks run

# Run only functional checks
python -m benchmarks run --type functional

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
    functional_checks=["CHECK_SAME_SHAPE", "CHECK_NO_NANS"],
))
```

The registry auto-discovers all `_*.py` files — no other changes needed.