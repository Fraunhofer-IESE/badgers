"""Data models for the benchmark framework."""
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