"""Data models for the benchmark framework."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np


@dataclass
class Scenario:
    """A named data factory that produces (X, y) for benchmarking."""
    name: str
    data_type: str  # "tabular", "time_series", "graph", "text"
    factory: Callable[[np.random.Generator], Tuple[Any, Any]]
    tags: List[str] = field(default_factory=list)


@dataclass
class FunctionalCheck:
    """A single assertion about a generator's output."""
    name: str
    description: str
    check: Callable[[Any, Any, Any, Any, Dict], bool]
    # check(original_X, original_y, transformed_X, transformed_y, params) -> bool


@dataclass
class GeneratorBenchmark:
    """Registration unit for one generator class."""
    generator_cls: Type
    name: str
    module_path: str  # e.g. "tabular_data.noise"
    default_params: Dict
    scenarios: List[Scenario]
    functional_checks: List[FunctionalCheck]


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
class FunctionalResult:
    """Result of running functional checks for one benchmark."""
    passed: int
    failed: int
    checks: List[Dict]  # list of {"name": str, "passed": bool}


@dataclass
class BenchmarkResult:
    """Complete result for one (generator, scenario) combination."""
    generator: str
    scenario: str
    params: Dict
    functional: Optional[FunctionalResult] = None
    performance: Optional[Dict[str, PerformanceStats]] = None
    # performance keys: "time_ms", "memory_mb"


@dataclass
class RunMeta:
    """Metadata about a benchmark run."""
    timestamp: str
    git_commit: str
    git_branch: str
    python_version: str
    platform: str