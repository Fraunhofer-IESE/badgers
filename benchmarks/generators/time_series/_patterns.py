"""Benchmark registrations for time series pattern generators."""
import numpy as np
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from badgers.generators.time_series.patterns import RandomlySpacedPatterns, Pattern

register(GeneratorBenchmark(
    generator_cls=RandomlySpacedPatterns,
    name="RandomlySpacedPatterns",
    module_path="time_series.patterns",
    default_params={
        "n_patterns": 3, "min_width_pattern": 5, "max_width_patterns": 10,
        "pattern": Pattern(values=np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])),
    },
    scenarios=[SCENARIO_SINE_WAVE],
))
