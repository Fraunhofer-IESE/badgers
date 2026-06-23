"""Benchmark registrations for time series outlier generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from benchmarks.checks.common import CHECK_SAME_SHAPE
from badgers.generators.time_series.outliers import RandomZerosGenerator

register(GeneratorBenchmark(
    generator_cls=RandomZerosGenerator,
    name="RandomZeros",
    module_path="time_series.outliers",
    default_params={"n_outliers": 10},
    scenarios=[SCENARIO_SINE_WAVE],
    functional_checks=[CHECK_SAME_SHAPE],
))
