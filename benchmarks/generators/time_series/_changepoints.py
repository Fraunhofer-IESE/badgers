"""Benchmark registrations for time series changepoint generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from benchmarks.checks.common import CHECK_SAME_SHAPE
from badgers.generators.time_series.changepoints import RandomChangeInMeanGenerator

register(GeneratorBenchmark(
    generator_cls=RandomChangeInMeanGenerator,
    name="RandomChangeInMean",
    module_path="time_series.changepoints",
    default_params={"n_changepoints": 5, "min_change": -1, "max_change": 1},
    scenarios=[SCENARIO_SINE_WAVE],
    functional_checks=[CHECK_SAME_SHAPE],
))
