"""Benchmark registrations for time series missingness generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from badgers.generators.time_series.missingness import MissingAtRandomGenerator

register(GeneratorBenchmark(
    generator_cls=MissingAtRandomGenerator,
    name="MissingAtRandom",
    module_path="time_series.missingness",
    default_params={"n_missing": 10},
    scenarios=[SCENARIO_SINE_WAVE],
))
