"""Benchmark registrations for time series season generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from badgers.generators.time_series.seasons import GlobalAdditiveSinusoidalSeasonGenerator

register(GeneratorBenchmark(
    generator_cls=GlobalAdditiveSinusoidalSeasonGenerator,
    name="SinusoidalSeason",
    module_path="time_series.seasons",
    default_params={"period": 10},
    scenarios=[SCENARIO_SINE_WAVE],
))
