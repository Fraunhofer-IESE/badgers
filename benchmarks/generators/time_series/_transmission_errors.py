"""Benchmark registrations for time series transmission error generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from badgers.generators.time_series.transmission_errors import RandomTimeSwitchGenerator

register(GeneratorBenchmark(
    generator_cls=RandomTimeSwitchGenerator,
    name="RandomTimeSwitch",
    module_path="time_series.transmission_errors",
    default_params={"n_switches": 5},
    scenarios=[SCENARIO_SINE_WAVE],
))
