"""Benchmark registrations for time series noise generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE
from badgers.generators.time_series.noise import LocalGaussianNoiseGenerator

register(GeneratorBenchmark(
    generator_cls=LocalGaussianNoiseGenerator,
    name="LocalGaussianNoise",
    module_path="time_series.noise",
    default_params={"noise_std": 0.1},
    scenarios=[SCENARIO_SINE_WAVE],
))
