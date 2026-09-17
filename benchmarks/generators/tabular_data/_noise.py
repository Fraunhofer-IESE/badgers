"""Benchmark registrations for tabular noise generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator, GaussianNoiseClassesGenerator

register(GeneratorBenchmark(
    generator_cls=GaussianNoiseGenerator,
    name="GaussianNoise",
    module_path="tabular_data.noise",
    default_params={"noise_std": 0.5},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=GaussianNoiseClassesGenerator,
    name="GaussianNoiseClasses",
    module_path="tabular_data.noise",
    default_params={"noise_std_per_class": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
