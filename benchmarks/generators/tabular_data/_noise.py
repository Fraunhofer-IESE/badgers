"""Benchmark registrations for tabular noise generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from benchmarks.checks.common import CHECK_SAME_SHAPE
from benchmarks.checks.tabular import CHECK_INCREASED_VARIANCE
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator

register(GeneratorBenchmark(
    generator_cls=GaussianNoiseGenerator,
    name="GaussianNoise",
    module_path="tabular_data.noise",
    default_params={"noise_std": 0.5},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
    functional_checks=[CHECK_SAME_SHAPE, CHECK_INCREASED_VARIANCE],
))
