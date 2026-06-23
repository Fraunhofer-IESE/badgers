"""Benchmark registrations for tabular imbalance generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS
from benchmarks.checks.common import CHECK_SAME_SHAPE
from badgers.generators.tabular_data.imbalance import RandomSamplingFeaturesGenerator

register(GeneratorBenchmark(
    generator_cls=RandomSamplingFeaturesGenerator,
    name="RandomSamplingFeatures",
    module_path="tabular_data.imbalance",
    default_params={},
    scenarios=[SCENARIO_SMALL_BLOBS],
    functional_checks=[CHECK_SAME_SHAPE],
))
