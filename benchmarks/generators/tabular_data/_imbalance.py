"""Benchmark registrations for tabular imbalance generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS
from badgers.generators.tabular_data.imbalance import (
    RandomSamplingFeaturesGenerator,
    RandomSamplingClassesGenerator,
    RandomSamplingTargetsGenerator,
)

register(GeneratorBenchmark(
    generator_cls=RandomSamplingFeaturesGenerator,
    name="RandomSamplingFeatures",
    module_path="tabular_data.imbalance",
    default_params={},
    scenarios=[SCENARIO_SMALL_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=RandomSamplingClassesGenerator,
    name="RandomSamplingClasses",
    module_path="tabular_data.imbalance",
    default_params={"proportion_classes": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}},
    scenarios=[SCENARIO_SMALL_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=RandomSamplingTargetsGenerator,
    name="RandomSamplingTargets",
    module_path="tabular_data.imbalance",
    default_params={},
    scenarios=[SCENARIO_SMALL_BLOBS],
))
