"""Benchmark registrations for tabular imbalance generators."""
from numpy.random import default_rng

from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import (
    SCENARIO_SMALL_BLOBS, SCENARIO_SMALL_BLOBS_CLASSES,
)
from badgers.core.selectors import ClassSelector, ProportionalSelector
from badgers.generators.tabular_data.imbalance import (
    RandomSamplingClassesGenerator,
    RandomSamplingTargetsGenerator,
)

register(GeneratorBenchmark(
    generator_cls=RandomSamplingClassesGenerator,
    name="RandomSamplingClasses",
    module_path="tabular_data.imbalance",
    default_params={
        "selector": ClassSelector([0], class_column=2,
                                  proportions={c: 0.2 for c in range(5)}, rng=default_rng(0))
    },
    scenarios=[SCENARIO_SMALL_BLOBS_CLASSES],
))

register(GeneratorBenchmark(
    generator_cls=RandomSamplingTargetsGenerator,
    name="RandomSamplingTargets",
    module_path="tabular_data.imbalance",
    default_params={
        "selector": ProportionalSelector([0, 1], column=0),
    },
    scenarios=[SCENARIO_SMALL_BLOBS],
))
