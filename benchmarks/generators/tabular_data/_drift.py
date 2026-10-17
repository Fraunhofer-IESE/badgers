"""Benchmark registrations for tabular drift generators."""
from numpy.random import default_rng

from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import (
    SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS,
    SCENARIO_SMALL_BLOBS_CLASSES, SCENARIO_MEDIUM_BLOBS_CLASSES,
)
from badgers.core.selectors import ClassSelector
from badgers.generators.tabular_data.drift import RandomShiftGenerator, RandomShiftClassesGenerator

_CLASS_PROPORTIONS = {"cls": {c: 1.0 for c in range(5)}}

register(GeneratorBenchmark(
    generator_cls=RandomShiftGenerator,
    name="RandomShift",
    module_path="tabular_data.drift",
    default_params={"shift_std": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=RandomShiftClassesGenerator,
    name="RandomShiftClasses",
    module_path="tabular_data.drift",
    default_params={
        "selector": ClassSelector("cls", {"cls": 2}, _CLASS_PROPORTIONS, rng=default_rng(0)),
        "shift_std": 0.1,
    },
    scenarios=[SCENARIO_SMALL_BLOBS_CLASSES],
))

register(GeneratorBenchmark(
    generator_cls=RandomShiftClassesGenerator,
    name="RandomShiftClassesMedium",
    module_path="tabular_data.drift",
    default_params={
        "selector": ClassSelector("cls", {f"f{i}": i for i in range(5)} | {"cls": 5}, _CLASS_PROPORTIONS, rng=default_rng(0)),
        "shift_std": 0.1,
    },
    scenarios=[SCENARIO_MEDIUM_BLOBS_CLASSES],
))
