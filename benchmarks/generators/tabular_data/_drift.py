"""Benchmark registrations for tabular drift generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from badgers.generators.tabular_data.drift import RandomShiftGenerator

register(GeneratorBenchmark(
    generator_cls=RandomShiftGenerator,
    name="RandomShift",
    module_path="tabular_data.drift",
    default_params={"shift_std": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
