"""Benchmark registrations for tabular missingness generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS
from benchmarks.checks.common import CHECK_SAME_SHAPE
from badgers.generators.tabular_data.missingness import MissingCompletelyAtRandom

register(GeneratorBenchmark(
    generator_cls=MissingCompletelyAtRandom,
    name="MCAR",
    module_path="tabular_data.missingness",
    default_params={"percentage_missing": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
    functional_checks=[CHECK_SAME_SHAPE],
))
