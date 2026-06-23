"""Benchmark registrations for graph missingness generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.graph import SCENARIO_ERDOS_RENYI
from badgers.generators.graph.missingness import NodesMissingCompletelyAtRandom

register(GeneratorBenchmark(
    generator_cls=NodesMissingCompletelyAtRandom,
    name="NodesMCAR",
    module_path="graph.missingness",
    default_params={"percentage_missing": 0.1},
    scenarios=[SCENARIO_ERDOS_RENYI],
    functional_checks=[],
))
