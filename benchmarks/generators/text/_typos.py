"""Benchmark registrations for text typo generators."""
from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.text import SCENARIO_WORD_LIST
from badgers.generators.text.typos import SwapLettersGenerator

register(GeneratorBenchmark(
    generator_cls=SwapLettersGenerator,
    name="SwapLetters",
    module_path="text.typos",
    default_params={"swap_proba": 0.5},
    scenarios=[SCENARIO_WORD_LIST],
    functional_checks=[],
))
