"""Predefined scenarios for text generators."""
from benchmarks.models import Scenario


_WORDS = [
    "algorithm", "benchmark", "computation", "database", "experiment",
    "framework", "generator", "hypothesis", "implementation", "kernel",
    "library", "machine", "network", "optimization", "pipeline",
    "quantum", "regression", "statistics", "transformer", "validation",
]


def _word_list_factory(rng):
    return list(_WORDS), None


SCENARIO_WORD_LIST = Scenario(
    name="word_list_20",
    data_type="text",
    factory=_word_list_factory,
    tags=["small"],
)