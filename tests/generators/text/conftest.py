import pytest


_WORDS = [
    "algorithm", "benchmark", "computation", "database", "experiment",
    "framework", "generator", "hypothesis", "implementation", "kernel",
    "library", "machine", "network", "optimization", "pipeline",
    "quantum", "regression", "statistics", "transformer", "validation",
]


@pytest.fixture
def text_word_list():
    """List of 20 technical words as (words, None)."""
    return list(_WORDS), None
