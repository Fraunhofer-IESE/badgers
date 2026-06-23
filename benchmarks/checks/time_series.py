"""Functional checks for time series generators."""
from benchmarks.models import FunctionalCheck


def _check_pattern_count(X, y, Xt, yt, params, generator=None):
    if generator is None:
        return True
    n_patterns = params.get("n_patterns", 0)
    if hasattr(generator, "patterns_indices_"):
        return len(generator.patterns_indices_) == n_patterns
    return True


CHECK_PATTERN_COUNT = FunctionalCheck(
    name="pattern_count",
    description="Correct number of patterns were injected",
    check=_check_pattern_count,
)