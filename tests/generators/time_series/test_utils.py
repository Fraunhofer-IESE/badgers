import numpy as np
import pytest

from badgers.generators.time_series.utils import generate_random_patterns_indices


def test_generate_random_patterns_indices__no_patterns_raises():
    """generate_random_patterns_indices raises AssertionError for n_patterns=0."""
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        generate_random_patterns_indices(
            random_generator=rng, signal_size=100, n_patterns=0,
            min_width_pattern=5, max_width_patterns=10,
        )


def test_generate_random_patterns_indices__single_pattern():
    """generate_random_patterns_indices: single pattern has correct bounds."""
    rng = np.random.default_rng(0)
    signal_size = 100
    patterns = generate_random_patterns_indices(
        random_generator=rng, signal_size=signal_size, n_patterns=1,
        min_width_pattern=5, max_width_patterns=10,
    )
    assert len(patterns) == 1
    for start, end in patterns:
        assert start < end
        assert end - start >= 5
        assert end - start < 10
        assert 0 <= start < signal_size
        assert 0 <= end < signal_size


def test_generate_random_patterns_indices__multiple_patterns():
    """generate_random_patterns_indices: 5 patterns all have correct bounds."""
    rng = np.random.default_rng(0)
    signal_size = 100
    patterns = generate_random_patterns_indices(
        random_generator=rng, signal_size=signal_size, n_patterns=5,
        min_width_pattern=5, max_width_patterns=10,
    )
    assert len(patterns) == 5
    for start, end in patterns:
        assert start < end
        assert end - start >= 5
        assert end - start < 10
        assert 0 <= start < signal_size
        assert 0 <= end < signal_size
