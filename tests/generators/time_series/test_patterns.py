import numpy as np
import pandas as pd

from badgers.generators.time_series.patterns import Pattern, add_offset, \
    add_linear_trend, scale, RandomlySpacedConstantPatterns, RandomlySpacedLinearPatterns, RandomlySpacedPatterns


def test_pattern__resample():
    """Pattern.resample produces correct length and endpoints."""
    pattern = Pattern(np.array([1, 2, 3, 4, 5]))
    resampled = pattern.resample(10)
    assert len(resampled) == 10
    assert resampled[0] == 1
    assert resampled[-1] == 5


def test_pattern__add_offset():
    """add_offset shifts pattern values correctly."""
    pattern = Pattern(np.array([1, 2, 3, 4, 5]))
    expected = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
    transformed = add_offset(pattern.values, 2)
    assert transformed.tolist() == expected.tolist()


def test_pattern__add_linear_trend():
    """add_linear_trend produces correct start and end values."""
    pattern = Pattern(np.array([1, 2, 3, 4, 5]))
    transformed = add_linear_trend(pattern.values, start_value=0, end_value=-1)
    assert len(transformed) == 5
    assert transformed[0] == 0
    assert transformed[-1] == -1


def test_pattern__scale():
    """scale multiplies pattern values correctly."""
    pattern = Pattern(np.array([1, 2, 3, 4, 5]))
    expected = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
    transformed = scale(pattern.values, scaling_factor=2)
    assert transformed.tolist() == expected.tolist()


def test_randomly_spaced_patterns__1d_no_overlap():
    """RandomlySpacedPatterns injects correct number of non-overlapping patterns (1D)."""
    n_patterns = 3
    X = np.zeros(shape=100).reshape(-1, 1)
    pattern = Pattern(values=np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]))
    generator = RandomlySpacedPatterns()
    Xt, _ = generator.generate(X=X, y=None, n_patterns=n_patterns, min_width_pattern=5, max_width_patterns=10, pattern=pattern)
    assert n_patterns == len(generator.patterns_indices_)
    for i in range(n_patterns - 1):
        assert generator.patterns_indices_[i][1] < generator.patterns_indices_[i + 1][0]


def test_randomly_spaced_patterns__2d_no_overlap():
    """RandomlySpacedPatterns injects correct number of non-overlapping patterns (2D)."""
    n_patterns = 3
    X = np.zeros(shape=(100, 2))
    pattern = Pattern(
        values=np.array([[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], [0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0]]).T)
    generator = RandomlySpacedPatterns()
    Xt, _ = generator.generate(X=X, y=None, n_patterns=n_patterns, min_width_pattern=5, max_width_patterns=10, pattern=pattern)
    assert n_patterns == len(generator.patterns_indices_)
    for i in range(n_patterns - 1):
        assert generator.patterns_indices_[i][1] < generator.patterns_indices_[i + 1][0]


def test_randomly_spaced_constant_patterns__injects_zeros(time_series_sine):
    """RandomlySpacedConstantPatterns injects constant-zero regions."""
    n_patterns = 3
    X_sine, _ = time_series_sine
    X = pd.DataFrame(data=X_sine)
    generator = RandomlySpacedConstantPatterns()
    Xt, _ = generator.generate(X=X, y=None, n_patterns=n_patterns, min_width_pattern=5, max_width_patterns=10,
                               constant_value=0)
    assert n_patterns == len(generator.patterns_indices_)
    for start, end in generator.patterns_indices_:
        assert Xt.iloc[start:end, :].values.tolist() == np.zeros((end - start, X.shape[1])).tolist()


def test_randomly_spaced_linear_patterns__interpolates(time_series_sine):
    """RandomlySpacedLinearPatterns linearly interpolates between endpoints."""
    X_sine, _ = time_series_sine
    X = pd.DataFrame(data=X_sine)
    generator = RandomlySpacedLinearPatterns()
    Xt, _ = generator.generate(X=X, y=None, n_patterns=3, min_width_pattern=5, max_width_patterns=10)
    for (start, end) in generator.patterns_indices_:
        for col in range(X.shape[1]):
            assert Xt.iloc[start:end, col].tolist() == \
                np.linspace(X.iloc[start, col], X.iloc[end, col], end - start).tolist()


if __name__ == '__main__':
    unittest.main()