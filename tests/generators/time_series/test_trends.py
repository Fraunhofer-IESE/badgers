import numpy as np
from numpy.random import default_rng

from badgers.generators.time_series.trends import GlobalAdditiveLinearTrendGenerator, AdditiveLinearTrendGenerator, \
    RandomlySpacedLinearTrends


def test_global_additive_linear_trend__matches_expected():
    """GlobalAdditiveLinearTrendGenerator produces expected linear trend output."""
    rng = default_rng(seed=0)
    generator = GlobalAdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(10, 4))
    slope = np.array([1, 2, 3, 4])

    expected_Xt = np.array([np.linspace(0, len(X) * s, len(X)) for s in slope]).T

    Xt = generator.generate(X, slope=slope)
    np.testing.assert_array_equal(Xt, expected_Xt)


def test_global_additive_linear_trend__scalar_slope_applies_to_every_column():
    """Scalar slope: the same per-row ramp is applied to every column of X.

    Regression: a scalar slope previously crashed on non-square X
    (ValueError: could not broadcast) and silently applied the ramp along
    COLUMNS on square X (n == n_cols).
    """
    rng = default_rng(seed=0)
    generator = GlobalAdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(10, 3))

    Xt = generator.generate(X, slope=2.0)

    expected = np.tile(np.linspace(0, 2.0 * 10, 10).reshape(-1, 1), 3)
    np.testing.assert_array_equal(Xt, expected)


def test_global_additive_linear_trend__scalar_slope_square_x_is_rowwise():
    """Scalar slope on square X (n == n_cols): the ramp runs along ROWS.

    Regression: the old code broadcast the 1-D ramp along columns, so each
    row j received the constant value ramp[j] — a transposed trend.
    """
    rng = default_rng(seed=0)
    generator = GlobalAdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(6, 6))

    Xt = generator.generate(X, slope=1.0)

    expected = np.tile(np.linspace(0, 6.0, 6).reshape(-1, 1), 6)
    np.testing.assert_array_equal(Xt, expected)


def test_additive_linear_trend__matches_expected():
    """AdditiveLinearTrendGenerator produces expected trend in specified range."""
    rng = default_rng(seed=0)
    generator = AdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(10, 4))
    slope = np.array([0, 0.5, 1, 2])

    expected_Xt = np.array([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 2. / 3., 4. / 3., 8. / 3.],
        [0., 4. / 3., 8. / 3., 16. / 3.],
        [0., 2., 4., 8.],
        [0., 2., 4., 8.],
        [0., 2., 4., 8.],
        [0., 2., 4., 8.],
    ])

    Xt = generator.generate(X, slope=slope, start=3, end=7)
    np.testing.assert_array_equal(Xt, expected_Xt)


def test_additive_linear_trend__scalar_slope_applies_to_every_column():
    """Scalar slope: the same per-row ramp is applied to every column of X.

    Regression: a scalar slope previously crashed on non-square X
    (ValueError: could not broadcast ... (n,) into (n,2)) and silently
    applied the ramp along COLUMNS on square X (n == n_cols).
    """
    rng = default_rng(seed=0)
    generator = AdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(10, 3))

    Xt = generator.generate(X, slope=2.0, start=3, end=7)

    ramp = np.linspace(0, 2.0 * (7 - 3), 7 - 3).reshape(-1, 1)
    expected = np.zeros(shape=(10, 3))
    expected[3:7, :] = ramp
    expected[7:, :] = ramp[-1, :]
    np.testing.assert_array_equal(Xt, expected)


def test_additive_linear_trend__scalar_slope_square_x_is_rowwise():
    """Scalar slope on square X (n == n_cols): the ramp runs along ROWS.

    Regression: the old code broadcast the 1-D ramp along columns inside the
    (n == n_cols) slice assignment, transposing the trend instead of crashing.
    """
    rng = default_rng(seed=0)
    generator = AdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(8, 8))

    Xt = generator.generate(X, slope=1.0, start=0, end=8)

    expected = np.tile(np.linspace(0, 8.0, 8).reshape(-1, 1), 8)
    np.testing.assert_array_equal(Xt, expected)


def test_additive_linear_trend__default_start_end():
    """start=None/end=None draw an int start and extend the trend to the signal end."""
    rng = default_rng(seed=0)
    generator = AdditiveLinearTrendGenerator(random_generator=rng)
    X = np.zeros(shape=(20, 2))
    slope = np.array([0.5, 0.25])

    Xt = generator.generate(X, slope=slope, start=None, end=None)

    offset = Xt - X
    assert Xt.shape == X.shape
    # the drawn start carries the ramp's leading 0, so the first NONZERO row is
    # one past the drawn start; the drawn start is unknowable from the output.
    first_nonzero = int(np.flatnonzero(offset[:, 0])[0])
    # start is drawn (as an int) in the first half of the signal
    assert 0 <= first_nonzero <= 10
    # the ramp from the drawn start on is LINEAR with constant positive step and
    # reaches its final value in the last row of the signal
    ramp = offset[first_nonzero - 1:]  # include the leading 0
    steps = np.diff(ramp, axis=0)
    np.testing.assert_allclose(steps, np.broadcast_to(steps[0], steps.shape))
    assert (steps > 0).all()
    np.testing.assert_allclose(ramp[-1], slope * (len(X) - first_nonzero + 1))


def test_randomly_spaced_linear_trends__outside_intervals_constant():
    """RandomlySpacedLinearTrends: outside pattern intervals, values are constant."""
    rng = default_rng(seed=0)
    generator = RandomlySpacedLinearTrends(random_generator=rng)
    X = np.zeros(shape=(100, 4))

    Xt = generator.generate(X, n_patterns=5, min_width_pattern=5, max_width_patterns=10)

    for i in range(1, len(generator.patterns_indices_)):
        s = generator.patterns_indices_[i - 1][1]
        e = generator.patterns_indices_[i][0]
        assert np.diff(Xt[s:e], axis=0).sum() == 0
