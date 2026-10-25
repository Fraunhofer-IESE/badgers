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
