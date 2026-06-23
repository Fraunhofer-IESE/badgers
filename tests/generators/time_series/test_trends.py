import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas._testing import assert_frame_equal

from badgers.generators.time_series.trends import GlobalAdditiveLinearTrendGenerator, AdditiveLinearTrendGenerator, \
    RandomlySpacedLinearTrends


def test_global_additive_linear_trend__matches_expected():
    """GlobalAdditiveLinearTrendGenerator produces expected linear trend output."""
    rng = default_rng(seed=0)
    generator = GlobalAdditiveLinearTrendGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(shape=(10, 4)), columns=[f'col{i}' for i in range(4)])
    slope = np.array([1, 2, 3, 4])

    expected_Xt = pd.DataFrame(
        data=np.array([np.linspace(0, len(X) * s, len(X)) for s in slope]).T,
        columns=X.columns, index=X.index,
    )

    Xt, _ = generator.generate(X, None, slope=slope)
    assert_frame_equal(Xt, expected_Xt)


def test_additive_linear_trend__matches_expected():
    """AdditiveLinearTrendGenerator produces expected trend in specified range."""
    rng = default_rng(seed=0)
    generator = AdditiveLinearTrendGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(shape=(10, 4)), columns=[f'col{i}' for i in range(4)])
    slope = np.array([0, 0.5, 1, 2])

    expected_Xt = pd.DataFrame(
        data=np.array([
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
        ]),
        columns=X.columns, index=X.index,
    )

    Xt, _ = generator.generate(X, None, slope=slope, start=3, end=7)
    assert_frame_equal(Xt, expected_Xt)


def test_randomly_spaced_linear_trends__outside_intervals_constant():
    """RandomlySpacedLinearTrends: outside pattern intervals, values are constant."""
    rng = default_rng(seed=0)
    generator = RandomlySpacedLinearTrends(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(shape=(100, 4)), columns=[f'col{i}' for i in range(4)])

    Xt, _ = generator.generate(X, None, n_patterns=5, min_width_pattern=5, max_width_patterns=10)

    for i in range(1, len(generator.patterns_indices_)):
        s = generator.patterns_indices_[i - 1][1]
        e = generator.patterns_indices_[i][0]
        assert Xt[s:e].diff().dropna().sum().sum() == 0
