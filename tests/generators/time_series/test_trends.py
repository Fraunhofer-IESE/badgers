import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas._testing import assert_frame_equal

from badgers.generators.time_series.trends import GlobalAdditiveLinearTrendGenerator, AdditiveLinearTrendGenerator, \
    RandomlySpacedLinearTrends


class TestGlobalAdditiveLinearTrendsGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)

    def test_global_additive_linear_trend_generator(self):
        # Test the generate method of GlobalAdditiveLinearTrendGenerator
        global_additive_linear_trend_generator = GlobalAdditiveLinearTrendGenerator(
            random_generator=self.random_generator)
        X = pd.DataFrame(data=np.zeros(shape=(10, 4)), columns=[f'col{i}' for i in range(4)])
        y = None

        slope = np.array([1, 2, 3, 4])

        expected_Xt = pd.DataFrame(
            data=np.array([np.linspace(0, len(X) * s, len(X)) for s in slope]).T,
            columns=X.columns, index=X.index
        )

        Xt, _ = global_additive_linear_trend_generator.generate(X, y, slope=slope)

        assert_frame_equal(Xt, expected_Xt)


class TestAdditiveLinearTrendsGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)

    def test_global_additive_linear_trend_generator(self):
        # Test the generate method of AdditiveLinearTrendGenerator
        additive_linear_trend_generator = AdditiveLinearTrendGenerator(
            random_generator=self.random_generator)
        X = pd.DataFrame(data=np.zeros(shape=(10, 4)), columns=[f'col{i}' for i in range(4)])
        y = None
        start = 3
        end = 7

        slope = np.array([0, 0.5, 1, 2])

        expected_Xt = pd.DataFrame(
            data=np.array(
                [
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 2. / 3., 4. / 3., 8. / 3.],
                    [0., 4. / 3., 8. / 3., 16. / 3.],
                    [0., 2., 4., 8.],
                    [0., 2., 4., 8.],
                    [0., 2., 4., 8.],
                    [0., 2., 4., 8.]
                ]
            ),
            columns=X.columns, index=X.index
        )

        Xt, _ = additive_linear_trend_generator.generate(X, y, slope=slope, start=start, end=end)

        assert_frame_equal(Xt, expected_Xt)


class TestRandomlySpacedLinearTrends(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)

    def test_global_additive_linear_trend_generator(self):
        # Test the generate method of AdditiveLinearTrendGenerator
        randomly_spaced_trend_generator = RandomlySpacedLinearTrends(
            random_generator=self.random_generator)
        X = pd.DataFrame(data=np.zeros(shape=(100, 4)), columns=[f'col{i}' for i in range(4)])
        y = None

        Xt, _ = randomly_spaced_trend_generator.generate(X, y, n_patterns=5, min_width_pattern=5, max_width_patterns=10)

        # assert outside time intervals, constant values
        for i in range(1, len(randomly_spaced_trend_generator.patterns_indices_)):
            s = randomly_spaced_trend_generator.patterns_indices_[i - 1][1]
            e = randomly_spaced_trend_generator.patterns_indices_[i][0]
            self.assertEqual(Xt[s:e].diff().dropna().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
