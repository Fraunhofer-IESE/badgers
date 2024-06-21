import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas._testing import assert_frame_equal

from badgers.generators.time_series.trends import GlobalAdditiveLinearTrendGenerator


class TestTrendsGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)

    def test_global_additive_linear_trend_generator(self):
        # Test the generate method of GlobalAdditiveLinearTrendGenerator
        global_additive_linear_trend_generator = GlobalAdditiveLinearTrendGenerator(
            random_generator=self.random_generator)
        X = pd.DataFrame(data=np.zeros(shape=(10, 4)), columns=[f'col{i}' for i in range(4)])
        y = None

        slope = np.array([1, 2, 3, 4])
        t = np.linspace(0, 1, len(X))
        expected_Xt = pd.DataFrame(data=t[:, np.newaxis] * slope, columns=X.columns, index=X.index)

        Xt, _ = global_additive_linear_trend_generator.generate(X, y, slope=slope)

        assert_frame_equal(Xt, expected_Xt)


if __name__ == '__main__':
    unittest.main()
