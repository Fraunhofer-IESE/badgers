import unittest

import numpy as np
from numpy.random import default_rng

from badgers.generators.time_series.trends import GlobalAdditiveLinearTrendGenerator


class TestTrendsGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)

    def test_global_additive_linear_trend_generator(self):
        # Test the generate method of GlobalAdditiveLinearTrendGenerator
        global_additive_linear_trend_generator = GlobalAdditiveLinearTrendGenerator(
            random_generator=self.random_generator, slope=1)
        X = np.zeros(100)
        y = None
        expected_Xt = np.linspace(0, 1, len(X))

        Xt, _ = global_additive_linear_trend_generator.generate(X, y)

        self.assertListEqual(list(Xt), list(expected_Xt))


if __name__ == '__main__':
    unittest.main()
