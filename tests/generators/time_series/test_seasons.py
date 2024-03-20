import unittest

import numpy as np
from numpy.random import default_rng

from badgers.generators.time_series.seasons import GlobalAdditiveSinusoidalSeasonGenerator


class TestSeasonsGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)
        self.global_additive_season_generator = GlobalAdditiveSinusoidalSeasonGenerator(
            random_generator=self.random_generator)

    def test_global_additive_sinusoidal_season_generator(self):
        # Test the generate method of GlobalAdditiveSinusoidalSeasonGenerator
        X = np.zeros(shape=100)
        y = None
        expected_period = 100
        generator = GlobalAdditiveSinusoidalSeasonGenerator(random_generator=self.random_generator,
                                                            period=expected_period)
        Xt, yt = generator.generate(X, y)

        t = np.arange(100)
        season = np.sin(t * 2 * np.pi / expected_period)
        Xt_expected = X + season

        self.assertListEqual(Xt_expected.tolist(), Xt.tolist())
        self.assertEqual(y, yt)


if __name__ == '__main__':
    unittest.main()
