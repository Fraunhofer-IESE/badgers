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
        period = 10
        generator = GlobalAdditiveSinusoidalSeasonGenerator(random_generator=self.random_generator)
        Xt, yt = generator.generate(X=X, y=y, period=period)

        t = np.arange(100)
        season = np.sin(t * 2 * np.pi / period)
        Xt_expected = X + season

        self.assertListEqual(Xt_expected.reshape(-1,1).tolist(), Xt.values.tolist())
        self.assertEqual(y, yt)


if __name__ == '__main__':
    unittest.main()
