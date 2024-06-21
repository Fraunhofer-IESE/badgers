import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.generators.time_series.changepoints import ChangePointsGenerator, RandomChangeInMeanGenerator
from pandas.testing import assert_frame_equal

class TestChangePointGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)
        self.n_changepoints = 10
        self.min_change = -5
        self.max_change = 5

    def test_ChangePointGenerator_init(self):
        generator = ChangePointsGenerator(random_generator=self.random_generator, n_changepoints=self.n_changepoints)
        self.assertEqual(generator.random_generator, self.random_generator)
        self.assertEqual(generator.n_changepoints, self.n_changepoints)
        self.assertEqual(generator.changepoints, None)

    def test_RandomChangeInMeanGenerator_init(self):
        generator = RandomChangeInMeanGenerator(random_generator=self.random_generator, n_changepoints=self.n_changepoints, min_change=self.min_change, max_change=self.max_change)
        self.assertEqual(generator.random_generator, self.random_generator)
        self.assertEqual(generator.n_changepoints, self.n_changepoints)
        self.assertEqual(generator.min_change, self.min_change)
        self.assertEqual(generator.max_change, self.max_change)
        self.assertEqual(generator.changepoints, None)

    def test_RandomChangeInMeanGenerator_generate(self):
        generator = RandomChangeInMeanGenerator(random_generator=self.random_generator, n_changepoints=self.n_changepoints, min_change=self.min_change, max_change=self.max_change)
        X = pd.DataFrame(data=np.zeros(100), columns=['dimension_0'], dtype=float)
        y = None

        Xt, _ = generator.generate(X.copy(), y)
        self.assertTrue(any(X != Xt))
        self.assertEqual(y, y)


if __name__ == '__main__':
    unittest.main()