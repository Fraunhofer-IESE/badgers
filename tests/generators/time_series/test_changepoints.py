import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.generators.time_series.changepoints import RandomChangeInMeanGenerator


class TestChangePointGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)
        self.n_changepoints = 10
        self.min_change = -5
        self.max_change = 5

    def test_RandomChangeInMeanGenerator_generate(self):
        generator = RandomChangeInMeanGenerator(random_generator=self.random_generator)
        X = pd.DataFrame(data=np.zeros(100), columns=['dimension_0'], dtype=float)
        y = None

        Xt, _ = generator.generate(X.copy(), y, n_changepoints=self.n_changepoints, min_change=self.min_change,
                                   max_change=self.max_change)
        self.assertTrue(any(X != Xt))
        self.assertEqual(y, y)


if __name__ == '__main__':
    unittest.main()
