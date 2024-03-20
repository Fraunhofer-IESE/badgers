import unittest
from numpy.random import default_rng

from badgers.generators.time_series.changepoints import ChangePointGenerator, RandomChangeInMeanGenerator


class TestChangePointGenerator(unittest.TestCase):
    def setUp(self):
        self.random_generator = default_rng(seed=0)
        self.n_changepoints = 10
        self.min_change = -5
        self.max_change = 5

    def test_ChangePointGenerator_init(self):
        generator = ChangePointGenerator(random_generator=self.random_generator, n_changepoints=self.n_changepoints)
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
        cp_generator = RandomChangeInMeanGenerator(random_generator=self.random_generator, n_changepoints=self.n_changepoints, min_change=self.min_change, max_change=self.max_change)
        X = [1, 2, 3, 4, 5]
        y = None

        Xt, _ = cp_generator.generate(X, y)
        self.assertNotEqual(Xt, X)
        self.assertEqual(y, y)


if __name__ == '__main__':
    unittest.main()