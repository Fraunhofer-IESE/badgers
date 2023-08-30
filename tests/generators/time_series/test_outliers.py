import unittest

import numpy as np

from badgers.generators.time_series.outliers import LocalZScoreGenerator, RandomZerosGenerator

class TestRandomZerosGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 4 * np.pi, 101)
        self.X = np.sin(self.t).reshape(-1, 1)

    def test_generator(self):
        n_outliers = 10
        generator = RandomZerosGenerator(n_outliers=n_outliers)
        Xt, _ = generator.generate(self.X, None)
        self.assertEqual(Xt.shape, self.X.shape)
        self.assertEqual(len(generator.outliers_indices_), n_outliers)

class TestLocalZScoreGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 4 * np.pi, 101)
        self.X = np.sin(self.t).reshape(-1, 1)

    def test_generator(self):
        n_outliers = 10
        generator = LocalZScoreGenerator(n_outliers=n_outliers)
        Xt, _ = generator.generate(self.X, None)
        self.assertEqual(Xt.shape, self.X.shape)
        self.assertEqual(len(generator.outliers_indices_), n_outliers)



if __name__ == '__main__':
    unittest.main()
