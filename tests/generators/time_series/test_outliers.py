import unittest

import numpy as np

from badgers.generators.time_series.outliers import LocalZScoreGenerator, RandomZerosGenerator

class TestRandomZerosGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 4 * np.pi, 101)
        self.X = np.sin(self.t).reshape(-1, 1)

    def test_generator(self):
        n_outliers = 10
        generator = RandomZerosGenerator()
        Xt, _ = generator.generate(X=self.X, y=None, n_outliers=n_outliers)
        self.assertEqual(Xt.shape, self.X.shape)
        self.assertEqual(len(generator.outliers_indices_), n_outliers)

class TestLocalZScoreGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 4 * np.pi, 101)
        self.X = np.sin(self.t).reshape(-1, 1)

    def test_generator(self):
        n_outliers = 10
        generator = LocalZScoreGenerator()
        Xt, _ = generator.generate(X=self.X, y=None, n_outliers=n_outliers)
        self.assertEqual(Xt.shape, self.X.shape)
        self.assertEqual(len(generator.outliers_indices_), n_outliers)
        self.assertFalse(Xt.isna().any()[0])



if __name__ == '__main__':
    unittest.main()
