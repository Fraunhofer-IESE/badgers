import unittest

import numpy as np

from badgers.generators.time_series.missingness import MissingAtRandomGenerator


class TestMissingAtRandomGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 4 * np.pi, 101)
        self.X = np.sin(self.t).reshape(-1, 1)

    def test_generator(self):
        n_missing = 10
        generator = MissingAtRandomGenerator()
        Xt, _ = generator.generate(X=self.X, y=None, n_missing=n_missing)
        self.assertEqual(Xt.shape, self.X.shape)
        self.assertEqual(len(generator.missing_indices_), n_missing)
        self.assertEqual(Xt.isna().sum()[0], n_missing)


if __name__ == '__main__':
    unittest.main()
