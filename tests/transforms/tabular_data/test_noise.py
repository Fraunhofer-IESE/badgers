import unittest

import pandas as pd
from numpy.random import default_rng

from badgers.transforms.tabular_data.noise import GaussianNoiseTransformer


class TestGaussianWhiteNoiseTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.noise_transformer = GaussianNoiseTransformer()

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=(10)).reshape(-1, 1)
        Xt = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        Xt = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        Xt = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        self.assertTrue(isinstance(Xt, pd.DataFrame))

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        Xt = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        self.assertTrue(isinstance(Xt, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()
