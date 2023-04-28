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
        X_transformed = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        X_transformed = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        X_transformed = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        X_transformed = self.noise_transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)


if __name__ == '__main__':
    unittest.main()
