import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.transforms.tabular_data.missingness import DummyMissingAtRandom, DummyMissingNotAtRandom, \
    MissingCompletelyAtRandom


class TestDummyMissingAtRandom(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = DummyMissingAtRandom(percentage_missing=10, random_generator=default_rng(seed=0))

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=(10)).reshape(-1, 1)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 1)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 100)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 1)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 100)


class TestDummyMissingNotAtRandom(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = DummyMissingNotAtRandom(percentage_missing=10, random_generator=default_rng(seed=0))

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=(10)).reshape(-1, 1)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 1)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 100)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 1)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 100)


class TestMissingCompletelyAtRandom(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = MissingCompletelyAtRandom(percentage_missing=10,
                                                     random_generator=default_rng(seed=0))

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=(10)).reshape(-1, 1)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 1)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 100)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 1)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of nans
        self.assertEqual(np.isnan(X_transformed).sum(), 100)


if __name__ == '__main__':
    unittest.main()
