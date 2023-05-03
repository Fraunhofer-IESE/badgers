import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.transforms.tabular_data.imbalance import RandomSamplingClassesTransformer


class TestRandomSamplingClassesTransformer(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = RandomSamplingClassesTransformer(random_generator=self.rng)

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=(10)).reshape(-1, 1)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        Xt = self.transformer.fit_transform(X, y)
        # assert arrays have same shape
        self.assertEqual(Xt.shape[1], X.shape[1])
        self.assertEqual(Xt.shape[0], self.transformer.labels_.shape[0])

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        y = np.array([1, 2, 3, 4, 5] * 20)
        Xt = self.transformer.fit_transform(X, y)
        # assert arrays have same shape
        self.assertEqual(Xt.shape[1], X.shape[1])
        self.assertEqual(Xt.shape[0], self.transformer.labels_.shape[0])

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        Xt = self.transformer.fit_transform(X, y)
        # assert arrays have same shape
        self.assertEqual(Xt.shape[1], X.shape[1])
        self.assertEqual(Xt.shape[0], self.transformer.labels_.shape[0])

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        y = np.array([1, 2, 3, 4, 5] * 20)
        Xt = self.transformer.fit_transform(X, y)
        # assert arrays have same shape
        self.assertEqual(Xt.shape[1], X.shape[1])
        self.assertEqual(Xt.shape[0], self.transformer.labels_.shape[0])


if __name__ == '__main__':
    unittest.main()
