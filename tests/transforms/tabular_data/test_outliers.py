from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.transforms.tabular_data.outliers import ZScoreTransformer


class TestZScoreTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = ZScoreTransformer(random_generator=self.rng, percentage_outliers=10)

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=(10)).reshape(-1, 1)
        means = np.mean(X, axis=0)
        vars = np.var(X, axis=0)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.extreme_values_mapping_), 1)
        for (row, col) in self.transformer.extreme_values_mapping_:
            val = X_transformed[row, col]
            self.assertGreaterEqual(abs(val - means[col]), 3. * vars[col])
            self.assertLessEqual(abs(val - means[col]), 5. * vars[col])

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        means = np.mean(X, axis=0)
        vars = np.var(X, axis=0)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.extreme_values_mapping_), 100)
        for (row, col) in self.transformer.extreme_values_mapping_:
            val = X_transformed[row, col]
            self.assertGreaterEqual(abs(val - means[col]), 3. * vars[col])
            self.assertLessEqual(abs(val - means[col]), 5. * vars[col])

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        means = np.mean(X, axis=0)
        vars = np.var(X, axis=0)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.extreme_values_mapping_), 1)
        for (row, col) in self.transformer.extreme_values_mapping_:
            val = X_transformed[row, col]
            self.assertGreaterEqual(abs(val - means[col]), 3. * vars[col])
            self.assertLessEqual(abs(val - means[col]), 5. * vars[col])

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        means = np.mean(X, axis=0)
        vars = np.var(X, axis=0)
        X_transformed = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, X_transformed.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.extreme_values_mapping_), 100)
        for (row, col) in self.transformer.extreme_values_mapping_:
            val = X_transformed[row, col]
            self.assertGreaterEqual(abs(val - means[col]), 3. * vars[col])
            self.assertLessEqual(abs(val - means[col]), 5. * vars[col])
