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
        stds = np.std(X, axis=0)
        Xt = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.outliers_indices_), 1)
        for (row, col) in self.transformer.outliers_indices_:
            value = Xt[row, col]
            z_score = abs(value - means[col]) / stds[col]
            self.assertGreaterEqual(z_score, 3.)
            self.assertLessEqual(z_score, 5.)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        Xt = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.outliers_indices_), 100)
        for (row, col) in self.transformer.outliers_indices_:
            value = Xt[row, col]
            z_score = abs(value - means[col]) / stds[col]
            self.assertGreaterEqual(z_score, 3.)
            self.assertLessEqual(z_score, 5.)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(10)).reshape(-1, 1),
            columns=['col']
        )
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        Xt = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.outliers_indices_), 1)
        for (row, col) in self.transformer.outliers_indices_:
            value = Xt[row, col]
            z_score = abs(value - means[col]) / stds[col]
            self.assertGreaterEqual(z_score, 3.)
            self.assertLessEqual(z_score, 5.)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        means = np.mean(X, axis=0)
        vars = np.std(X, axis=0)
        Xt = self.transformer.transform(X)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        # assert number of extreme values
        self.assertEqual(len(self.transformer.outliers_indices_), 100)
        for (row, col) in self.transformer.outliers_indices_:
            value = Xt[row, col]
            z_score = abs(value - means[col]) / vars[col]
            self.assertGreaterEqual(z_score, 3.)
            self.assertLessEqual(z_score, 5.)
