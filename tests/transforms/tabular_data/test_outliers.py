from unittest import TestCase
from unittest.util import safe_repr

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.ensemble import IsolationForest

from badgers.transforms.tabular_data.outliers import ZScoreTransformer, PCATransformer


class TestOutliersTransformer(TestCase):

    def assertDecreaseOutliernessScore(self, X, Xt, transformer):
        """
        Check whether the outlierness score computed by an isolation forest decreases after applying the transformer

        :param X: the original dataset
        :param Xt: the transformed dataset
        :param transformer: the transformer object used

        """
        isf = IsolationForest()
        isf.fit(X)
        original_anomaly_score = isf.score_samples(X)
        new_anomaly_score = isf.score_samples(Xt)
        for row in transformer.outliers_indices_:
            if original_anomaly_score[row] < new_anomaly_score[row]:
                msg = 'Anomaly score should be smaller after applying the transformer. ' \
                      'Anomaly score after applying the transformer %s is not smaller than before applying it %s' % (
                          safe_repr(new_anomaly_score[row]), safe_repr(original_anomaly_score[row]))
                self.fail(msg)

    def assertNumberOutliers(self, X, transformer):
        self.assertEqual(
            len(transformer.outliers_indices_),
            int(transformer.percentage_extreme_values * X.shape[0] / 100)
        )


class TestZScoreTransformer(TestOutliersTransformer):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = ZScoreTransformer(random_generator=self.rng, percentage_outliers=10)

    def perform_test(self, X):
        """

        :param X:
        :return:
        """
        # compute means and stds for checking z-score
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        Xt = self.transformer.transform(X.copy())
        # assert number of outliers
        self.assertNumberOutliers(X, self.transformer)
        # assert z-score > 3
        for row in self.transformer.outliers_indices_:
            values = Xt[row, :]
            z_scores = abs(values - means) / stds
            for z_score in z_scores:
                self.assertGreaterEqual(z_score, 3.)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        # assert outlierness score
        self.assertDecreaseOutliernessScore(X, Xt, self.transformer)

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=10).reshape(-1, 1)
        self.perform_test(X)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        self.perform_test(X)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=10).reshape(-1, 1),
            columns=['col']
        )
        self.perform_test(X)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        self.perform_test(X)


class TestPCATransformer(TestOutliersTransformer):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = PCATransformer(random_generator=self.rng, percentage_outliers=10)

    def perform_test(self, X):
        """

        :param X:
        :return:
        """
        Xt = self.transformer.transform(X.copy())
        # assert number of outliers
        self.assertNumberOutliers(X, self.transformer)
        # assert arrays have same shape
        self.assertEqual(X.shape, Xt.shape)
        # assert outlierness score
        self.assertDecreaseOutliernessScore(X, Xt, self.transformer)

    def test_transform_numpy_1D_array(self):
        X = self.rng.normal(size=10).reshape(-1, 1)
        self.perform_test(X)

    def test_transform_numpy_2D_array(self):
        X = self.rng.normal(size=(100, 10))
        self.perform_test(X)

    def test_transform_pandas_1D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=10).reshape(-1, 1),
            columns=['col']
        )
        self.perform_test(X)

    def test_transform_pandas_2D_array(self):
        X = pd.DataFrame(
            data=self.rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
        self.perform_test(X)
