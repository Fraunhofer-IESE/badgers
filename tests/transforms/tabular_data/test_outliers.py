from unittest import TestCase
from unittest.util import safe_repr

import numpy as np
from numpy.random import default_rng
from sklearn.ensemble import IsolationForest

from badgers.transforms.tabular_data.outliers import OutliersTransformer, ZScoreTransformer
from tests.transforms.tabular_data import generate_test_data


class TestOutliersTransformer(TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformers_classes = OutliersTransformer.__subclasses__()
        self.input_test_data = generate_test_data(rng=self.rng)

    def test_all_transformers(self):
        """
        run generic tests for all transformer classes:
        - checks that the number of outliers corresponds to what was given
        - checks that the transformed array has the same shape as the input array
        - checks that the transformed data points have a lower outlierness score
        """
        for cls in self.transformers_classes:
            transformer = cls()
            for input_type, X in self.input_test_data.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt = transformer.transform(X.copy())
                    # assert number of outliers
                    self.assertNumberOutliers(X, transformer)
                    # assert arrays have same shape
                    self.assertEqual(X.shape, Xt.shape)
                    # assert outlierness score
                    self.assertDecreaseOutliernessScore(X, Xt, transformer)

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


class TestZScoreTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = ZScoreTransformer(random_generator=self.rng, percentage_outliers=10)
        self.input_test_data = generate_test_data(rng=self.rng)

    def assert_zscore_larger_than_3(self, X):
        """
        Asserts that the zscore of the generated outliers data points is greater than 3
        """
        # compute means and stds for checking z-score
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        Xt = self.transformer.transform(X.copy())
        # assert z-score > 3
        for row in self.transformer.outliers_indices_:
            values = Xt[row, :]
            z_scores = abs(values - means) / stds
            for z_score in z_scores:
                self.assertGreaterEqual(z_score, 3.)

    def test_zscore(self):
        """
        Checks that the zscore of the generated outliers data points is greater than 3 for all different type of inputs
        """
        for input_type, X in self.input_test_data.items():
            with self.subTest(input_type=input_type):
                self.assert_zscore_larger_than_3(X)
