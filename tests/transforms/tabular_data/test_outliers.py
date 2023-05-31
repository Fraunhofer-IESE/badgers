from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.transforms.tabular_data.outliers import OutliersTransformer, ZScoreSampling, HistogramSampling
from tests.transforms.tabular_data import generate_test_data_only_features


class TestOutliersTransformer(TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformers_classes = OutliersTransformer.__subclasses__()
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_all_transformers(self):
        """
        run generic tests for all transformer classes:
        - checks that the number of outliers corresponds to what was given
        - checks that the transformed array has the same size as the input array
        - checks that the transformed data points have a lower outlierness score
        """
        for cls in self.transformers_classes:
            transformer = cls()
            for input_type, X in self.input_test_data.items():
                if X.shape[1] > 5 and transformer.__class__ is HistogramSampling:
                    with self.assertRaises(NotImplementedError):
                        _ = transformer.transform(X.copy())
                else:
                    with self.subTest(transformer=transformer.__class__, input_type=input_type):
                        outliers = transformer.transform(X.copy())
                        # assert number of outliers
                        self.assertEqual(outliers.shape[0], int(transformer.percentage_extreme_values * X.shape[0] / 100))
                        # assert shape
                        self.assertEqual(outliers.shape[1], X.shape[1])
                        # TODO assert outlierness score


class TestZScoreTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformer = ZScoreSampling(random_generator=self.rng, percentage_outliers=10)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def assert_zscore_larger_than_3(self, X):
        """
        Asserts that, at least in one dimension, the zscore of the generated outliers data points is greater than 3
        """
        # compute means and stds for checking z-score
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        outliers = self.transformer.transform(X.copy())
        # assert z-score > 3
        for row in range(outliers.shape[0]):
            values = outliers[row, :]
            z_scores = abs(values - means) / stds
            self.assertTrue(all(z_scores > 3.))

    def test_zscore(self):
        """
        Checks that the zscore of the generated outliers data points is greater than 3 for all different type of inputs
        """
        for input_type, X in self.input_test_data.items():
            with self.subTest(input_type=input_type):
                self.assert_zscore_larger_than_3(X)
