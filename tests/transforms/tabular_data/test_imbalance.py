import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba
from badgers.transforms.tabular_data.imbalance import RandomSamplingFeaturesTransformer, \
    RandomSamplingClassesTransformer, RandomSamplingTargetsTransformer
from tests.transforms.tabular_data import generate_test_data_with_classification_labels, \
    generate_test_data_with_regression_targets


class TestRandomSamplingClassesTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_transform(self):

        for input_type, (X, y) in self.input_test_data.items():
            if input_type in ['numpy_1D', 'pandas_1D']:
                proportion_classes = {0: 0.9, 1: 0.1}
            elif input_type in ['numpy_2D', 'pandas_2D']:
                proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
            else:
                self.fail()
            transformer = RandomSamplingClassesTransformer(proportion_classes=proportion_classes)

            Xt = transformer.fit_transform(X.copy(), y)
            # assert arrays have same size
            self.assertEqual(Xt.shape[1], X.shape[1])
            self.assertEqual(Xt.shape[0], len(transformer.labels_))


class TestRandomSamplingFeaturesTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_all_transformers(self):
        def proba_func(X):
            feature = X[:, 0]
            return normalize_proba(
                (np.max(feature) - feature) / (np.max(feature) - np.min(feature))
            )

        transformer = RandomSamplingFeaturesTransformer(sampling_proba_func=proba_func)

        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=transformer.__class__, input_type=input_type):
                Xt = transformer.transform(X.copy())
                # assert arrays have same size
                self.assertEqual(Xt.shape[1], X.shape[1])
                self.assertEqual(Xt.shape[0], X.shape[0])


class TestRandomSamplingTargetsTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_regression_targets(rng=self.rng)

    def test_all_transformers(self):
        def proba_func(y):
            return normalize_proba(
                (np.max(y) - y) / (np.max(y) - np.min(y))
            )

        transformer = RandomSamplingTargetsTransformer(sampling_proba_func=proba_func)

        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=transformer.__class__, input_type=input_type):
                Xt = transformer.fit_transform(X.copy(), y)
                # assert arrays have same size
                self.assertEqual(Xt.shape[1], X.shape[1])
                self.assertEqual(Xt.shape[0], X.shape[0])


if __name__ == '__main__':
    unittest.main()
