import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba
from badgers.generators.tabular_data.imbalance import RandomSamplingFeaturesGenerator, \
    RandomSamplingClassesGenerator, RandomSamplingTargetsGenerator
from tests.generators.tabular_data import generate_test_data_with_classification_labels, \
    generate_test_data_with_regression_targets


class TestRandomSamplingClassesGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_generate(self):

        for input_type, (X, y) in self.input_test_data.items():

            proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}

            generator = RandomSamplingClassesGenerator(random_generator=self.rng)

            Xt, yt = generator.generate(X.copy(), y, proportion_classes=proportion_classes)
            # assert arrays have same size
            if input_type[-2:] == '1D':
                self.assertEqual(Xt.shape[1], 1)
            else:
                self.assertEqual(Xt.shape[1], 10)
            self.assertEqual(Xt.shape[0], len(yt))
            # assert pandas DataFrame get the same columns before and after the call to generate function
            if input_type[:6] == 'pandas':
                self.assertListEqual(list(X.columns), list(Xt.columns))


class TestRandomSamplingFeaturesGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_generate(self):
        def proba_func(X):
            feature = X.iloc[:, 0]
            return normalize_proba(
                (np.max(feature) - feature) / (np.max(feature) - np.min(feature))
            )

        generator = RandomSamplingFeaturesGenerator()

        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y, sampling_proba_func=proba_func)
                # assert arrays have same size
                if input_type[-2:] == '1D':
                    self.assertEqual(Xt.shape[1], 1)
                else:
                    self.assertEqual(Xt.shape[1], 10)
                self.assertEqual(Xt.shape[0], len(yt))
                # assert pandas DataFrame get the same columns before and after the call to generate function
                if input_type in ['pandas_1D', 'pandas_2D']:
                    self.assertListEqual(list(X.columns), list(Xt.columns))


class TestRandomSamplingTargetsGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_regression_targets(rng=self.rng)

    def test_generate(self):
        def proba_func(y):
            return normalize_proba(
                (np.max(y) - y) / (np.max(y) - np.min(y))
            )

        generator = RandomSamplingTargetsGenerator()

        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y, sampling_proba_func=proba_func)
                # assert arrays have same size
                if input_type[-2:] == '1D':
                    self.assertEqual(Xt.shape[1], 1)
                else:
                    self.assertEqual(Xt.shape[1], 10)
                self.assertEqual(Xt.shape[0], len(yt))
                # assert pandas DataFrame get the same columns before and after the call to generate function
                if input_type in ['pandas_1D', 'pandas_2D']:
                    self.assertListEqual(list(X.columns), list(Xt.columns))


if __name__ == '__main__':
    unittest.main()
