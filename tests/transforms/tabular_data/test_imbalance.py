import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba
from badgers.transforms.tabular_data.imbalance import RandomSamplingFeaturesTransformer, RandomSamplingClassesTransformer
from tests.transforms.tabular_data import generate_test_data_with_labels


class TestIRandomSamplingClassesTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_labels(rng=self.rng)

    def test_all_transformers(self):
        transformer = RandomSamplingClassesTransformer()
        for input_type, (X,y) in self.input_test_data.items():
            with self.subTest(transformer=transformer.__class__, input_type=input_type):
                Xt = transformer.fit_transform(X.copy(), y)
                # assert arrays have same size
                self.assertEqual(Xt.shape[1], X.shape[1])
                self.assertEqual(Xt.shape[0], transformer.labels_.shape[0])

class TestRandomSamplingFeaturesTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_labels(rng=self.rng)



    def test_all_transformers(self):
        def proba_func(X):
            feature = X[:,0]
            return normalize_proba(
                (np.max(feature)-feature) / (np.max(feature)-np.min(feature))
            )

        transformer = RandomSamplingFeaturesTransformer(percentage_missing=10, sampling_proba_func=proba_func)

        for input_type, (X,y) in self.input_test_data.items():
            with self.subTest(transformer=transformer.__class__, input_type=input_type):
                Xt = transformer.transform(X.copy())
                # assert arrays have same size
                self.assertEqual(Xt.shape[1], X.shape[1])
                self.assertEqual(Xt.shape[0], int(X.shape[0] * (100 - transformer.percentage_missing) / 100))

if __name__ == '__main__':
    unittest.main()
