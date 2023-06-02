import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.generators.tabular_data.missingness import MissingValueGenerator
from tests.transforms.tabular_data import generate_test_data_only_features


class TestMissingValueTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformers_classes = MissingValueGenerator.__subclasses__()
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_all_transformers(self):
        for cls in self.transformers_classes:
            transformer = cls()
            for input_type, X in self.input_test_data.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt, _ = transformer.generate(X.copy(), None)
                    # assert arrays have same size
                    self.assertEqual(X.shape, Xt.shape)
                    # assert that the right number of nans have been generated
                    self.assertEqual(np.sum(np.isnan(Xt).sum()), transformer.percentage_missing / 100. * X.shape[0] * X.shape[1])


if __name__ == '__main__':
    unittest.main()
