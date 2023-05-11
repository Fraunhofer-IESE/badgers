import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.transforms.tabular_data.missingness import MissingValueTransformer
from tests.transforms.tabular_data import generate_test_data_without_labels


class TestMissingValueTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformers_classes = MissingValueTransformer.__subclasses__()
        self.input_test_data = generate_test_data_without_labels(rng=self.rng)

    def test_all_transformers(self):
        for cls in self.transformers_classes:
            transformer = cls()
            for input_type, X in self.input_test_data.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt = transformer.transform(X.copy())
                    # assert arrays have same shape
                    self.assertEqual(X.shape, Xt.shape)
                    # assert that the right number of nans have been generated
                    self.assertEqual(np.isnan(Xt).sum(), transformer.percentage_missing / 100. * X.shape[0] * X.shape[1])


if __name__ == '__main__':
    unittest.main()
