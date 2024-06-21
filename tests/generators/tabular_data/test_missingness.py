import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.generators.tabular_data.missingness import MissingValueGenerator
from tests.generators.tabular_data import generate_test_data_only_features


class TestMissingValueGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generators_classes = MissingValueGenerator.__subclasses__()
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_all_generators(self):
        percentage_missing = 0.1
        for cls in self.generators_classes:
            transformer = cls()
            for input_type, (X, y) in self.input_test_data.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt, _ = transformer.generate(X.copy(), y, percentage_missing=percentage_missing)
                    # assert arrays have same size
                    # assert arrays have same size
                    if input_type[-2:] == '1D':
                        self.assertEqual(Xt.shape[1], 1)
                    else:
                        self.assertEqual(Xt.shape[1], 10)

                    # assert that the right number of nans have been generated
                    if input_type[-2:] == '1D':
                        self.assertEqual(
                            np.sum(np.isnan(Xt).sum()),
                            int(percentage_missing * 100)
                        )
                    else:
                        self.assertEqual(
                            np.sum(np.isnan(Xt).sum()),
                            int(percentage_missing * 1000)
                        )



if __name__ == '__main__':
    unittest.main()
