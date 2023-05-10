import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.transforms.tabular_data.noise import NoiseTransformer
from tests.transforms.tabular_data import generate_test_data


class TestNoiseTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformers_classes = NoiseTransformer.__subclasses__()
        self.input_test_data = generate_test_data(rng=self.rng)

    def assertIncreaseVariance(self, X, Xt):
        self.assertTrue(all(np.var(X, axis=0) < np.var(Xt, axis=0)))

    def test_all_transformers(self):
        """
        run generic tests for all transformer classes:
        - checks that the transformed array has the same shape as the input array
        - checks that the variance of the transformed array is greater than the one of the input array
        """
        for cls in self.transformers_classes:
            transformer = cls()
            for input_type, X in self.input_test_data.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt = transformer.transform(X.copy())
                    # assert arrays have same shape
                    self.assertEqual(X.shape, Xt.shape)
                    # assert variance is greater after the transformation
                    self.assertIncreaseVariance(X, Xt)


if __name__ == '__main__':
    unittest.main()
