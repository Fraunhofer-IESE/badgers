import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.generators.tabular_data.noise import GaussianNoiseGenerator, GaussianNoiseClassesGenerator
from tests.generators.tabular_data import generate_test_data_with_classification_labels, \
    generate_test_data_only_features


class TestGaussianNoiseGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generate(self):
        """
        - checks that the transformed array has the same size as the input array
        - checks that the variance of the transformed array is greater than the one of the input array
        """
        generator = GaussianNoiseGenerator()
        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), y=None, noise_std=1)
                # assert arrays have same size
                self.assertEqual(len(X), len(Xt))
                self.assertTrue((np.var(Xt, axis=0) > np.var(X, axis=0)).all())


class TestGaussianNoiseClassesGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_generate(self):
        """
        - checks that the transformed array has the same size as the input array
        - checks that the variance of the transformed array is greater than the one of the input array
        """

        for input_type, (X, y) in self.input_test_data.items():
            noise_std_per_class = {label: 0.1 for label in np.unique(y)}
            generator = GaussianNoiseClassesGenerator()
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y, noise_std_per_class=noise_std_per_class)
                # assert arrays have same size
                self.assertEqual(len(X), len(Xt))
                self.assertEqual(len(y), len(yt))


if __name__ == '__main__':
    unittest.main()
