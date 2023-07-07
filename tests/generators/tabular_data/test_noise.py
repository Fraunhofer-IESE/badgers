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
        generator = GaussianNoiseGenerator(repeat=1)
        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), None)
                # assert arrays have same size
                self.assertEqual(X.shape, Xt.shape)

        generator = GaussianNoiseGenerator(repeat=5)
        for input_type, (X, y) in self.input_test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), None)
                # assert shapes increase with repeat > 1
                self.assertEqual(X.shape[0] * 5, Xt.shape[0])
                self.assertEqual(X.shape[1], Xt.shape[1])


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
            generator = GaussianNoiseClassesGenerator(repeat=1, noise_std_per_class=noise_std_per_class)
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y)
                # assert arrays have same size
                self.assertEqual(X.shape, Xt.shape)
                self.assertEqual(len(y), len(yt))

        for input_type, (X, y) in self.input_test_data.items():
            noise_std_per_class = {label: 0.1 for label in np.unique(y)}
            generator = GaussianNoiseClassesGenerator(repeat=5, noise_std_per_class=noise_std_per_class)
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y)
                # assert shapes increase with repeat > 1
                self.assertEqual(X.shape[0] * 5, Xt.shape[0])
                self.assertEqual(len(y) * 5, len(yt))
                self.assertEqual(X.shape[1], Xt.shape[1])


if __name__ == '__main__':
    unittest.main()
