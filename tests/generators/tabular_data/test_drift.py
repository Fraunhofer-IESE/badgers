from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.generators.tabular_data.drift import RandomShiftGenerator, RandomShiftClassesGenerator
from tests.generators.tabular_data import generate_test_data_with_classification_labels, \
    generate_test_data_only_features


class TestRandomShiftGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.test_data = generate_test_data_only_features(rng=self.rng)

    def test_generate_shift_std_number(self):
        """
        Asserts:
        - Arrays X and Xt have the same shape
        """
        generator = RandomShiftGenerator(random_generator=self.rng)
        for input_type, (X, y) in self.test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), y, shift_std=0.1)
                # assert arrays have same size
                self.assertEqual(len(X), len(Xt))

    def test_generate_shift_std_array(self):
        """
        Asserts:
        - Arrays X and Xt have the same shape
        """
        generator = RandomShiftGenerator(random_generator=self.rng)
        for input_type, (X, y) in self.test_data.items():
            if input_type[-2:] == '1D':
                shift_std = np.array([0.1])
            else:
                shift_std = np.linspace(0.1, 1, 10)
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
                # assert arrays have same size
                self.assertEqual(len(X), len(Xt))


class TestRandomShiftGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_generate_shift_std_number(self):
        """
        Asserts:
        - Arrays X and Xt have the same shape
        """
        generator = RandomShiftClassesGenerator(random_generator=self.rng)
        for input_type, (X, y) in self.test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y, shift_std=0.1)
                # assert arrays have same size
                self.assertEqual(len(X), len(Xt))

    def test_generate_shift_std_array(self):
        """
        Asserts:
        - Arrays X and Xt have the same shape
        """
        generator = RandomShiftClassesGenerator(random_generator=self.rng)
        for input_type, (X, y) in self.test_data.items():
            if input_type[-2:] == '1D':
                shift_std = np.array([0.1])
            else:
                shift_std = np.linspace(0.1, 1, 5)
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
                # assert arrays have same size
                self.assertEqual(len(X), len(Xt))
