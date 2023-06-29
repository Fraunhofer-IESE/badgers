from unittest import TestCase

from numpy.random import default_rng

from badgers.generators.tabular_data.drift import RandomShiftGenerator, RandomShiftClassesGenerator
from tests.generators.tabular_data import generate_test_data_with_classification_labels, \
    generate_test_data_only_features


class TestRandomShiftGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.test_data = generate_test_data_only_features(rng=self.rng)

    def test_generate(self):
        """
        Asserts:
        - Arrays X and Xt have the same shape
        """
        generator = RandomShiftGenerator(random_generator=self.rng)
        for input_type, (X, y) in self.test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, _ = generator.generate(X.copy(), y)
                # assert arrays have same size
                self.assertEqual(X.shape, Xt.shape)


class TestRandomShiftGenerator(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.test_data = generate_test_data_with_classification_labels(rng=self.rng)

    def test_generate(self):
        """
        Asserts:
        - Arrays X and Xt have the same shape
        """
        generator = RandomShiftClassesGenerator(random_generator=self.rng, shift_std=0.1)
        for input_type, (X, y) in self.test_data.items():
            with self.subTest(transformer=generator.__class__, input_type=input_type):
                Xt, yt = generator.generate(X.copy(), y)
                # assert arrays have same size
                self.assertEqual(X.shape, Xt.shape)
