from unittest import TestCase

from numpy.random import default_rng

from badgers.transforms.tabular_data.drift import DriftTransformer
from tests.transforms.tabular_data import generate_test_data_with_labels, generate_test_data_without_labels


class TestDriftTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_data_without_labels = generate_test_data_without_labels(rng=self.rng)
        self.transformer_without_fit_classes = [cls for cls in DriftTransformer.__subclasses__() if
                                                not hasattr(cls, 'fit')]
        self.input_data_with_labels = generate_test_data_with_labels(rng=self.rng)
        self.transformer_with_fit_classes = [cls for cls in DriftTransformer.__subclasses__() if hasattr(cls, 'fit')]

    def test_all_transformers(self):
        """
        run generic tests for all transformer classes:
        - checks that the transformed array has the same shape as the input array
        """
        for cls in self.transformer_without_fit_classes:
            transformer = cls()
            for input_type, X in self.input_data_without_labels.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt = transformer.transform(X.copy())
                    # assert arrays have same shape
                    self.assertEqual(X.shape, Xt.shape)

        for cls in self.transformer_with_fit_classes:
            transformer = cls()
            for input_type, (X, y) in self.input_data_with_labels.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt = transformer.fit_transform(X.copy(), y)
                    # assert arrays have same shape
                    self.assertEqual(Xt.shape, X.shape)
