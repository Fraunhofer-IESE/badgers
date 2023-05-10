import unittest
from unittest import TestCase

from numpy.random import default_rng

from badgers.transforms.tabular_data.imbalance import ImbalanceTransformer
from tests.transforms.tabular_data import generate_test_data_with_labels


class TestImbalanceTransformer(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.transformers_classes = ImbalanceTransformer.__subclasses__()
        self.input_test_data = generate_test_data_with_labels(rng=self.rng)

    def test_all_transformers(self):
        for cls in self.transformers_classes:
            transformer = cls()
            for input_type, (X,y) in self.input_test_data.items():
                with self.subTest(transformer=transformer.__class__, input_type=input_type):
                    Xt = transformer.fit_transform(X.copy(), y)
                    # assert arrays have same shape
                    self.assertEqual(Xt.shape[1], X.shape[1])
                    self.assertEqual(Xt.shape[0], transformer.labels_.shape[0])


if __name__ == '__main__':
    unittest.main()
