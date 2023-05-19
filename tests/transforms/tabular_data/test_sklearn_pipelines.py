import unittest

from numpy.random import default_rng
from sklearn.pipeline import make_pipeline

from badgers.transforms.tabular_data.noise import NoiseTransformer
from badgers.transforms.tabular_data.outliers import OutliersTransformer
from tests.transforms.tabular_data import generate_test_data_without_labels


class TestPipelinesOutliers(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_without_labels(rng=self.rng)
        self.transformers_classes = OutliersTransformer.__subclasses__()

    def test_pipeline_single_transformer(self):
        """
        Test sklearn pipeline.transform(X). The pipeline consists in a single OutliersTransformer

        Loop through all OutliersTransformer subclasses and for each creates a sklearn pipeline to test
        """
        for cls in self.transformers_classes:
            pipeline = make_pipeline(cls())
            for input_type, X in self.input_test_data.items():
                with self.subTest(input_type=input_type):
                    Xt = pipeline.transform(X)
                    self.assertEqual(X.shape[1], Xt.shape[1])


class TestPipelinesNoise(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_without_labels(rng=self.rng)
        self.transformers_classes = NoiseTransformer.__subclasses__()

    def test_pipeline_single_transformer(self):
        """
        Test sklearn pipeline.transform(X). The pipeline consists in a single NoiseTransformer

        Loop through all NoiseTransformer subclasses and for each creates a sklearn pipeline to test
        """
        for cls in self.transformers_classes:
            pipeline = make_pipeline(cls())
            for input_type, X in self.input_test_data.items():
                with self.subTest(input_type=input_type):
                    Xt = pipeline.transform(X)
                    self.assertEqual(X.shape, Xt.shape)


if __name__ == '__main__':
    unittest.main()
