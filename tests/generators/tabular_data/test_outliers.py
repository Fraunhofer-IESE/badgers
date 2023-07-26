from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from badgers.generators.tabular_data.outliers import ZScoreSamplingGenerator, HistogramSamplingGenerator, \
    HypersphereSamplingGenerator, LowDensitySamplingGenerator, DecompositionAndOutlierGenerator, \
    IndependentHistogramsGenerator
from tests.generators.tabular_data import generate_test_data_only_features


class TestOutliersGenerator(TestCase):
    """
    Implements generic tests for all OutliersGenerator objects
    """

    def assert_shape_yt(self, yt, outliers):
        """
        asserts that yt and outliers have the same length
        """
        self.assertEqual(yt.shape[0], outliers.shape[0])

    def assert_shape_outliers(self, X, outliers, generator):
        """
        asserts that the correct number of outliers have been produced
        with the correct number of features
        """
        self.assertEqual(outliers.shape[0], int(generator.n_outliers))
        self.assertEqual(outliers.shape[1], X.shape[1])


class TestZScoreSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = ZScoreSamplingGenerator(random_generator=self.rng, n_outliers=10)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def assert_zscore_larger_than_3(self, X, outliers):
        """
        Asserts that, at least in one dimension, the zscore of the generated outliers data points is greater than 3
        """
        # compute means and stds for checking z-score
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        # assert z-score > 3
        for row in range(outliers.shape[0]):
            values = outliers[row, :]
            z_scores = abs(values - means) / stds
            self.assertTrue(all(z_scores > 3.))

    def test_generator(self):
        """

        """
        for input_type, (X, y) in self.input_test_data.items():
            outliers, yt = self.generator.generate(X.copy(), y)
            with self.subTest(input_type=input_type):
                self.assert_zscore_larger_than_3(X, outliers)
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, generator=self.generator)


class TestHistogramSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = HistogramSamplingGenerator(random_generator=self.rng, n_outliers=10)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        for input_type, (X, y) in self.input_test_data.items():
            if X.shape[1] > 5:
                with self.assertRaises(NotImplementedError):
                    _, _ = self.generator.generate(X.copy(), y)
            else:
                outliers, yt = self.generator.generate(X.copy(), y)
                with self.subTest(input_type=input_type):
                    self.assert_shape_yt(yt, outliers)
                    self.assert_shape_outliers(X, outliers, generator=self.generator)


class TestHypersphereSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = HypersphereSamplingGenerator(random_generator=self.rng, n_outliers=10)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        for input_type, (X, y) in self.input_test_data.items():
            outliers, yt = self.generator.generate(X.copy(), y)
            with self.subTest(input_type=input_type):
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, generator=self.generator)


class TestLowDensitySamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = LowDensitySamplingGenerator(random_generator=self.rng, n_outliers=10)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        for input_type, (X, y) in self.input_test_data.items():
            if input_type in ['numpy_1D', 'pandas_1D']:
                self.skipTest("Not testing numpy_1D and pandas_1D")
            else:
                outliers, yt = self.generator.generate(X.copy(), y)
                with self.subTest(input_type=input_type):
                    self.assert_shape_yt(yt, outliers)
                    self.assert_shape_outliers(X, outliers, generator=self.generator)


class TestIndependentHistogramsGenerator(TestOutliersGenerator):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = IndependentHistogramsGenerator(random_generator=self.rng, n_outliers=10, bins=3)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        for input_type, (X, y) in self.input_test_data.items():
            outliers, yt = self.generator.generate(X.copy(), y)
            with self.subTest(input_type=input_type):
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, generator=self.generator)


class TestDecompositionAndOutlierGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        X, y = make_blobs(centers=5, n_features=10, n_samples=100)

        outliers_generators = [
            ZScoreSamplingGenerator(),
            HistogramSamplingGenerator(),
            HypersphereSamplingGenerator(),
            LowDensitySamplingGenerator()
        ]
        for outlier_generator in outliers_generators:
            generator = DecompositionAndOutlierGenerator(
                decomposition_transformer=PCA(n_components=3),
                outlier_generator=outlier_generator
            )
            outliers, yt = generator.generate(X.copy(), y)
            with self.subTest(outlier_generator=outlier_generator):
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, generator=generator)
