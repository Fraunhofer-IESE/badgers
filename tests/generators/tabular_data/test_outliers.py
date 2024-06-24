import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
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
        self.assertEqual(len(yt), len(outliers))

    def assert_shape_outliers(self, X, outliers, n_outliers):
        """
        asserts that the correct number of outliers have been produced
        with the correct number of features
        """
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        self.assertEqual(outliers.shape[0], n_outliers)
        self.assertEqual(outliers.shape[1], X.shape[1])


class TestZScoreSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = ZScoreSamplingGenerator(random_generator=self.rng)
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
        n_outliers = 10
        for input_type, (X, y) in self.input_test_data.items():
            outliers, yt = self.generator.generate(X.copy(), y, n_outliers=n_outliers)
            with self.subTest(input_type=input_type):
                self.assert_zscore_larger_than_3(X, outliers)
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, n_outliers=n_outliers)


class TestHistogramSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = HistogramSamplingGenerator(random_generator=self.rng)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        n_outliers = 10
        bins = 3
        for input_type, (X, y) in self.input_test_data.items():
            if input_type[-2:] == '2D':
                with self.subTest(input_type=input_type, ncols=10):
                    with self.assertRaises(NotImplementedError):
                        _, _ = self.generator.generate(X.copy(), y, n_outliers=n_outliers, bins=bins)

                    with self.subTest(input_type=input_type, ncols=3):
                        X = pd.DataFrame(X).iloc[:,:3]
                        outliers, yt = self.generator.generate(X, y, n_outliers=n_outliers, bins=bins)
                        self.assert_shape_yt(yt, outliers)
                        self.assert_shape_outliers(X, outliers, n_outliers)


class TestHypersphereSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = HypersphereSamplingGenerator(random_generator=self.rng)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        n_outliers = 10
        for input_type, (X, y) in self.input_test_data.items():
            outliers, yt = self.generator.generate(X=X.copy(), y=y, n_outliers=n_outliers)
            with self.subTest(input_type=input_type):
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, n_outliers)


class TestLowDensitySamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = LowDensitySamplingGenerator(random_generator=self.rng)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        n_outliers = 10
        for input_type, (X, y) in self.input_test_data.items():
            if input_type in ['numpy_1D', 'pandas_1D']:
                self.skipTest("Not testing numpy_1D and pandas_1D")
            else:
                outliers, yt = self.generator.generate(X=X.copy(), y=y, n_outliers=n_outliers)


class TestIndependentHistogramsGenerator(TestOutliersGenerator):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = IndependentHistogramsGenerator(random_generator=self.rng)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        n_outliers = 10
        bins = 3
        for input_type, (X, y) in self.input_test_data.items():
            outliers, yt = self.generator.generate(X.copy(), y, n_outliers=n_outliers, bins=bins)
            with self.subTest(input_type=input_type):
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, n_outliers)


class TestDecompositionAndOutlierGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_generator(self):
        """

        """
        X, y = make_blobs(centers=5, n_features=10, n_samples=100)
        n_outliers = 10
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
            outliers, yt = generator.generate(X.copy(), y, n_outliers=n_outliers)
            with self.subTest(outlier_generator=outlier_generator):
                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, n_outliers)


if __name__ == '__main__':
    unittest.main()
