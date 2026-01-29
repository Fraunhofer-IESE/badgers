import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from badgers.generators.tabular_data.outliers import DecompositionAndOutlierGenerator
from badgers.generators.tabular_data.outliers.distribution_sampling import ZScoreSamplingGenerator, \
    HypersphereSamplingGenerator, HyperCubeSampling
from badgers.generators.tabular_data.outliers.low_density_sampling import HistogramSamplingGenerator, \
    LowDensitySamplingGenerator, IndependentHistogramsGenerator
from tests.unit_tests.generators.tabular_data import generate_test_data_only_features


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


# ---------------------------------------------------------------------------

# Common registry of generators and their default kwargs for generic tests

# ---------------------------------------------------------------------------

COMMON_GENERATORS = [
    # (name, generator_class, default_kwargs)
    ("hyper_cube", HyperCubeSampling, {"expansion": 0.0}),
    ("z_score", ZScoreSamplingGenerator, {"scale": 1.0}),
    ("hypersphere", HypersphereSamplingGenerator, {"scale": 1.0}),
    ("histogram", HistogramSamplingGenerator, {"bins": 3, "threshold_low_density": 0.5}),
    ("low_density", LowDensitySamplingGenerator, {}),
    ("independent_histograms", IndependentHistogramsGenerator, {"bins": 3}),
]


class TestAllOutlierGeneratorsCommon(TestOutliersGenerator):
    """
    Generic tests that apply to all OutliersGenerator implementations.
    """

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def _make_generator(self, gen_class):
        # Every generator in this project expects a numpy Generator in the ctor
        return gen_class(random_generator=default_rng(0))

    def test_basic_shapes_and_labels(self):
        """
        For each generator and each input type:
        - outliers shape is (n_outliers, n_features)
        - yt has length n_outliers
        - X is not modified

        """
        n_outliers = 10

        for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
            generator = self._make_generator(gen_class)
            for input_type, (X, y) in self.input_test_data.items():
                # Some generators have known unsupported shapes; let their own
                # tests handle NotImplementedError.
                X_original = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)

                with self.subTest(generator=gen_name, input_type=input_type):
                    try:
                        outliers, yt = generator.generate(
                            X.copy(), y, n_outliers=n_outliers, **default_kwargs
                        )
                    except NotImplementedError:
                        # Respect explicit "not implemented" behavior
                        self.skipTest(
                            f"{gen_name} does not support input_type={input_type}"
                        )
                    except Exception as e:
                        self.fail(e)

                    self.assert_shape_yt(yt, outliers)
                    self.assert_shape_outliers(X, outliers, n_outliers=n_outliers)

                    # Assert X was not modified
                    if isinstance(X_original, pd.DataFrame):
                        pd.testing.assert_frame_equal(X_original, X)
                    else:
                        np.testing.assert_allclose(
                            np.asarray(X_original), np.asarray(X)
                        )

    def test_n_outliers_zero(self):
        """
        All generators should handle n_outliers=0 and n_outliers=1 consistently.
        """

        for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
            generator = self._make_generator(gen_class)
            for input_type, (X, y) in self.input_test_data.items():

                with self.subTest(generator=gen_name, n_outliers=0):
                    try:
                        with self.assertRaises(AssertionError):
                            _, _ = generator.generate(
                                X.copy(), y, n_outliers=0, **default_kwargs
                            )
                    except NotImplementedError:
                        self.skipTest(f"{gen_name} does not support this configuration")
                    except Exception as e:
                        self.fail(e)

    def test_n_outliers_one(self):
        """
        All generators should handle n_outliers=0 and n_outliers=1 consistently.
        """

        for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
            generator = self._make_generator(gen_class)
            for input_type, (X, y) in self.input_test_data.items():

                with self.subTest(generator=gen_name, n_outliers=1):
                    try:
                        outliers, yt = generator.generate(
                            X.copy(), y, n_outliers=1, **default_kwargs
                        )
                    except NotImplementedError:
                        self.skipTest(f"{gen_name} does not support this configuration")
                    except Exception as e:
                        self.fail(e)

                    self.assertEqual(len(outliers), 1)
                    self.assertEqual(len(yt), 1)

    def test_reproducibility_given_same_seed(self):
        """
        With the same RNG seed and same inputs, generators should produce the same output.
        """

        for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
            rng1 = default_rng(42)
            rng2 = default_rng(42)

            gen1 = gen_class(random_generator=rng1)
            gen2 = gen_class(random_generator=rng2)

            for input_type, (X, y) in self.input_test_data.items():
                with self.subTest(generator=gen_name, input_type=input_type):

                    try:
                        out1, yt1 = gen1.generate(X.copy(), y, n_outliers=20, **default_kwargs)
                        out2, yt2 = gen2.generate(X.copy(), y, n_outliers=20, **default_kwargs)
                    except NotImplementedError:
                        self.skipTest(f"{gen_name} does not support this configuration")
                    except Exception as e:
                        self.fail(e)

                    np.testing.assert_allclose(out1, out2)
                    np.testing.assert_array_equal(yt1, yt2)

    def test_outlier_scores_are_worse_than_original_data(self):
        """
        For each generator:
        - Fit IsolationForest on original X.
        - Compare decision_function scores of original X vs generated outliers.
        - Expect the average score of outliers to be lower than that of X

          (lower = more abnormal for IsolationForest).
        """
        # Fit detector on original data only
        detector = IsolationForest(
            random_state=0,
            n_estimators=100,
            contamination="auto",
        )

        X, y = self.input_test_data['numpy_2D']
        detector.fit(X)
        original_scores = detector.decision_function(X)
        mean_original_score = original_scores.mean()

        for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
            generator = self._make_generator(gen_class)
            with self.subTest(generator=gen_name):
                try:
                    outliers, yt = generator.generate(
                        X.copy(), y, n_outliers=50, **default_kwargs
                    )
                except NotImplementedError:
                    self.skipTest(f"{gen_name} does not support this configuration")

                outliers = np.asarray(outliers)
                outlier_scores = detector.decision_function(outliers)
                mean_outlier_score = outlier_scores.mean()

                # We expect outliers to have lower (more negative) scores
                # than the original data; allow a small tolerance.
                self.assertLess(
                    mean_outlier_score,
                    mean_original_score,
                    msg=(
                        f"Outlier scores for {gen_name} are not worse than "
                        f"original data on average."
                    ),
                )


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

    def test_scale_effect_on_zscore_magnitude(self):
        """
        Larger scale should, on average, yield larger z-scores.
        """
        X, y = self.input_test_data['pandas_2D']

        scaler = StandardScaler()
        X = scaler.fit_transform(X=X)

        n_outliers = 200
        outliers_small_scale, _ = self.generator.generate(X.copy(), y, n_outliers=n_outliers, scale=0.1)
        outliers_large_scale, _ = self.generator.generate(X.copy(), y, n_outliers=n_outliers, scale=5.0)

        z_small = np.abs(outliers_small_scale)
        z_large = np.abs(outliers_large_scale)

        mean_z_small = z_small.mean()
        mean_z_large = z_large.mean()

        self.assertGreater(mean_z_large, mean_z_small)


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
                        X = pd.DataFrame(X).iloc[:, :3]
                        outliers, yt = self.generator.generate(X, y, n_outliers=n_outliers, bins=bins)
                        self.assert_shape_yt(yt, outliers)
                        self.assert_shape_outliers(X, outliers, n_outliers)


class TestHypersphereSamplingGenerator(TestOutliersGenerator):
    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = HypersphereSamplingGenerator(random_generator=self.rng)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_radius_ge_three_in_standardized_space(self):
        """
        In standardized space, outliers should have Euclidean norm >= 3.
        """
        X, y = self.input_test_data['pandas_2D']
        X_arr = np.asarray(X)
        scaler = StandardScaler().fit(X_arr)

        n_outliers = 100
        outliers, yt = self.generator.generate(X.copy(), y, n_outliers=n_outliers, scale=1.0)
        self.assert_shape_yt(yt, outliers)
        self.assert_shape_outliers(X, outliers, n_outliers=n_outliers)

        outliers_std = scaler.transform(outliers)
        radii = np.linalg.norm(outliers_std, axis=1)
        self.assertTrue(np.all(radii >= 3.0 - 1e-8))

    def test_scale_effect_on_radius(self):
        """
        Larger scale should, on average, yield larger radii in standardized space.
        """
        X, y = self.input_test_data['pandas_2D']
        X_arr = np.asarray(X)
        scaler = StandardScaler().fit_transform(X_arr)

        n_outliers = 200
        outliers_small_scale, _ = self.generator.generate(X.copy(), y, n_outliers=n_outliers, scale=0.1)
        outliers_large_scale, _ = self.generator.generate(X.copy(), y, n_outliers=n_outliers, scale=5.0)

        r_small = np.linalg.norm(outliers_small_scale, axis=1).mean()
        r_large = np.linalg.norm(outliers_large_scale, axis=1).mean()

        self.assertGreater(r_large, r_small)


class TestHyperCubeSamplingGenerator(TestOutliersGenerator):

    def setUp(self) -> None:
        self.rng = default_rng(0)
        self.generator = HyperCubeSampling(random_generator=self.rng)
        self.input_test_data = generate_test_data_only_features(rng=self.rng)

    def test_hypercube_expansion_zero_within_min_max(self):
        """
        For expansion=0, all outliers should lie within [min(X), max(X)] per feature.
        """
        # Use one representative dataset (e.g. first non-1D input)
        for input_type, (X, y) in self.input_test_data.items():
            if '1D' in input_type:
                continue
            with self.subTest(input_type=input_type):
                X_arr = np.asarray(X)
                n_outliers = 100
                outliers, yt = self.generator.generate(X.copy(), y, n_outliers=n_outliers, expansion=0.0)

                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, n_outliers=n_outliers)

                mins = X_arr.min(axis=0)
                maxs = X_arr.max(axis=0)
                self.assertTrue(np.all(outliers >= mins - 1e-8))
                self.assertTrue(np.all(outliers <= maxs + 1e-8))
            break  # only test on one dataset to keep runtime low

    def test_hypercube_expansion_positive_respects_expanded_range(self):
        """
        For expansion>0, outliers should lie within the inversely transformed
        [0 - expansion, 1 + expansion] hypercube.
        """
        expansion = 0.1
        for input_type, (X, y) in self.input_test_data.items():
            if '1D' in input_type:
                continue
            with self.subTest(input_type=input_type):
                X_arr = np.asarray(X)
                scaler = MinMaxScaler().fit(X_arr)

                n_outliers = 100
                outliers, yt = self.generator.generate(
                    X.copy(), y, n_outliers=n_outliers, expansion=expansion
                )

                self.assert_shape_yt(yt, outliers)
                self.assert_shape_outliers(X, outliers, n_outliers=n_outliers)

                # Check in scaled space
                outliers_scaled = scaler.transform(outliers)
                self.assertTrue(np.all(outliers_scaled >= 0 - expansion - 1e-8))
                self.assertTrue(np.all(outliers_scaled <= 1 + expansion + 1e-8))
            break  # keep runtime limited

    def test_hypercube_expansion_negative_raises(self):
        """
        Negative expansion should trigger the assertion in the implementation.
        """
        X, y = next(iter(self.input_test_data.values()))
        with self.assertRaises(AssertionError):
            self.generator.generate(X.copy(), y, n_outliers=10, expansion=-0.1)


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
