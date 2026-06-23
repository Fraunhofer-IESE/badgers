import unittest
import numpy as np
import pandas as pd
from benchmarks.checks.common import CHECK_SAME_SHAPE, CHECK_NO_NANS
from benchmarks.checks.tabular import CHECK_INCREASED_VARIANCE, CHECK_OUTLIER_COUNT
from benchmarks.checks.time_series import CHECK_PATTERN_COUNT


class TestCommonChecks(unittest.TestCase):
    def test_same_shape_passes(self):
        X = np.zeros((10, 2))
        Xt = np.zeros((10, 2))
        self.assertTrue(CHECK_SAME_SHAPE.check(X, None, Xt, None, {}))

    def test_same_shape_fails(self):
        X = np.zeros((10, 2))
        Xt = np.zeros((5, 2))
        self.assertFalse(CHECK_SAME_SHAPE.check(X, None, Xt, None, {}))

    def test_no_nans_passes(self):
        Xt = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        self.assertTrue(CHECK_NO_NANS.check(None, None, Xt, None, {}))

    def test_no_nans_fails(self):
        Xt = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        self.assertFalse(CHECK_NO_NANS.check(None, None, Xt, None, {}))


class TestTabularChecks(unittest.TestCase):
    def test_increased_variance_passes(self):
        X = np.zeros((100, 2))
        Xt = np.random.normal(0, 1, (100, 2))
        self.assertTrue(CHECK_INCREASED_VARIANCE.check(X, None, Xt, None, {}))

    def test_increased_variance_fails(self):
        X = np.random.normal(0, 2, (100, 2))
        Xt = np.random.normal(0, 0.1, (100, 2))
        self.assertFalse(CHECK_INCREASED_VARIANCE.check(X, None, Xt, None, {}))

    def test_outlier_count_passes(self):
        X = np.zeros((100, 2))
        Xt = np.zeros((10, 2))
        yt = np.array(["outliers"] * 10)
        self.assertTrue(CHECK_OUTLIER_COUNT.check(X, None, Xt, yt, {"n_outliers": 10}))

    def test_outlier_count_fails(self):
        X = np.zeros((100, 2))
        Xt = np.zeros((5, 2))
        yt = np.array(["outliers"] * 5)
        self.assertFalse(CHECK_OUTLIER_COUNT.check(X, None, Xt, yt, {"n_outliers": 10}))


class TestTimeSeriesChecks(unittest.TestCase):
    def test_pattern_count_passes(self):
        X = np.zeros((100, 1))
        Xt = np.zeros((100, 1))
        class FakeGen:
            patterns_indices_ = [(10, 20), (40, 50), (70, 80)]
        self.assertTrue(CHECK_PATTERN_COUNT.check(
            X, None, Xt, None, {"n_patterns": 3},
            generator=FakeGen(),
        ))

    def test_pattern_count_fails(self):
        class FakeGen:
            patterns_indices_ = [(10, 20)]
        self.assertFalse(CHECK_PATTERN_COUNT.check(
            np.zeros((100, 1)), None, np.zeros((100, 1)), None,
            {"n_patterns": 3},
            generator=FakeGen(),
        ))