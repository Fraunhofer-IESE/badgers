import unittest
from benchmarks.comparator import compare_results, _classify_change


class TestClassifyChange(unittest.TestCase):
    def test_regression(self):
        self.assertEqual(_classify_change(100, 150, 0.2), "regression")

    def test_improvement(self):
        self.assertEqual(_classify_change(150, 100, 0.2), "improvement")

    def test_unchanged(self):
        self.assertEqual(_classify_change(100, 110, 0.2), "unchanged")

    def test_exact_threshold(self):
        self.assertEqual(_classify_change(100, 120, 0.2), "unchanged")

    def test_zero_baseline(self):
        self.assertEqual(_classify_change(0, 10, 0.2), "regression")


class TestCompareResults(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "meta": {"git_branch": "main"},
            "results": [
                {
                    "generator": "tabular_data.noise.GaussianNoise",
                    "scenario": "small_blobs_2d",
                    "params": {"noise_std": 0.5},
                    "performance": {
                        "time_ms": {"mean": 10.0, "min": 9.0, "max": 11.0,
                                     "median": 10.0, "stddev": 1.0, "iterations": 5},
                        "memory_mb": {"mean": 2.0, "min": 1.8, "max": 2.2,
                                       "median": 2.0, "stddev": 0.2, "iterations": 5},
                    },
                },
            ],
        }

    def test_no_changes(self):
        target = {
            "meta": {"git_branch": "feature"},
            "results": [
                {
                    "generator": "tabular_data.noise.GaussianNoise",
                    "scenario": "small_blobs_2d",
                    "params": {"noise_std": 0.5},
                    "performance": {
                        "time_ms": {"mean": 10.0, "min": 9.0, "max": 11.0,
                                     "median": 10.0, "stddev": 1.0, "iterations": 5},
                        "memory_mb": {"mean": 2.0, "min": 1.8, "max": 2.2,
                                       "median": 2.0, "stddev": 0.2, "iterations": 5},
                    },
                },
            ],
        }
        report = compare_results(self.baseline, target)
        self.assertIn("unchanged", report.lower())

    def test_regression_detected(self):
        target = {
            "meta": {"git_branch": "feature"},
            "results": [
                {
                    "generator": "tabular_data.noise.GaussianNoise",
                    "scenario": "small_blobs_2d",
                    "params": {"noise_std": 0.5},
                    "performance": {
                        "time_ms": {"mean": 25.0, "min": 24.0, "max": 26.0,
                                     "median": 25.0, "stddev": 1.0, "iterations": 5},
                        "memory_mb": {"mean": 2.0, "min": 1.8, "max": 2.2,
                                       "median": 2.0, "stddev": 0.2, "iterations": 5},
                    },
                },
            ],
        }
        report = compare_results(self.baseline, target)
        self.assertIn("regression", report.lower())

    def test_missing_generator_in_target(self):
        target = {"meta": {"git_branch": "feature"}, "results": []}
        report = compare_results(self.baseline, target)
        self.assertIn("missing", report.lower())

    def test_missing_generator_in_baseline(self):
        target = {
            "meta": {"git_branch": "feature"},
            "results": [
                {
                    "generator": "tabular_data.noise.NewGen",
                    "scenario": "small_blobs_2d",
                    "params": {},
                    "performance": {
                        "time_ms": {"mean": 5.0, "min": 4.0, "max": 6.0,
                                     "median": 5.0, "stddev": 1.0, "iterations": 5},
                        "memory_mb": {"mean": 1.0, "min": 0.9, "max": 1.1,
                                       "median": 1.0, "stddev": 0.1, "iterations": 5},
                    },
                },
            ],
        }
        report = compare_results(self.baseline, target)
        self.assertIn("new", report.lower())