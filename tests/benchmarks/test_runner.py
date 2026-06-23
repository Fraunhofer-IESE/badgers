import unittest
import numpy as np
from benchmarks.models import Scenario, GeneratorBenchmark
from benchmarks.runner import run_performance, run_all


class FakeGenerator:
    def __init__(self, random_generator=None):
        self.random_generator = random_generator

    def generate(self, X, y, **params):
        return X, y


class TestRunPerformance(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=lambda rng: (np.zeros((100, 2)), None),
        )

    def test_measures_time_and_memory(self):
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
        )
        results = run_performance([gb], iterations=3)
        self.assertEqual(len(results), 1)
        perf = results[0].performance
        self.assertIn("time_ms", perf)
        self.assertIn("memory_mb", perf)
        self.assertEqual(perf["time_ms"].iterations, 3)
        self.assertGreaterEqual(perf["time_ms"].min, 0)

    def test_run_all_delegates_to_performance(self):
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
        )
        results = run_all([gb], iterations=2)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].performance)
        self.assertIn("time_ms", results[0].performance)