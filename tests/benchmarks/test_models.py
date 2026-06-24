import unittest
import numpy as np
from benchmarks.models import (
    Scenario, GeneratorBenchmark,
    PerformanceStats, BenchmarkResult, RunMeta
)


class TestScenario(unittest.TestCase):
    def test_create_scenario(self):
        def factory(rng):
            return np.zeros((10, 2)), None

        s = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=factory,
            tags=["small", "2D"],
        )
        self.assertEqual(s.name, "test_scenario")
        self.assertEqual(s.data_type, "tabular")
        self.assertEqual(s.tags, ["small", "2D"])
        X, y = s.factory(np.random.default_rng(0))
        self.assertEqual(X.shape, (10, 2))
        self.assertIsNone(y)

    def test_scenario_default_tags(self):
        s = Scenario(
            name="test",
            data_type="tabular",
            factory=lambda rng: (None, None),
        )
        self.assertEqual(s.tags, [])


class TestGeneratorBenchmark(unittest.TestCase):
    def test_create_benchmark(self):
        s = Scenario("s1", "tabular", lambda rng: (None, None))

        gb = GeneratorBenchmark(
            generator_cls=type("FakeGen", (), {}),
            name="FakeGen",
            module_path="tabular_data.test",
            default_params={"p": 1},
            scenarios=[s],
        )
        self.assertEqual(gb.name, "FakeGen")
        self.assertEqual(gb.module_path, "tabular_data.test")
        self.assertEqual(len(gb.scenarios), 1)


class TestPerformanceStats(unittest.TestCase):
    def test_create_stats(self):
        ps = PerformanceStats(
            min=1.0, max=5.0, mean=3.0, median=2.5, stddev=1.5, iterations=5
        )
        self.assertEqual(ps.min, 1.0)
        self.assertEqual(ps.iterations, 5)


class TestBenchmarkResult(unittest.TestCase):
    def test_create_result(self):
        ps = PerformanceStats(1, 2, 1.5, 1.5, 0.5, 5)
        br = BenchmarkResult(
            generator="tabular_data.test.FakeGen",
            scenario="test_scenario",
            params={"p": 1},
            performance={"time_ms": ps, "memory_mb": ps},
        )
        self.assertEqual(br.generator, "tabular_data.test.FakeGen")
        self.assertEqual(br.performance["time_ms"].mean, 1.5)


class TestRunMeta(unittest.TestCase):
    def test_create_meta(self):
        rm = RunMeta(
            timestamp="2026-06-22T14:30:00",
            git_commit="abc1234",
            git_branch="main",
            python_version="3.12.4",
            platform="Windows-10",
        )
        self.assertEqual(rm.git_branch, "main")