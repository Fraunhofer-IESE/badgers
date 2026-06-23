import unittest
import numpy as np
from benchmarks.models import (
    Scenario, FunctionalCheck, GeneratorBenchmark,
    PerformanceStats, FunctionalResult, BenchmarkResult, RunMeta
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


class TestFunctionalCheck(unittest.TestCase):
    def test_create_check(self):
        def my_check(X, y, Xt, yt, params):
            return Xt.shape == X.shape

        fc = FunctionalCheck(
            name="same_shape",
            description="Output has same shape as input",
            check=my_check,
        )
        self.assertEqual(fc.name, "same_shape")
        self.assertTrue(
            fc.check(
                np.zeros((10, 2)), None,
                np.zeros((10, 2)), None,
                {},
            )
        )


class TestGeneratorBenchmark(unittest.TestCase):
    def test_create_benchmark(self):
        s = Scenario("s1", "tabular", lambda rng: (None, None))
        fc = FunctionalCheck("c1", "desc", lambda *a, **kw: True)

        gb = GeneratorBenchmark(
            generator_cls=type("FakeGen", (), {}),
            name="FakeGen",
            module_path="tabular_data.test",
            default_params={"p": 1},
            scenarios=[s],
            functional_checks=[fc],
        )
        self.assertEqual(gb.name, "FakeGen")
        self.assertEqual(gb.module_path, "tabular_data.test")
        self.assertEqual(len(gb.scenarios), 1)
        self.assertEqual(len(gb.functional_checks), 1)


class TestPerformanceStats(unittest.TestCase):
    def test_create_stats(self):
        ps = PerformanceStats(
            min=1.0, max=5.0, mean=3.0, median=2.5, stddev=1.5, iterations=5
        )
        self.assertEqual(ps.min, 1.0)
        self.assertEqual(ps.iterations, 5)


class TestFunctionalResult(unittest.TestCase):
    def test_all_passed(self):
        fr = FunctionalResult(
            passed=3, failed=0,
            checks=[
                {"name": "c1", "passed": True},
                {"name": "c2", "passed": True},
                {"name": "c3", "passed": True},
            ],
        )
        self.assertEqual(fr.passed, 3)
        self.assertEqual(fr.failed, 0)

    def test_some_failed(self):
        fr = FunctionalResult(
            passed=2, failed=1,
            checks=[
                {"name": "c1", "passed": True},
                {"name": "c2", "passed": False},
                {"name": "c3", "passed": True},
            ],
        )
        self.assertEqual(fr.passed, 2)
        self.assertEqual(fr.failed, 1)


class TestBenchmarkResult(unittest.TestCase):
    def test_create_result(self):
        ps = PerformanceStats(1, 2, 1.5, 1.5, 0.5, 5)
        fr = FunctionalResult(1, 0, [{"name": "c1", "passed": True}])
        br = BenchmarkResult(
            generator="tabular_data.test.FakeGen",
            scenario="test_scenario",
            params={"p": 1},
            functional=fr,
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