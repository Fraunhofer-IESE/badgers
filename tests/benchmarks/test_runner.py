import unittest
import numpy as np
from benchmarks.models import Scenario, FunctionalCheck, GeneratorBenchmark, BenchmarkResult
from benchmarks.runner import run_functional


class FakeGenerator:
    def __init__(self, random_generator=None):
        self.random_generator = random_generator

    def generate(self, X, y, **params):
        return X, y


class TestRunFunctional(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=lambda rng: (np.zeros((10, 2)), None),
        )

    def test_all_checks_pass(self):
        fc = FunctionalCheck("always_pass", "desc", lambda *a, **kw: True)
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
            functional_checks=[fc],
        )
        results = run_functional([gb])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].functional.passed, 1)
        self.assertEqual(results[0].functional.failed, 0)

    def test_some_checks_fail(self):
        fc_pass = FunctionalCheck("pass", "desc", lambda *a, **kw: True)
        fc_fail = FunctionalCheck("fail", "desc", lambda *a, **kw: False)
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
            functional_checks=[fc_pass, fc_fail],
        )
        results = run_functional([gb])
        self.assertEqual(results[0].functional.passed, 1)
        self.assertEqual(results[0].functional.failed, 1)

    def test_filter_by_module_path(self):
        fc = FunctionalCheck("c1", "desc", lambda *a, **kw: True)
        gb1 = GeneratorBenchmark(
            generator_cls=FakeGenerator, name="Gen1",
            module_path="tabular_data.noise", default_params={},
            scenarios=[self.scenario], functional_checks=[fc],
        )
        gb2 = GeneratorBenchmark(
            generator_cls=FakeGenerator, name="Gen2",
            module_path="time_series.outliers", default_params={},
            scenarios=[self.scenario], functional_checks=[fc],
        )
        results = run_functional([gb1, gb2], filter_path="tabular_data")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].generator, "tabular_data.noise.Gen1")

    def test_empty_registry(self):
        results = run_functional([])
        self.assertEqual(len(results), 0)

    def test_generator_raises_not_implemented(self):
        class BrokenGen:
            def __init__(self, random_generator=None):
                pass
            def generate(self, X, y, **params):
                raise NotImplementedError("not supported")

        fc = FunctionalCheck("c1", "desc", lambda *a, **kw: True)
        gb = GeneratorBenchmark(
            generator_cls=BrokenGen, name="Broken",
            module_path="test.broken", default_params={},
            scenarios=[self.scenario], functional_checks=[fc],
        )
        results = run_functional([gb])
        self.assertEqual(len(results), 0)  # skipped


class TestRunPerformance(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=lambda rng: (np.zeros((100, 2)), None),
        )

    def test_measures_time_and_memory(self):
        fc = FunctionalCheck("c1", "desc", lambda *a, **kw: True)
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
            functional_checks=[fc],
        )
        from benchmarks.runner import run_performance
        results = run_performance([gb], iterations=3)
        self.assertEqual(len(results), 1)
        perf = results[0].performance
        self.assertIn("time_ms", perf)
        self.assertIn("memory_mb", perf)
        self.assertEqual(perf["time_ms"].iterations, 3)
        self.assertGreaterEqual(perf["time_ms"].min, 0)

    def test_run_all_combines_both(self):
        fc = FunctionalCheck("c1", "desc", lambda *a, **kw: True)
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
            functional_checks=[fc],
        )
        from benchmarks.runner import run_all
        results = run_all([gb], iterations=2)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].functional)
        self.assertIsNotNone(results[0].performance)