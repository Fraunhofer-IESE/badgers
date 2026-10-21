import unittest
import warnings

import numpy as np
from benchmarks.models import Scenario, GeneratorBenchmark
from benchmarks.runner import run_performance, run_all


class FakeGenerator:
    def __init__(self, random_generator=None):
        self.random_generator = random_generator

    def generate(self, X, **params):
        return X


class TestRunPerformance(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            name="test_scenario",
            data_type="tabular",
            factory=lambda rng: np.zeros((100, 2)),
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

    def test_runner_passes_single_object_and_default_params(self):
        calls = []

        class RecordingGenerator(FakeGenerator):
            def generate(self, X, **params):
                calls.append((X, dict(params)))
                return X

        gb = GeneratorBenchmark(
            generator_cls=RecordingGenerator,
            name="RecGen",
            module_path="test.module",
            default_params={"noise": 0.1},
            scenarios=[self.scenario],
        )
        results = run_performance([gb], iterations=1)
        self.assertEqual(len(results), 1)
        # one warm-up call + one measured call
        self.assertEqual(len(calls), 2)
        warm_X, warm_params = calls[0]
        gen_X, gen_params = calls[1]
        for X, params in [(warm_X, warm_params), (gen_X, gen_params)]:
            self.assertEqual(X.shape, (100, 2))
            self.assertEqual(params, {"noise": 0.1})
        # warm-up and timed calls see equal content (same seeded factory)
        np.testing.assert_array_equal(warm_X, gen_X)

    def test_unsupported_generator_scenario_warns_not_implemented(self):
        class UnsupportedGenerator(FakeGenerator):
            def generate(self, X, **params):
                raise NotImplementedError("scenario not supported")

        gb = GeneratorBenchmark(
            generator_cls=UnsupportedGenerator,
            name="UnsupportedGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
        )
        with self.assertWarns(UserWarning):
            results = run_performance([gb], iterations=2)
        self.assertEqual(results, [])

    def test_generic_generator_exception_is_swallowed(self):
        class BrokenGenerator(FakeGenerator):
            def generate(self, X, **params):
                raise ValueError("boom")

        gb = GeneratorBenchmark(
            generator_cls=BrokenGenerator,
            name="BrokenGen",
            module_path="test.module",
            default_params={},
            scenarios=[self.scenario],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_performance([gb], iterations=2)
        self.assertEqual(results, [])

    def test_factory_failure_warns_and_skips_scenario(self):
        def failing_factory(rng):
            raise RuntimeError("no data")

        scenario = Scenario(
            name="failing_scenario",
            data_type="tabular",
            factory=failing_factory,
        )
        gb = GeneratorBenchmark(
            generator_cls=FakeGenerator,
            name="FakeGen",
            module_path="test.module",
            default_params={},
            scenarios=[scenario],
        )
        with self.assertWarns(UserWarning):
            results = run_performance([gb], iterations=2)
        self.assertEqual(results, [])