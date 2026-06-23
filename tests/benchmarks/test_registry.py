import pathlib
import unittest
from unittest.mock import patch, MagicMock
from benchmarks.models import Scenario, FunctionalCheck, GeneratorBenchmark
from benchmarks.registry import register, get_registry, discover, _registry


class TestRegistry(unittest.TestCase):
    def setUp(self):
        _registry.clear()

    def tearDown(self):
        _registry.clear()

    def _make_benchmark(self, name="TestGen", module_path="test.module"):
        s = Scenario("s1", "tabular", lambda rng: (None, None))
        fc = FunctionalCheck("c1", "desc", lambda *a, **kw: True)
        return GeneratorBenchmark(
            generator_cls=type("FakeGen", (), {}),
            name=name,
            module_path=module_path,
            default_params={},
            scenarios=[s],
            functional_checks=[fc],
        )

    def test_register_adds_to_registry(self):
        gb = self._make_benchmark()
        register(gb)
        self.assertEqual(len(_registry), 1)
        self.assertIs(_registry[0], gb)

    def test_register_multiple(self):
        gb1 = self._make_benchmark(name="Gen1")
        gb2 = self._make_benchmark(name="Gen2")
        register(gb1)
        register(gb2)
        self.assertEqual(len(_registry), 2)

    def test_get_registry_returns_copy(self):
        gb = self._make_benchmark()
        register(gb)
        result = get_registry()
        self.assertEqual(len(result), 1)
        result.clear()
        self.assertEqual(len(_registry), 1)  # original unchanged

    def _make_mock_path(self, stem):
        """Create a mock Path that supports suffix, stem, and relative_to."""
        mock_path = MagicMock()
        mock_path.suffix = ".py"
        mock_path.stem = stem
        mock_path.with_suffix.return_value = mock_path
        mock_path.relative_to.return_value = pathlib.PurePosixPath(f"generators/{stem}.py")
        return mock_path

    def test_discover_imports_benchmark_modules(self):
        with patch("benchmarks.registry.importlib.import_module") as mock_import:
            with patch.object(pathlib.Path, "exists", return_value=True):
                with patch.object(pathlib.Path, "rglob") as mock_rglob:
                    mock_rglob.return_value = [self._make_mock_path("_noise")]
                    discover()
                    self.assertTrue(mock_import.called)

    def test_discover_skips_init_files(self):
        with patch("benchmarks.registry.importlib.import_module") as mock_import:
            with patch.object(pathlib.Path, "exists", return_value=True):
                with patch.object(pathlib.Path, "rglob") as mock_rglob:
                    mock_rglob.return_value = [
                        self._make_mock_path("__init__"),
                    ]
                    discover()
                    mock_import.assert_not_called()