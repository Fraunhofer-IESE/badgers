import unittest
import json
import tempfile
import pathlib
from unittest.mock import patch, MagicMock
from benchmarks.cli import build_parser, cmd_run


class TestCLIParsing(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_run_defaults(self):
        args = self.parser.parse_args(["run"])
        self.assertEqual(args.type, "all")
        self.assertIsNone(args.generators)

    def test_run_functional_only(self):
        args = self.parser.parse_args(["run", "--type", "functional"])
        self.assertEqual(args.type, "functional")

    def test_run_performance_only(self):
        args = self.parser.parse_args(["run", "--type", "performance"])
        self.assertEqual(args.type, "performance")

    def test_run_with_filter(self):
        args = self.parser.parse_args(["run", "--generators", "tabular_data"])
        self.assertEqual(args.generators, "tabular_data")

    def test_baseline_save(self):
        args = self.parser.parse_args(["baseline", "save"])
        self.assertEqual(args.name, "latest")

    def test_baseline_save_named(self):
        args = self.parser.parse_args(["baseline", "save", "--name", "v1.0"])
        self.assertEqual(args.name, "v1.0")

    def test_baseline_list(self):
        args = self.parser.parse_args(["baseline", "list"])
        self.assertEqual(args.command, "list")

    def test_compare_default(self):
        args = self.parser.parse_args(["compare"])
        self.assertIsNone(args.baseline)
        self.assertIsNone(args.target)

    def test_compare_with_baseline(self):
        args = self.parser.parse_args(["compare", "--baseline", "v0.0.13"])
        self.assertEqual(args.baseline, "v0.0.13")


class TestCmdRun(unittest.TestCase):
    @patch("benchmarks.cli.discover")
    @patch("benchmarks.cli.run_all")
    def test_cmd_run_saves_results(self, mock_run_all, mock_discover):
        from benchmarks.models import BenchmarkResult
        mock_discover.return_value = [MagicMock()]  # non-empty so it doesn't exit
        mock_run_all.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("benchmarks.cli.RESULTS_DIR", pathlib.Path(tmpdir)):
                args = MagicMock()
                args.type = "all"
                args.generators = None
                args.iterations = 5
                args.timeout = 60
                cmd_run(args)

                json_files = list(pathlib.Path(tmpdir).glob("*.json"))
                self.assertEqual(len(json_files), 1)