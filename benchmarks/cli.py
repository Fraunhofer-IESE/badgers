"""CLI for the benchmark framework."""
import argparse
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import List

from benchmarks.models import RunMeta, BenchmarkResult
from benchmarks.registry import discover
from benchmarks.runner import run_functional, run_performance, run_all

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
BASELINES_DIR = pathlib.Path(__file__).parent / "baselines"


def _get_git_info() -> tuple:
    """Get current git commit and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    return commit, branch


def _make_meta() -> RunMeta:
    """Create RunMeta with current environment info."""
    commit, branch = _get_git_info()
    return RunMeta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        git_branch=branch,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def _serialize_results(results: List[BenchmarkResult], meta: RunMeta) -> dict:
    """Convert results to JSON-serializable dict."""
    return {
        "meta": {
            "timestamp": meta.timestamp,
            "git_commit": meta.git_commit,
            "git_branch": meta.git_branch,
            "python_version": meta.python_version,
            "platform": meta.platform,
        },
        "results": [
            {
                "generator": r.generator,
                "scenario": r.scenario,
                "params": r.params,
                "functional": {
                    "passed": r.functional.passed,
                    "failed": r.functional.failed,
                    "checks": r.functional.checks,
                } if r.functional else None,
                "performance": {
                    key: {
                        "min": ps.min, "max": ps.max, "mean": ps.mean,
                        "median": ps.median, "stddev": ps.stddev,
                        "iterations": ps.iterations,
                    }
                    for key, ps in r.performance.items()
                } if r.performance else None,
            }
            for r in results
        ],
    }


def cmd_run(args):
    """Execute the 'run' subcommand."""
    benchmarks = discover()
    if not benchmarks:
        print("No benchmarks discovered. Check that benchmark registration files exist.")
        sys.exit(1)

    if args.type == "functional":
        results = run_functional(benchmarks, args.generators)
    elif args.type == "performance":
        results = run_performance(benchmarks, args.generators, args.iterations, args.timeout)
    else:
        results = run_all(benchmarks, args.generators, args.iterations, args.timeout)

    meta = _make_meta()
    data = _serialize_results(results, meta)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{meta.git_branch}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {filepath}")
    print(f"  Generators: {len(set(r.generator for r in results))}")
    print(f"  Scenarios:  {len(results)}")


def cmd_baseline(args):
    """Execute the 'baseline' subcommand."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "save":
        result_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_files:
            print("No result files found. Run benchmarks first with: python -m benchmarks run")
            sys.exit(1)

        latest = result_files[0]
        baseline_path = BASELINES_DIR / f"{args.name}.json"
        baseline_path.write_bytes(latest.read_bytes())
        print(f"Baseline '{args.name}' saved from {latest.name}")

    elif args.command == "list":
        baselines = sorted(BASELINES_DIR.glob("*.json"))
        if not baselines:
            print("No baselines saved yet.")
        else:
            for b in baselines:
                print(f"  {b.stem}")


def cmd_compare(args):
    """Execute the 'compare' subcommand."""
    from benchmarks.comparator import compare_results

    # Determine baseline file
    if args.baseline:
        baseline_path = BASELINES_DIR / f"{args.baseline}.json"
        if not baseline_path.exists():
            baseline_path = pathlib.Path(args.baseline)
    else:
        baseline_files = sorted(BASELINES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not baseline_files:
            print("No baseline found. Save one with: python -m benchmarks baseline save")
            sys.exit(1)
        baseline_path = baseline_files[0]

    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}")
        sys.exit(1)

    # Determine target file
    if args.target:
        target_path = pathlib.Path(args.target)
    else:
        result_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_files:
            print("No result files found. Run benchmarks first with: python -m benchmarks run")
            sys.exit(1)
        target_path = result_files[0]

    if not target_path.exists():
        print(f"Target not found: {target_path}")
        sys.exit(1)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(target_path, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    report = compare_results(baseline_data, target_data)
    print(report)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Badgers benchmark framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--type", choices=["functional", "performance", "all"],
        default="all", help="Type of benchmarks to run (default: all)"
    )
    run_parser.add_argument(
        "--generators", type=str, default=None,
        help="Filter by module path prefix (e.g. 'tabular_data.outliers')"
    )
    run_parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of performance measurement iterations (default: 5)"
    )
    run_parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Timeout per scenario in seconds (default: 60)"
    )

    # baseline
    baseline_parser = subparsers.add_parser("baseline", help="Manage baselines")
    baseline_sub = baseline_parser.add_subparsers(dest="command", required=True)
    save_parser = baseline_sub.add_parser("save", help="Save current results as baseline")
    save_parser.add_argument("--name", type=str, default="latest", help="Baseline name")
    baseline_sub.add_parser("list", help="List saved baselines")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare results against baseline")
    compare_parser.add_argument("--baseline", type=str, default=None, help="Baseline name or path")
    compare_parser.add_argument("--target", type=str, default=None, help="Target result file path")

    return parser


def main():
    """Main entry point for python -m benchmarks."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "compare":
        cmd_compare(args)