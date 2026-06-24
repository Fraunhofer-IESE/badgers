"""Registry for GeneratorBenchmark instances with auto-discovery."""
import importlib
import pathlib
import warnings
from typing import List

from benchmarks.models import GeneratorBenchmark

_registry: List[GeneratorBenchmark] = []


def register(benchmark: GeneratorBenchmark) -> None:
    """Register a GeneratorBenchmark instance."""
    _registry.append(benchmark)


def get_registry() -> List[GeneratorBenchmark]:
    """Return a copy of the current registry."""
    return list(_registry)


def discover() -> List[GeneratorBenchmark]:
    """Auto-discover benchmark registrations by importing _*.py modules
    from benchmarks/generators/."""
    generators_dir = pathlib.Path(__file__).parent / "generators"
    if not generators_dir.exists():
        return list(_registry)

    for py_file in generators_dir.rglob("_*.py"):
        if py_file.stem == "__init__":
            continue
        rel_path = py_file.relative_to(pathlib.Path(__file__).parent)
        module_path = "benchmarks." + str(rel_path.with_suffix("")).replace("\\", ".").replace("/", ".")
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            warnings.warn(f"Could not import benchmark module {module_path}: {e}")

    return list(_registry)