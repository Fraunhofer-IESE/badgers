"""Benchmark registrations for tabular data outlier generators."""
import numpy as np

from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import (
    SCENARIO_SMALL_BLOBS,
    SCENARIO_MEDIUM_BLOBS,
    SCENARIO_LARGE_BLOBS,
)

from badgers.generators.tabular_data.outliers.distribution_sampling import HyperCubeSampling
from badgers.generators.tabular_data.outliers.instance_sampling import UniformInstanceAttributeSampling
from badgers.generators.tabular_data.outliers.low_density_sampling import IndependentHistogramsGenerator


register(GeneratorBenchmark(
    generator_cls=HyperCubeSampling,
    name="HyperCubeSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=UniformInstanceAttributeSampling,
    name="UniformInstanceAttributeSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=IndependentHistogramsGenerator,
    name="IndependentHistogramsGenerator",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "bins": 10},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
