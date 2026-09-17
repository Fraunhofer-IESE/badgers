"""Benchmark registrations for tabular data outlier generators."""
import numpy as np

from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import (
    SCENARIO_SMALL_BLOBS,
    SCENARIO_MEDIUM_BLOBS,
    SCENARIO_LARGE_BLOBS,
)

from badgers.generators.tabular_data.outliers.distribution_sampling import (
    HyperCubeSampling,
    HypersphereSamplingGenerator,
    ZScoreSamplingGenerator,
)
from badgers.generators.tabular_data.outliers.instance_sampling import UniformInstanceAttributeSampling
from badgers.generators.tabular_data.outliers.low_density_sampling import (
    HistogramSamplingGenerator,
    IndependentHistogramsGenerator,
    LowDensitySamplingGenerator,
)


register(GeneratorBenchmark(
    generator_cls=HyperCubeSampling,
    name="HyperCubeSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=HyperCubeSampling,
    name="HyperCubeSampling (large)",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 1000},
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=UniformInstanceAttributeSampling,
    name="UniformInstanceAttributeSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=UniformInstanceAttributeSampling,
    name="UniformInstanceAttributeSampling (large)",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 1000},
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=IndependentHistogramsGenerator,
    name="IndependentHistogramsGenerator",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "bins": 10},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=IndependentHistogramsGenerator,
    name="IndependentHistogramsGenerator (large)",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 1000, "bins": 10},
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=HistogramSamplingGenerator,
    name="HistogramSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "bins": 10, "threshold_low_density": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS],  # only 2D — raises on >5D
))

register(GeneratorBenchmark(
    generator_cls=ZScoreSamplingGenerator,
    name="ZScoreSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "scale": 1.0},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=ZScoreSamplingGenerator,
    name="ZScoreSampling (large)",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 1000, "scale": 1.0},
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=LowDensitySamplingGenerator,
    name="LowDensitySampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "threshold_low_density": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=LowDensitySamplingGenerator,
    name="LowDensitySampling (large)",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 1000, "threshold_low_density": 0.1},
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=HypersphereSamplingGenerator,
    name="HypersphereSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "scale": 1.0},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=HypersphereSamplingGenerator,
    name="HypersphereSampling (large)",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 1000, "scale": 1.0},
    scenarios=[SCENARIO_LARGE_BLOBS],
))
