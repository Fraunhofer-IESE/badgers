"""Benchmark registrations for tabular data outlier generators."""
from numpy.random import default_rng

from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import (
    SCENARIO_SMALL_BLOBS,
    SCENARIO_MEDIUM_BLOBS,
    SCENARIO_LARGE_BLOBS,
)
from badgers.core.selectors import CountSelector
from badgers.core.sampling import (
    HypersphereSampler,
    IndependentHistogramSampler,
    JointHistogramSampler,
    KernelDensitySampler,
    TreeCellSampler,
    UniformOutOfDistributionSampler,
    ZScoreSampler,
)
from badgers.generators.tabular_data.outliers.instance_sampling import UniformInstanceAttributeSampling
from badgers.generators.tabular_data.outliers.sampling import OutlierSamplingGenerator


def _count_selector(count):
    """Callable selector: exactly ``count`` seed rows, every column perturbed.

    Reproduces the pre-refactor default of the sampling generators, whose
    implicit selector was ``CountSelector(range(X.shape[1]), count, ...)``;
    the column scope is resolved at call time so the same preset works for
    every scenario dimensionality.
    """
    def selector(X, targets=None):
        sel = CountSelector(targets, count=count, rng=default_rng(0))
        return sel(X, targets)
    return selector


register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="UniformHyperCube",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": UniformOutOfDistributionSampler(expansion=0.0),
    },
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="UniformHyperCube (large)",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(1000),
        "out_of_distribution_sampler": UniformOutOfDistributionSampler(expansion=0.0),
    },
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
    generator_cls=OutlierSamplingGenerator,
    name="IndependentHistograms",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": IndependentHistogramSampler(bins=10),
    },
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="IndependentHistograms (large)",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(1000),
        "out_of_distribution_sampler": IndependentHistogramSampler(bins=10),
    },
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="HistogramSampling",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": JointHistogramSampler(bins=10,
                                                             threshold_low_density=0.1),
    },
    scenarios=[SCENARIO_SMALL_BLOBS],  # joint histogram supports at most 5 columns
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="ZScoreSampling",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": ZScoreSampler(scale=1.0),
    },
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="ZScoreSampling (large)",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(1000),
        "out_of_distribution_sampler": ZScoreSampler(scale=1.0),
    },
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="LowDensitySampling",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": KernelDensitySampler(threshold_low_density=0.1),
    },
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="LowDensitySampling (large)",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(1000),
        "out_of_distribution_sampler": KernelDensitySampler(threshold_low_density=0.1),
    },
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="TreeCellSampling",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": TreeCellSampler(threshold_low_density=0.1),
    },
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="TreeCellSampling (large)",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(1000),
        "out_of_distribution_sampler": TreeCellSampler(threshold_low_density=0.1),
    },
    scenarios=[SCENARIO_LARGE_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="HypersphereSampling",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(10),
        "out_of_distribution_sampler": HypersphereSampler(scale=1.0),
    },
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=OutlierSamplingGenerator,
    name="HypersphereSampling (large)",
    module_path="tabular_data.outliers",
    default_params={
        "selector": _count_selector(1000),
        "out_of_distribution_sampler": HypersphereSampler(scale=1.0),
    },
    scenarios=[SCENARIO_LARGE_BLOBS],
))
