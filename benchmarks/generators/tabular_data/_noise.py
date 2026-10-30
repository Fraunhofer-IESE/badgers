"""Benchmark registrations for tabular noise generators."""
import scipy.stats
from numpy.random import default_rng

from benchmarks.models import GeneratorBenchmark
from benchmarks.registry import register
from benchmarks.scenarios.tabular import SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS, SCENARIO_SMALL_BLOBS_CLASSES
from badgers.core.selectors import ClassSelector
from badgers.generators.tabular_data.noise import AdditiveNoiseGenerator, GaussianNoiseClassesGenerator

_CLASS_PROPORTIONS = {c: 1.0 for c in range(5)}

register(GeneratorBenchmark(
    generator_cls=AdditiveNoiseGenerator,
    name="AdditiveNoise",
    module_path="tabular_data.noise",
    default_params={"distribution": scipy.stats.norm(scale=0.5)},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=GaussianNoiseClassesGenerator,
    name="GaussianNoiseClasses",
    module_path="tabular_data.noise",
    default_params={
        "selector": ClassSelector([0], class_column=2, proportions=_CLASS_PROPORTIONS, rng=default_rng(0)),
        "noise_std_per_class": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
    },
    scenarios=[SCENARIO_SMALL_BLOBS_CLASSES],
))
