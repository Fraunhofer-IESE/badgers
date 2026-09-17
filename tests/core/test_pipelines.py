import numpy as np
import pytest

from numpy.random import default_rng
from sklearn.datasets import make_blobs

from badgers.core.pipeline import Pipeline
from badgers.generators.tabular_data.imbalance import RandomSamplingClassesGenerator
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator


def test_pipeline__preserves_shape():
    """Pipeline output should have same number of rows as input."""
    random_generator = default_rng(0)
    X, y = make_blobs(centers=3, random_state=0)
    generators = {
        'imbalance': RandomSamplingClassesGenerator(random_generator=random_generator),
        'noise': GaussianNoiseGenerator(random_generator=random_generator)
    }
    pipeline = Pipeline(generators=generators)
    params = {
        'imbalance': {'proportion_classes': {0: 0.5, 1: 0.25, 2: 0.25}},
        'noise': {'noise_std': 0.5}
    }
    Xt, yt = pipeline.generate(X=X, y=y, params=params)
    assert len(Xt) == len(X)
    assert len(yt) == len(y)


def test_pipeline__modifies_data():
    """Pipeline with noise should modify the data."""
    random_generator = default_rng(0)
    X, y = make_blobs(centers=3, random_state=0)
    generators = {
        'noise': GaussianNoiseGenerator(random_generator=random_generator)
    }
    pipeline = Pipeline(generators=generators)
    params = {'noise': {'noise_std': 1.0}}
    Xt, yt = pipeline.generate(X=X.copy(), y=y, params=params)
    assert not np.allclose(Xt, X)


def test_pipeline__empty_generators_preserves_data():
    """Pipeline with no generators should return data unchanged."""
    X, y = make_blobs(centers=3, random_state=0)
    pipeline = Pipeline(generators={})
    Xt, yt = pipeline.generate(X=X.copy(), y=y, params={})
    assert np.allclose(Xt, X)
    assert np.array_equal(yt, y)
