"""Predefined scenarios for tabular data generators."""
import numpy as np
from sklearn.datasets import make_blobs
from benchmarks.models import Scenario


def _make_blobs_factory(n_samples, n_features):
    def factory(rng):
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=5,
            random_state=42,
        )
        return X, y
    return factory


SCENARIO_SMALL_BLOBS = Scenario(
    name="small_blobs_2d",
    data_type="tabular",
    factory=_make_blobs_factory(100, 2),
    tags=["small", "2D", "classification"],
)

SCENARIO_MEDIUM_BLOBS = Scenario(
    name="medium_blobs_5d",
    data_type="tabular",
    factory=_make_blobs_factory(1000, 5),
    tags=["medium", "5D", "classification"],
)

SCENARIO_LARGE_BLOBS = Scenario(
    name="large_blobs_10d",
    data_type="tabular",
    factory=_make_blobs_factory(5000, 10),
    tags=["large", "10D", "classification"],
)