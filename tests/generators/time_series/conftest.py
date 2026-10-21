import numpy as np
import pytest


@pytest.fixture
def time_series_sine():
    """200-point sine wave."""
    t = np.linspace(0, 4 * np.pi, 200)
    X = np.sin(t).reshape(-1, 1)
    return X


@pytest.fixture
def time_series_walk(rng):
    """200-point random walk."""
    steps = rng.normal(0, 1, size=(200, 1))
    X = np.cumsum(steps, axis=0)
    return X
