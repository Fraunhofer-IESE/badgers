import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.generators.time_series.changepoints import RandomChangeInMeanGenerator


def test_random_change_in_mean__modifies_data():
    """RandomChangeInMeanGenerator modifies the input data."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(100), columns=['dimension_0'], dtype=float)

    Xt = generator.generate(X.copy(), n_changepoints=10, min_change=-5, max_change=5)
    assert any(X != Xt)


def test_random_change_in_mean__preserves_shape():
    """Output has same shape as input."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(100), columns=['dimension_0'], dtype=float)

    Xt = generator.generate(X.copy(), n_changepoints=5, min_change=-2, max_change=2)
    assert Xt.shape == X.shape


def test_random_change_in_mean__zero_changepoints_preserves_data():
    """With n_changepoints=0, data should be unchanged."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.random.randn(100), columns=['dimension_0'], dtype=float)

    Xt = generator.generate(X.copy(), n_changepoints=0, min_change=-5, max_change=5)
    assert np.allclose(Xt, X)


def test_random_change_in_mean__creates_distinct_segments():
    """With n_changepoints > 0, there should be at least 2 distinct mean segments."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(200), columns=['dimension_0'], dtype=float)

    Xt = generator.generate(X.copy(), n_changepoints=5, min_change=-5, max_change=5)
    # Count unique values — should have at least 2 distinct segments
    unique_values = len(set(np.round(Xt.flatten(), decimals=4)))
    assert unique_values >= 2


def test_random_change_in_mean__distinct_indices():
    """Changepoint indices are distinct (no position is drawn twice)."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = np.zeros((200, 1))

    generator.generate(X.copy(), n_changepoints=40, min_change=-5, max_change=5)

    indices = [i for i, _ in generator.changepoints]
    assert len(indices) == len(set(indices)) == 40


def test_random_change_in_mean__sorted_reported_jumps_match():
    """Changepoints are reported sorted by index and each reported change is the
    realized level jump at that position."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = np.zeros((200, 1))

    Xt = generator.generate(X.copy(), n_changepoints=10, min_change=-5, max_change=5)

    indices = [i for i, _ in generator.changepoints]
    assert indices == sorted(indices)
    level = np.diff(Xt[:, 0], prepend=Xt[0, 0])
    for i, change in generator.changepoints:
        np.testing.assert_allclose(level[i], change)
