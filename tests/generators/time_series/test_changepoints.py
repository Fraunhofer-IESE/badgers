import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.generators.time_series.changepoints import RandomChangeInMeanGenerator


def test_random_change_in_mean__modifies_data():
    """RandomChangeInMeanGenerator modifies the input data."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(100), columns=['dimension_0'], dtype=float)

    Xt, _ = generator.generate(X.copy(), None, n_changepoints=10, min_change=-5, max_change=5)
    assert any(X != Xt)


def test_random_change_in_mean__preserves_shape():
    """Output has same shape as input."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(100), columns=['dimension_0'], dtype=float)

    Xt, _ = generator.generate(X.copy(), None, n_changepoints=5, min_change=-2, max_change=2)
    assert Xt.shape == X.shape


def test_random_change_in_mean__zero_changepoints_preserves_data():
    """With n_changepoints=0, data should be unchanged."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.random.randn(100), columns=['dimension_0'], dtype=float)

    Xt, _ = generator.generate(X.copy(), None, n_changepoints=0, min_change=-5, max_change=5)
    assert np.allclose(Xt, X)


def test_random_change_in_mean__creates_distinct_segments():
    """With n_changepoints > 0, there should be at least 2 distinct mean segments."""
    rng = default_rng(seed=0)
    generator = RandomChangeInMeanGenerator(random_generator=rng)
    X = pd.DataFrame(data=np.zeros(200), columns=['dimension_0'], dtype=float)

    Xt, _ = generator.generate(X.copy(), None, n_changepoints=5, min_change=-5, max_change=5)
    # Count unique values — should have at least 2 distinct segments
    unique_values = len(set(np.round(Xt.flatten(), decimals=4)))
    assert unique_values >= 2
