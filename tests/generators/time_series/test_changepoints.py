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
