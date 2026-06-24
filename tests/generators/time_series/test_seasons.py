import numpy as np
from numpy.random import default_rng

from badgers.generators.time_series.seasons import GlobalAdditiveSinusoidalSeasonGenerator


def test_global_additive_sinusoidal_season__matches_expected():
    """GlobalAdditiveSinusoidalSeasonGenerator produces expected sinusoidal output."""
    rng = default_rng(seed=0)
    X = np.zeros(shape=100)
    period = 10
    generator = GlobalAdditiveSinusoidalSeasonGenerator(random_generator=rng)
    Xt, yt = generator.generate(X=X, y=None, period=period)

    t = np.arange(100)
    season = np.sin(t * 2 * np.pi / period)
    Xt_expected = X + season

    assert Xt_expected.reshape(-1, 1).tolist() == Xt.values.tolist()
    assert yt is None
