import numpy as np

from badgers.generators.time_series.missingness import MissingAtRandomGenerator


def test_missing_at_random__correct_shape_and_count(time_series_sine):
    """MissingAtRandomGenerator produces correct shape, missing count, and NaN count."""
    n_missing = 10
    X, _ = time_series_sine
    generator = MissingAtRandomGenerator()
    Xt, _ = generator.generate(X=X, y=None, n_missing=n_missing)
    assert Xt.shape == X.shape
    assert len(generator.missing_indices_) == n_missing
    assert np.isnan(Xt).sum() == n_missing
