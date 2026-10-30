import numpy as np

from badgers.generators.time_series.missingness import MissingCompletelyAtRandomGenerator


def test_missing_completely_at_random__correct_shape_and_count(time_series_sine):
    """MissingCompletelyAtRandomGenerator produces correct shape, missing count, and NaN count."""
    n_missing = 10
    X = time_series_sine
    generator = MissingCompletelyAtRandomGenerator(random_generator=np.random.default_rng(0))
    Xt = generator.generate(X=X, n_missing=n_missing)
    assert Xt.shape == X.shape
    assert len(generator.missing_indices_) == n_missing
    assert np.isnan(Xt).sum() == n_missing
