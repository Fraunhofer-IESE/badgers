import numpy as np

from badgers.generators.time_series.outliers import LocalZScoreGenerator, RandomZerosGenerator


def test_random_zeros__correct_shape_and_count(time_series_sine):
    """RandomZerosGenerator produces correct shape and outlier count."""
    n_outliers = 10
    X = time_series_sine
    generator = RandomZerosGenerator()
    Xt = generator.generate(X=X, n_outliers=n_outliers)
    assert Xt.shape == X.shape
    assert len(generator.outliers_indices_) == n_outliers


def test_local_zscore__correct_shape_and_count(time_series_sine):
    """LocalZScoreGenerator produces correct shape, count, and no NaN."""
    n_outliers = 10
    X = time_series_sine
    generator = LocalZScoreGenerator()
    Xt = generator.generate(X=X, n_outliers=n_outliers)
    assert Xt.shape == X.shape
    assert len(generator.outliers_indices_) == n_outliers
    assert not np.isnan(Xt).any()
