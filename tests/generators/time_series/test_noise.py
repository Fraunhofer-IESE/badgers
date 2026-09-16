import numpy as np

from badgers.generators.time_series.noise import GlobalGaussianNoiseGenerator, LocalGaussianNoiseGenerator


def test_local_gaussian_noise__generates(time_series_sine):
    """LocalGaussianNoiseGenerator preserves shape and modifies data within patterns."""
    X, _ = time_series_sine
    generator = LocalGaussianNoiseGenerator()
    Xt, _ = generator.generate(X, None, noise_std=0.5)

    # Shape preserved
    assert Xt.shape == X.shape

    # At least one pattern was generated
    assert len(generator.patterns_indices_) > 0

    # Data within patterns differs from original
    for start, end in generator.patterns_indices_:
        assert not np.allclose(Xt[start:end], X[start:end])

    # Data outside patterns is unchanged
    first_start = generator.patterns_indices_[0][0]
    if first_start > 0:
        assert np.allclose(Xt[:first_start], X[:first_start])


def test_global_gaussian_noise__generates(time_series_sine):
    """GlobalGaussianNoiseGenerator preserves shape and increases variance."""
    X, _ = time_series_sine
    generator = GlobalGaussianNoiseGenerator()
    Xt, _ = generator.generate(X, None, noise_std=0.5)

    # Shape preserved
    assert Xt.shape == X.shape

    # Data is modified (not identical)
    assert not np.allclose(Xt, X)

    # Variance increased
    assert np.var(Xt) > np.var(X)
