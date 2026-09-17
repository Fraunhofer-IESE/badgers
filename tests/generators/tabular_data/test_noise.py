import numpy as np
from badgers.generators.tabular_data.noise import (
    GaussianNoiseGenerator, GaussianNoiseClassesGenerator,
)


def test_gaussian_noise__preserves_shape(tabular_data):
    X, y = tabular_data
    generator = GaussianNoiseGenerator()
    Xt, _ = generator.generate(X.copy(), y=None, noise_std=1)
    assert len(X) == len(Xt)


def test_gaussian_noise__increases_variance(tabular_data):
    X, y = tabular_data
    generator = GaussianNoiseGenerator()
    Xt, _ = generator.generate(X.copy(), y=None, noise_std=1)
    assert (np.var(Xt, axis=0) > np.var(X, axis=0)).all()


def test_gaussian_noise__zero_std_preserves_data(tabular_data):
    """With noise_std=0, the data should be unchanged."""
    X, y = tabular_data
    X_copy = X.copy()
    generator = GaussianNoiseGenerator()
    Xt, _ = generator.generate(X_copy, y=None, noise_std=0)
    assert np.allclose(Xt, X_copy)


def test_gaussian_noise_classes__preserves_shape(tabular_data_labeled):
    X, y = tabular_data_labeled
    noise_std_per_class = {label: 0.1 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator()
    Xt, yt = generator.generate(X.copy(), y, noise_std_per_class=noise_std_per_class)
    assert len(X) == len(Xt)
    assert len(y) == len(yt)


def test_gaussian_noise_classes__modifies_data(tabular_data_labeled):
    """With non-zero noise_std, data should be modified."""
    X, y = tabular_data_labeled
    noise_std_per_class = {label: 0.5 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator()
    Xt, yt = generator.generate(X.copy(), y, noise_std_per_class=noise_std_per_class)
    assert not np.allclose(Xt, X)
