import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba
from badgers.generators.tabular_data.imbalance import RandomSamplingFeaturesGenerator, \
    RandomSamplingClassesGenerator, RandomSamplingTargetsGenerator


def test_random_sampling_classes__preserves_shape_and_columns(tabular_data_labeled):
    """RandomSamplingClassesGenerator preserves shape and DataFrame columns."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1

    proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))

    Xt, yt = generator.generate(X.copy(), y, proportion_classes=proportion_classes)
    assert Xt.shape[1] == n_features
    assert Xt.shape[0] == len(yt)


def test_random_sampling_features__preserves_shape_and_columns(tabular_data_labeled):
    """RandomSamplingFeaturesGenerator preserves shape and DataFrame columns."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1

    def proba_func(X):
        feature = X[:, 0]
        return normalize_proba(
            (np.max(feature) - feature) / (np.max(feature) - np.min(feature))
        )

    generator = RandomSamplingFeaturesGenerator()
    Xt, yt = generator.generate(X.copy(), y, sampling_proba_func=proba_func)
    assert Xt.shape[1] == n_features
    assert Xt.shape[0] == len(yt)


def test_random_sampling_targets__preserves_shape_and_columns(tabular_data_labeled):
    """RandomSamplingTargetsGenerator preserves shape and DataFrame columns."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1

    def proba_func(y):
        return normalize_proba(
            (np.max(y) - y) / (np.max(y) - np.min(y))
        )

    generator = RandomSamplingTargetsGenerator()
    Xt, yt = generator.generate(X.copy(), y, sampling_proba_func=proba_func)
    assert Xt.shape[1] == n_features
    assert Xt.shape[0] == len(yt)
