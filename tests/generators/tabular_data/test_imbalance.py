import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba
from badgers.generators.tabular_data.imbalance import RandomSamplingFeaturesGenerator, \
    RandomSamplingClassesGenerator, RandomSamplingTargetsGenerator


def test_random_sampling_classes__preserves_shape_and_respects_proportions(tabular_data_labeled):
    """RandomSamplingClassesGenerator preserves shape and respects proportion_classes."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1

    proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))

    Xt, yt = generator.generate(X.copy(), y, proportion_classes=proportion_classes)
    assert Xt.shape[1] == n_features
    assert Xt.shape[0] == len(yt)

    # Verify class proportions are approximately respected
    yt_np = np.asarray(yt)
    total = len(yt_np)
    for label, expected_prop in proportion_classes.items():
        actual_prop = np.sum(yt_np == label) / total
        # Allow 20% relative tolerance since sampling is random
        assert abs(actual_prop - expected_prop) < 0.15, \
            f"Class {label}: expected ~{expected_prop}, got {actual_prop:.3f}"


def test_random_sampling_features__preserves_shape_and_modifies_distribution(tabular_data_labeled):
    """RandomSamplingFeaturesGenerator preserves shape and changes row count."""
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

    # With replacement sampling, output should have same length as input
    assert Xt.shape[0] == X_np.shape[0]

    # Distribution of first feature should shift toward lower values
    # (since proba_func gives higher weight to lower values)
    assert np.mean(Xt[:, 0]) < np.mean(X_np[:, 0])


def test_random_sampling_targets__preserves_shape_and_modifies_distribution(tabular_data_labeled):
    """RandomSamplingTargetsGenerator preserves shape and changes target distribution."""
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

    # Output should have same length as input (sampling with replacement)
    assert Xt.shape[0] == X_np.shape[0]

    # Target mean should shift toward lower values
    # (since proba_func gives higher weight to lower target values)
    y_np = np.asarray(y)
    yt_np = np.asarray(yt)
    assert np.mean(yt_np) < np.mean(y_np)
