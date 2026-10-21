import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import ClassSelector, ProportionalSelector
from badgers.core.utils import normalize_proba
from badgers.generators.tabular_data.imbalance.imbalance import (
    RandomSamplingClassesGenerator,
    RandomSamplingFeaturesGenerator,
    RandomSamplingTargetsGenerator,
)


def _with_class_column(X, y):
    return np.column_stack([np.asarray(X), np.asarray(y)])


def test_random_sampling_classes__drops_proportion_per_class(tabular_data_labeled):
    """Class selectors read X and drop the requested rows from each class."""
    X, y = tabular_data_labeled
    Xc = _with_class_column(X, y)
    proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
    rng = default_rng(0)
    selector = ClassSelector(
        "cls", {"cls": Xc.shape[1] - 1}, {"cls": proportion_classes}, rng=rng
    )
    generator = RandomSamplingClassesGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    assert Xt.shape[1] == Xc.shape[1]
    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    for label in np.unique(Xc[:, -1]):
        class_size = np.sum(Xc[:, -1] == label)
        expected_dropped = min(
            int(proportion_classes[label] * Xc.shape[0]), class_size
        )
        assert np.sum(labels & (Xc[:, -1] == label)) == expected_dropped


def test_random_sampling_classes__selector_none_is_noop(tabular_data_labeled):
    """selector=None keeps every row and records no dropped rows."""
    X, y = tabular_data_labeled
    Xc = _with_class_column(X, y)
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))

    Xt = generator.generate(Xc.copy())

    np.testing.assert_array_equal(Xt, Xc)
    assert not generator.get_imbalance_labels().any()


def test_random_sampling_classes__class_column_of_X(tabular_data_labeled):
    """The class variable is selected from its mapped column in X."""
    X, y = tabular_data_labeled
    Xc = _with_class_column(X, y)
    rng = default_rng(0)
    selector = ClassSelector(
        "cls", {"cls": Xc.shape[1] - 1}, {"cls": {0: 0.5}}, rng=rng
    )
    generator = RandomSamplingClassesGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    expected = np.sum(Xc[:, -1] == 0) - min(
        int(0.5 * Xc.shape[0]), np.sum(Xc[:, -1] == 0)
    )
    assert np.sum(Xt[:, -1] == 0) == expected


def test_random_sampling_features__preserves_shape_and_modifies_distribution(
    tabular_data_labeled,
):
    """Feature sampling returns one X object with replacement."""
    X, _ = tabular_data_labeled
    X_np = np.asarray(X)

    def proba_func(X):
        feature = X[:, 0]
        return normalize_proba(
            (np.max(feature) - feature) / (np.max(feature) - np.min(feature))
        )

    generator = RandomSamplingFeaturesGenerator()
    Xt = generator.generate(X_np.copy(), sampling_proba_func=proba_func)

    assert Xt.shape == X_np.shape
    assert np.mean(Xt[:, 0]) < np.mean(X_np[:, 0])


def test_random_sampling_targets__resamples_with_replacement():
    """Low deficit probability rows are replicated more."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    selector = ProportionalSelector("f0", {"f0": 0, "f1": 1})

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X, selector=selector, replace=True)

    assert Xt.shape == X.shape
    assert np.mean(Xt[:, 0]) < np.mean(X[:, 0])


def test_random_sampling_targets__without_replacement_is_thinning(
    tabular_data_labeled,
):
    """replace=False yields distinct rows and honors requested output size."""
    X, _ = tabular_data_labeled
    X_np = np.asarray(X)
    mapping = {name: i for i, name in enumerate(["f0", "f1", "cls"][:X_np.shape[1]])}
    selector = ProportionalSelector("f0", mapping)

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X_np.copy(), selector=selector, replace=False, size=5
    )

    assert Xt.shape == (5, X_np.shape[1])
    assert len({tuple(row) for row in Xt}) == 5


def test_random_sampling_targets__boolean_mask_selector_is_exact_drop():
    """A boolean selector drops fired rows and the getter records that mask."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    keep_idx = np.arange(0, len(X), 2)
    fired = np.ones(len(X), dtype=bool)
    fired[keep_idx] = False

    class FixedMaskSelector:
        def __call__(self, X):
            return {"f0": fired}

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(), selector=FixedMaskSelector(), replace=False
    )

    np.testing.assert_array_equal(Xt, X[keep_idx])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), fired)


def test_random_sampling_targets__selector_none_raises(tabular_data_labeled):
    X, _ = tabular_data_labeled
    generator = RandomSamplingTargetsGenerator()
    with pytest.raises(ValueError, match="selector"):
        generator.generate(np.asarray(X).copy())


def test_imbalance_labels__classes_generator_raises_before_generate():
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(RuntimeError, match="generate"):
        generator.get_imbalance_labels()


def test_imbalance_labels__features_generator_tracks_dropped_rows(
    tabular_data_labeled,
):
    """False labels are sampled; True labels are absent from the output."""
    X, _ = tabular_data_labeled
    X_np = np.asarray(X)
    generator = RandomSamplingFeaturesGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X_np.copy())

    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == X_np.shape[0]
    drawn = set(map(tuple, Xt))
    for row in X_np[~labels]:
        assert tuple(row) in drawn
    for row in X_np[labels]:
        assert tuple(row) not in drawn


def test_imbalance_labels__targets_generator_fractional_selector():
    """Fractional scores label rows not selected into the generated X."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    selector = ProportionalSelector("f0", {"f0": 0, "f1": 1})
    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X, selector=selector, replace=False, size=4)

    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == len(X)
    np.testing.assert_array_equal(
        sorted(map(tuple, Xt)), sorted(map(tuple, X[~labels]))
    )


def test_imbalance_labels__classes_generator_match_class_column(tabular_data_labeled):
    X, y = tabular_data_labeled
    Xc = _with_class_column(X, y)
    rng = default_rng(0)
    selector = ClassSelector(
        "cls", {"cls": Xc.shape[1] - 1}, {"cls": {0: 0.5}}, rng=rng
    )
    generator = RandomSamplingClassesGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    assert labels[Xc[:, -1] != 0].sum() == 0
