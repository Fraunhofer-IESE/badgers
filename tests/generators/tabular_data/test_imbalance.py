import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import ClassSelector, ProportionalSelector
from badgers.core.utils import normalize_proba
from badgers.generators.tabular_data.imbalance.imbalance import RandomSamplingFeaturesGenerator, \
    RandomSamplingClassesGenerator, RandomSamplingTargetsGenerator


def test_random_sampling_classes__drops_proportion_per_class(tabular_data_labeled):
    """RandomSamplingClassesGenerator drops fired rows; class shares match proportions."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1

    proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
    rng = default_rng(0)
    # classes live in a COLUMN of X (y is deprecated as a data carrier)
    Xc = np.column_stack([X_np, np.asarray(y)])
    selector = ClassSelector("cls", {"cls": Xc.shape[1] - 1}, {"cls": proportion_classes}, rng=rng)
    generator = RandomSamplingClassesGenerator(random_generator=rng)

    Xt, yt = generator.generate(Xc.copy(), y, selector=selector)
    assert Xt.shape[1] == Xc.shape[1]
    assert Xt.shape[0] == len(yt)

    # each class loses min(int(prop * n_total), class size) rows
    yt_np = np.asarray(yt)
    y_np = np.asarray(Xc[:, -1])
    n = Xc.shape[0]
    for label in np.unique(y_np):
        class_size = int(np.sum(y_np == label))
        expected_dropped = min(int(proportion_classes[label] * n), class_size)
        actual_dropped = class_size - np.sum(yt_np == label)
        assert actual_dropped == expected_dropped, \
            f"Class {label}: expected {expected_dropped} dropped, got {actual_dropped}"


def test_random_sampling_classes__selector_none_is_noop(tabular_data_labeled):
    """selector=None keeps every row."""
    X, y = tabular_data_labeled
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(X.copy(), y)
    np.testing.assert_array_equal(np.asarray(Xt), np.asarray(X))
    np.testing.assert_array_equal(np.asarray(yt), np.asarray(y))


def test_random_sampling_classes__class_column_of_X(tabular_data_labeled):
    """The class variable may be a column of X (resolved via column_mapping)."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    Xc = np.column_stack([X_np, np.asarray(y)])
    rng = default_rng(0)
    selector = ClassSelector("cls", {"cls": Xc.shape[1] - 1}, {"cls": {0: 0.5}}, rng=rng)
    generator = RandomSamplingClassesGenerator(random_generator=rng)
    Xt, yt = generator.generate(Xc.copy(), y, selector=selector)
    # only class 0 rarefied: kept rows have either cls != 0 or belong to kept half
    kept_cls0 = int(np.sum(yt == 0))
    expected = int(np.sum(np.asarray(y) == 0)) - min(int(0.5 * Xc.shape[0]), int(np.sum(np.asarray(y) == 0)))
    assert kept_cls0 == expected


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


def test_random_sampling_targets__resamples_with_replacement():
    """Rows with LOW deficit probability are replicated more (weighted resampling)."""
    # 6 rows: f0 = 0..5; deficit proba proportional to f0, so low-f0 rows dominate
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    selector = ProportionalSelector("f0", {"f0": 0, "f1": 1})

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(X, np.arange(6), selector=selector, replace=True)
    assert Xt.shape[1] == 2
    assert Xt.shape[0] == 6  # size defaults to input row count
    # weighted toward low f0 -> resampled mean strictly below the input mean
    assert np.mean(Xt[:, 0]) < np.mean(X[:, 0])


def test_random_sampling_targets__without_replacement_is_thinning(tabular_data_labeled):
    """replace=False yields distinct rows (no duplication)."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    mapping = {name: i for i, name in enumerate(["f0", "f1", "cls"][:n_features])}
    selector = ProportionalSelector("f0", mapping)

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(X.copy(), y, selector=selector, replace=False, size=5)
    assert Xt.shape == (5, n_features)
    # all drawn rows must be distinct
    rows = {tuple(np.asarray(row)) for row in np.atleast_2d(Xt)}
    assert len(rows) == 5
    # y follows X
    assert len(yt) == 5


def test_random_sampling_targets__boolean_mask_selector_is_exact_drop(tabular_data_labeled):
    """A boolean selector: fired rows (deficit) get keep weight 0."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n = X_np.shape[0]
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1

    keep_idx = np.arange(0, n, 2)
    fired = np.ones(n, dtype=bool)
    fired[keep_idx] = False

    class FixedMaskSelector:
        def __call__(self, X, y=None):
            return {"f0": fired}

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(X.copy(), y, selector=FixedMaskSelector(), replace=False)
    np.testing.assert_allclose(np.asarray(Xt), X_np[keep_idx])


def test_random_sampling_targets__selector_none_raises(tabular_data_labeled):
    X, y = tabular_data_labeled
    generator = RandomSamplingTargetsGenerator()
    with pytest.raises(ValueError, match="selector"):
        generator.generate(X.copy(), y)


def test_random_sampling_targets__y_is_unused_for_selection(tabular_data_labeled):
    """Same selector call with y=data or y=None gives the same output rows."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    mapping = {name: i for i, name in enumerate(["f0", "f1", "cls"][:n_features])}
    selector = ProportionalSelector("f0", mapping)

    g1 = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    g2 = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt1, _ = g1.generate(X_np.copy(), y, selector=selector)
    Xt2, _ = g2.generate(X_np.copy(), None, selector=selector)
    np.testing.assert_allclose(Xt1, Xt2)


# --- get_imbalance_labels(): the fired/dropped mask of the last generate() call ---


def test_imbalance_labels__classes_generator_matches_dropped_rows(tabular_data_labeled):
    """RandomSamplingClassesGenerator: labels True exactly where rows were dropped."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    Xc = np.column_stack([X_np, np.asarray(y)])
    rng = default_rng(0)
    selector = ClassSelector("cls", {"cls": Xc.shape[1] - 1}, {"cls": {0: 0.5}}, rng=rng)
    generator = RandomSamplingClassesGenerator(random_generator=rng)
    Xt, yt = generator.generate(Xc.copy(), y, selector=selector)

    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == Xc.shape[0]
    # kept rows = complement of the fired mask
    np.testing.assert_array_equal(np.asarray(Xt), Xc[~labels])
    # only rows of class 0 can have fired
    assert labels[np.asarray(Xc[:, -1]) != 0].sum() == 0
    # output row count == number of False labels
    assert np.sum(~labels) == Xc.shape[0] - labels.sum() == len(yt)


def test_imbalance_labels__classes_generator_selector_none_all_false(tabular_data_labeled):
    """selector=None is a no-op: every label False."""
    X, y = tabular_data_labeled
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))
    generator.generate(np.asarray(X).copy(), y)
    labels = generator.get_imbalance_labels()
    assert not labels.any()


def test_imbalance_labels__classes_generator_raises_before_generate(tabular_data_labeled):
    """The getter raises when no generate() call happened yet."""
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(RuntimeError, match="generate"):
        generator.get_imbalance_labels()


def test_imbalance_labels__features_generator_is_complement_of_output(tabular_data_labeled):
    """RandomSamplingFeaturesGenerator: every False label row is drawn (rows may repeat)."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    generator = RandomSamplingFeaturesGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X_np.copy(), y)
    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == X_np.shape[0]
    # every kept (False) row appears in the resampled output
    drawn = set(map(tuple, np.asarray(Xt)))
    for row in X_np[~labels]:
        assert tuple(row) in drawn
    # every True label row is absent from the output
    for row in X_np[labels]:
        assert tuple(row) not in drawn


def test_imbalance_labels__targets_generator_boolean_selector_exact(tabular_data_labeled):
    """RandomSamplingTargetsGenerator with a boolean selector: True == dropped."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n = X_np.shape[0]
    keep_idx = np.arange(0, n, 2)
    fired = np.ones(n, dtype=bool)
    fired[keep_idx] = False

    class FixedMaskSelector:
        def __call__(self, X, y=None):
            return {"f0": fired}

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X_np.copy(), y, selector=FixedMaskSelector(), replace=False)
    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    np.testing.assert_array_equal(labels, fired)
    np.testing.assert_array_equal(np.asarray(Xt), X_np[~labels])


def test_imbalance_labels__targets_generator_fractional_selector_all_fired_dropped():
    """Fractional scores: rows never drawn carry the deficit (True)."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    selector = ProportionalSelector("f0", {"f0": 0, "f1": 1})
    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    # replace=False: distinct rows -> the strict complement invariant holds
    Xt, _ = generator.generate(X, np.arange(6), selector=selector, replace=False, size=4)
    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == 6
    np.testing.assert_array_equal(sorted(map(tuple, np.asarray(Xt))),
                                  sorted(map(tuple, X[~labels])))
    # replace=True: rows may repeat, but kept (False) rows and dropped (True)
    # rows are still disjoint in the output
    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt2, _ = generator.generate(X, np.arange(6), selector=selector, replace=True)
    labels2 = generator.get_imbalance_labels()
    drawn = set(map(tuple, np.asarray(Xt2)))
    for row in X[~labels2]:
        assert tuple(row) in drawn
    for row in X[labels2]:
        assert tuple(row) not in drawn


def test_imbalance_labels__last_call_wins(tabular_data_labeled):
    """The labels describe the LAST generate() call, not an earlier one."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    Xc = np.column_stack([X_np, np.asarray(y)])
    rng = default_rng(0)
    generator = RandomSamplingClassesGenerator(random_generator=rng)
    # first call: drops half of class 0
    generator.generate(Xc.copy(), y,
                       selector=ClassSelector("cls", {"cls": Xc.shape[1] - 1}, {"cls": {0: 0.5}}, rng=rng))
    n_dropped_first = generator.get_imbalance_labels().sum()
    # second call: no-op
    generator.generate(Xc.copy(), y)
    labels = generator.get_imbalance_labels()
    assert labels.sum() == 0
    assert n_dropped_first > 0
