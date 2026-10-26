import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import (
    BernoulliSelector,
    ClassCountSelector,
    ClassSelector,
    ProportionalSelector,
    TopQuantileSelector,
)
from badgers.generators.tabular_data.imbalance import (
    ConstantMultiplicity,
    MaskedMultiplicity,
    OversamplingGenerator,
    UndersamplingGenerator,
)


# ---------------------------------------------------------------------------
# UndersamplingGenerator
# ---------------------------------------------------------------------------


def test_undersampling__class_selector_drops_proportion_per_class(
    tabular_data_labeled,
):
    """Class selectors read X and drop the requested rows from each class."""
    Xc = np.asarray(tabular_data_labeled)
    proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions=proportion_classes, rng=rng
    )
    generator = UndersamplingGenerator(random_generator=rng)

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


def test_undersampling__selector_none_is_noop(tabular_data_labeled):
    """selector=None keeps every row and records no dropped rows."""
    Xc = np.asarray(tabular_data_labeled)
    generator = UndersamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(Xc.copy())

    np.testing.assert_array_equal(Xt, Xc)
    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool and len(labels) == Xc.shape[0]
    assert not labels.any()


def test_undersampling__class_column_of_X(tabular_data_labeled):
    """The class variable is selected from its mapped column in X."""
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions={0: 0.5}, rng=rng
    )
    generator = UndersamplingGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    expected = np.sum(Xc[:, -1] == 0) - min(
        int(0.5 * Xc.shape[0]), np.sum(Xc[:, -1] == 0)
    )
    assert np.sum(Xt[:, -1] == 0) == expected


def test_undersampling__class_selector_is_exact_per_class_drop(
    tabular_data_labeled,
):
    """A boolean ClassSelector drops exactly the selected rows of that class."""
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions={0: 0.5}, rng=rng
    )
    generator = UndersamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(Xc.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    expected_dropped = min(int(0.5 * Xc.shape[0]), np.sum(Xc[:, -1] == 0))
    assert np.sum(labels & (Xc[:, -1] == 0)) == expected_dropped
    assert labels[Xc[:, -1] != 0].sum() == 0


def test_undersampling__boolean_mask_selector_is_exact_drop(tabular_data):
    """A boolean selector drops selected rows and the getter records that mask."""
    X = np.asarray(tabular_data)
    keep_idx = np.arange(0, len(X), 2)
    selected = np.ones(len(X), dtype=bool)
    selected[keep_idx] = False

    class FixedMaskSelector:
        def __call__(self, X, targets=None):
            return {0: selected}

    generator = UndersamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), selector=FixedMaskSelector())

    np.testing.assert_array_equal(Xt, X[keep_idx])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), selected)


def test_undersampling__fractional_selector_drops_consistent_rows(tabular_data):
    """A fractional selector (ProportionalSelector) drops rows stochastically,
    and the output is always exactly the complement of the label mask."""
    X = np.asarray(tabular_data)
    selector = ProportionalSelector([0], 0)
    generator = UndersamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool and len(labels) == X.shape[0]
    assert Xt.shape[1] == X.shape[1]
    np.testing.assert_array_equal(Xt, X[~labels])


def test_undersampling__value_free_bernoulli_selector(tabular_data):
    """A value-free Bernoulli selector drops a stochastic ~rate fraction of rows."""
    X = np.asarray(tabular_data)
    rng = default_rng(0)
    selector = BernoulliSelector([0], rate=0.5, rng=rng)
    generator = UndersamplingGenerator(random_generator=default_rng(1))

    Xt = generator.generate(X.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool and len(labels) == X.shape[0]
    np.testing.assert_array_equal(Xt, X[~labels])
    # rate=0.5 over 100 rows: the drop count is statistically ~50.
    assert 20 <= np.sum(labels) <= 80


def test_undersampling__value_driven_top_quantile(tabular_data):
    """TopQuantileSelector drops exactly the rows at/above the column quantile."""
    X = np.asarray(tabular_data)
    selector = TopQuantileSelector([0], column=0, quantile=0.5)
    generator = UndersamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X.copy(), selector=selector)

    threshold = np.quantile(X[:, 0], 0.5)
    expected_mask = X[:, 0] >= threshold
    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(labels, expected_mask)
    np.testing.assert_array_equal(Xt, X[~labels])


def test_undersampling__value_driven_class_count_selector(tabular_data_labeled):
    """ClassCountSelector drops exactly the requested number of rows per class."""
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassCountSelector(
        [0], class_column=Xc.shape[1] - 1, counts={0: 5, 2: 3}, rng=rng
    )
    generator = UndersamplingGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    assert np.sum(labels & (Xc[:, -1] == 0)) == 5
    assert np.sum(labels & (Xc[:, -1] == 2)) == 3
    assert labels[(Xc[:, -1] != 0) & (Xc[:, -1] != 2)].sum() == 0


def test_undersampling__imbalance_labels_match_class_column(tabular_data_labeled):
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions={0: 0.5}, rng=rng
    )
    generator = UndersamplingGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    assert labels[Xc[:, -1] != 0].sum() == 0


def test_undersampling__imbalance_labels_tracks_most_recent_call():
    """The getter reports dropped rows from the latest generation."""
    X = np.arange(4.0).reshape(-1, 1)
    first_mask = np.array([True, False, False, False])
    second_mask = np.array([False, False, False, True])

    class FixedMaskSelector:
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, X, targets=None):
            return {0: self.mask}

    generator = UndersamplingGenerator(random_generator=default_rng(0))

    Xt_first = generator.generate(X.copy(), selector=FixedMaskSelector(first_mask))
    np.testing.assert_array_equal(Xt_first, X[~first_mask])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), first_mask)

    Xt_second = generator.generate(
        X.copy(), selector=FixedMaskSelector(second_mask)
    )
    np.testing.assert_array_equal(Xt_second, X[~second_mask])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), second_mask)


def test_undersampling__imbalance_labels_before_generate_raises():
    """No labels exist until the first generation."""
    generator = UndersamplingGenerator(random_generator=default_rng(0))
    with pytest.raises(AttributeError, match="imbalance"):
        generator.get_imbalance_labels()


# ---------------------------------------------------------------------------
# OversamplingGenerator
# ---------------------------------------------------------------------------


def test_oversampling__selector_none_is_noop(tabular_data):
    """selector=None keeps every row once and records no duplicated rows."""
    X = np.asarray(tabular_data)
    generator = OversamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X.copy())

    np.testing.assert_array_equal(Xt, X)
    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool and len(labels) == X.shape[0]
    assert not labels.any()


def test_oversampling__constant_multiplicity_2():
    """ConstantMultiplicity(2) doubles the row count with adjacent copies."""
    X = np.arange(8.0).reshape(4, 2)
    generator = OversamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X, ConstantMultiplicity(2))

    assert Xt.shape == (8, 2)
    # Row order preserved: row i is immediately followed by its copy.
    np.testing.assert_array_equal(Xt, np.repeat(X, 2, axis=0))
    # Each input row appears exactly twice.
    for row in X:
        assert np.sum((Xt == row).all(axis=1)) == 2
    labels = generator.get_imbalance_labels()
    assert len(labels) == X.shape[0]
    assert labels.all()


def test_oversampling__constant_multiplicity_3():
    """ConstantMultiplicity(3) gives every input row exactly three copies."""
    X = np.arange(6.0).reshape(3, 2)
    generator = OversamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X, ConstantMultiplicity(3))

    assert Xt.shape == (9, 2)
    np.testing.assert_array_equal(Xt, np.repeat(X, 3, axis=0))
    for row in X:
        assert np.sum((Xt == row).all(axis=1)) == 3
    assert generator.get_imbalance_labels().all()


def test_oversampling__masked_multiplicity_bool_mask():
    """MaskedMultiplicity with a bool array duplicates only masked rows."""
    X = np.arange(8.0).reshape(4, 2)
    mask = np.array([True, False, True, False])
    generator = OversamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X, MaskedMultiplicity(mask, multiplicity=2))

    # Masked rows (0, 2) appear twice, others once, original order kept.
    np.testing.assert_array_equal(Xt, np.repeat(X, [2, 1, 2, 1], axis=0))
    assert Xt.shape == (6, 2)
    np.testing.assert_array_equal(generator.get_imbalance_labels(), mask)


def test_oversampling__masked_multiplicity_callable_mask():
    """MaskedMultiplicity with a callable mask depends on the data at call time."""
    X = np.column_stack([np.arange(4.0), [0, 1, 0, 1]])
    generator = OversamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(
        X, MaskedMultiplicity(lambda X: X[:, 1] == 1, multiplicity=3)
    )

    np.testing.assert_array_equal(Xt, np.repeat(X, [1, 3, 1, 3], axis=0))
    assert Xt.shape == (8, 2)
    np.testing.assert_array_equal(
        generator.get_imbalance_labels(), np.array([False, True, False, True])
    )


def test_oversampling__imbalance_labels_equal_multiplicity_gt_1():
    """Labels are (multiplicity > 1), aligned with the input rows."""
    X = np.arange(8.0).reshape(4, 2)
    multiplicities = np.array([3, 1, 1, 2])
    generator = OversamplingGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X, multiplicities)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(labels, multiplicities > 1)
    assert len(labels) == X.shape[0]
    assert Xt.shape == (int(multiplicities.sum()), 2)
    np.testing.assert_array_equal(Xt, np.repeat(X, multiplicities, axis=0))


def test_oversampling__rejects_deficit_style_and_invalid_multiplicities():
    """Mix-up guards: deficit-style, zero, negative, and non-integer values."""
    X = np.arange(8.0).reshape(4, 2)
    generator = OversamplingGenerator(random_generator=default_rng(0))

    # Fractional values in (0, 1) look like a deficit Selector.
    with pytest.raises(ValueError, match="deficit Selector"):
        generator.generate(X, np.full(4, 0.5))
    # A 0 multiplicity would drop a row; oversampling never drops.
    with pytest.raises(ValueError, match="0 copies"):
        generator.generate(X, np.array([2, 0, 1, 1]))
    # Negative multiplicities are invalid.
    with pytest.raises(ValueError, match="must be >= 1"):
        generator.generate(X, np.array([2, -1, 1, 1]))
    # Non-integer values are not supported.
    with pytest.raises(ValueError, match="non-integer"):
        generator.generate(X, 3.5)
    with pytest.raises(ValueError, match="non-integer"):
        generator.generate(X, np.full(4, 2.5))
    # Bare ints broadcast to every row; bare int arrays are used as-is.
    np.testing.assert_array_equal(
        generator.generate(X, 2), np.repeat(X, 2, axis=0)
    )
    np.testing.assert_array_equal(
        generator.generate(X, np.array([1, 2, 1, 1])),
        np.repeat(X, [1, 2, 1, 1], axis=0),
    )
