import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import ClassSelector, ProportionalSelector
from badgers.generators.tabular_data.imbalance.imbalance import (
    RandomSamplingClassesGenerator,
    RandomSamplingTargetsGenerator,
)


def test_random_sampling_classes__drops_proportion_per_class(tabular_data_labeled):
    """Class selectors read X and drop the requested rows from each class."""
    Xc = np.asarray(tabular_data_labeled)
    proportion_classes = {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1}
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions=proportion_classes, rng=rng
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
    Xc = np.asarray(tabular_data_labeled)
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))

    Xt = generator.generate(Xc.copy())

    np.testing.assert_array_equal(Xt, Xc)
    assert not generator.get_imbalance_labels().any()


def test_random_sampling_classes__class_column_of_X(tabular_data_labeled):
    """The class variable is selected from its mapped column in X."""
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions={0: 0.5}, rng=rng
    )
    generator = RandomSamplingClassesGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    expected = np.sum(Xc[:, -1] == 0) - min(
        int(0.5 * Xc.shape[0]), np.sum(Xc[:, -1] == 0)
    )
    assert np.sum(Xt[:, -1] == 0) == expected


def test_random_sampling_targets__continuous_selector_preserves_shape(
    tabular_data,
):
    """A fractional selector over a continuous column resamples n rows with
    replacement; every generated row comes from the input."""
    X = np.asarray(tabular_data)
    selector = ProportionalSelector([0], 0)

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), selector=selector, replace=True)

    assert Xt.shape == X.shape
    input_rows = {tuple(row) for row in X}
    for row in Xt:
        assert tuple(row) in input_rows
    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool and len(labels) == X.shape[0]


def test_random_sampling_targets__resamples_with_replacement():
    """Low deficit probability rows are replicated more."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    selector = ProportionalSelector([0], 0)

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X, selector=selector, replace=True)

    assert Xt.shape == X.shape
    assert np.mean(Xt[:, 0]) < np.mean(X[:, 0])


def test_random_sampling_targets__without_replacement_is_thinning(
    tabular_data,
):
    """replace=False yields distinct rows and honors requested output size."""
    X = np.asarray(tabular_data)
    selector = ProportionalSelector([0], 0)

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(), selector=selector, replace=False, size=5
    )

    assert Xt.shape == (5, X.shape[1])
    assert len({tuple(row) for row in Xt}) == 5


def test_random_sampling_targets__boolean_mask_selector_is_exact_drop():
    """A boolean selector drops selected rows and the getter records that mask."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    keep_idx = np.arange(0, len(X), 2)
    selected = np.ones(len(X), dtype=bool)
    selected[keep_idx] = False

    class FixedMaskSelector:
        def __call__(self, X, targets=None):
            return {0: selected}

    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(), selector=FixedMaskSelector(), replace=False
    )

    np.testing.assert_array_equal(Xt, X[keep_idx])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), selected)


def test_random_sampling_targets__selector_none_raises(tabular_data):
    generator = RandomSamplingTargetsGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="selector"):
        generator.generate(np.asarray(tabular_data).copy())


def test_imbalance_labels__classes_generator_raises_before_generate():
    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(RuntimeError, match="generate"):
        generator.get_imbalance_labels()


def test_random_sampling_targets__class_selector_is_exact_per_class_drop(
    tabular_data_labeled,
):
    """A boolean ClassSelector fed to the resampler drops exactly the selected
    rows of that class (the removed per-class behavior, one generator)."""
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions={0: 0.5}, rng=rng
    )
    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))

    Xt = generator.generate(Xc.copy(), selector=selector, replace=False)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    expected_dropped = min(int(0.5 * Xc.shape[0]), np.sum(Xc[:, -1] == 0))
    assert np.sum(labels & (Xc[:, -1] == 0)) == expected_dropped
    assert labels[Xc[:, -1] != 0].sum() == 0


def test_imbalance_labels__targets_generator_fractional_selector():
    """Fractional scores label rows not selected into the generated X."""
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    selector = ProportionalSelector([0], 0)
    generator = RandomSamplingTargetsGenerator(random_generator=default_rng(0))

    Xt = generator.generate(X, selector=selector, replace=False, size=4)

    labels = generator.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == len(X)
    np.testing.assert_array_equal(
        sorted(map(tuple, Xt)), sorted(map(tuple, X[~labels]))
    )


def test_imbalance_labels__classes_generator_match_class_column(tabular_data_labeled):
    Xc = np.asarray(tabular_data_labeled)
    rng = default_rng(0)
    selector = ClassSelector(
        [0], class_column=Xc.shape[1] - 1, proportions={0: 0.5}, rng=rng
    )
    generator = RandomSamplingClassesGenerator(random_generator=rng)

    Xt = generator.generate(Xc.copy(), selector=selector)

    labels = generator.get_imbalance_labels()
    np.testing.assert_array_equal(Xt, Xc[~labels])
    assert labels[Xc[:, -1] != 0].sum() == 0


def test_imbalance_labels__classes_generator_tracks_most_recent_call():
    """The getter reports dropped rows from the latest X-only generation."""
    X = np.arange(4.0).reshape(-1, 1)
    first_mask = np.array([True, False, False, False])
    second_mask = np.array([False, False, False, True])

    class FixedMaskSelector:
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, X, targets=None):
            return {0: self.mask}

    generator = RandomSamplingClassesGenerator(random_generator=default_rng(0))

    Xt_first = generator.generate(X.copy(), selector=FixedMaskSelector(first_mask))
    np.testing.assert_array_equal(Xt_first, X[~first_mask])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), first_mask)

    Xt_second = generator.generate(
        X.copy(), selector=FixedMaskSelector(second_mask)
    )
    np.testing.assert_array_equal(Xt_second, X[~second_mask])
    np.testing.assert_array_equal(generator.get_imbalance_labels(), second_mask)
