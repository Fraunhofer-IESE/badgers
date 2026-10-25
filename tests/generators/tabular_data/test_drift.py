import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import (
    AlwaysSelector,
    BernoulliSelector,
    ClassSelector,
    TopQuantileSelector,
)
from badgers.generators.tabular_data.drift.drift import (
    FixedShiftClassesGenerator,
    FixedShiftGenerator,
    RandomShiftGenerator,
    RandomShiftClassesGenerator,
)


def test_random_shift__selector_none_matches_whole_data_shift(tabular_data):
    """selector=None keeps the backward-compatible behavior: every row shifts."""
    X = np.asarray(tabular_data, dtype=float)
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=0.1, selector=None)
    # every row moved by the same per-column amount
    delta = Xt - X
    np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=1e-8)


def test_random_shift__selector_restricts_shifted_rows(tabular_data):
    """Only the rows fired by the selector are shifted; others stay untouched."""
    X = np.asarray(tabular_data, dtype=float)
    selector = TopQuantileSelector([0], 0, 0.8)
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=2.0, selector=selector)
    fired = selector(X)[0]
    assert fired.any() and not fired.all()
    np.testing.assert_allclose(Xt[~fired], X[~fired], atol=1e-12)
    assert not np.allclose(Xt[fired], X[fired])


def test_random_shift__class_selector_shifts_only_that_class(tabular_data_labeled):
    """A ClassSelector over the class column shifts only the listed class's rows."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, cls_col])
    selector = ClassSelector([0], class_column=cls_col, proportions={classes[0]: 1.0}, rng=default_rng(0))
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0)
    target = Xc[:, cls_col] == classes[0]
    assert not np.allclose(Xt[target], Xc[target])
    assert np.allclose(Xt[~target], Xc[~target])


def test_random_shift__masks_computed_on_unmutated_input():
    """The selector sees the original rows (compute-all-masks-first invariant)."""
    seen = []

    class Spy:
        def __call__(self, X, targets=None):
            seen.append(np.array(X, copy=True))
            return {0: np.ones(X.shape[0], dtype=bool)}

    X = np.arange(12.0).reshape(6, 2)
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    generator.generate(X.copy(), shift_std=1.0, selector=Spy())
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], X)


def test_random_shift__always_selector_equals_selector_none(tabular_data):
    """AlwaysSelector is the degenerate gate and reproduces the default path."""
    X = np.asarray(tabular_data, dtype=float)
    g1 = RandomShiftGenerator(random_generator=default_rng(0))
    g2 = RandomShiftGenerator(random_generator=default_rng(0))
    Xt_none = g1.generate(X.copy(), shift_std=0.1, selector=None)
    Xt_always = g2.generate(X.copy(), shift_std=0.1, selector=AlwaysSelector([0]))
    np.testing.assert_array_equal(Xt_none, Xt_always)


def test_random_shift__selector_none_is_relevant_default_selector(tabular_data):
    """selector=None IS a relevant default selector: AlwaysSelector over
    every column. The explicit and default paths are identical."""
    X = np.asarray(tabular_data, dtype=float)
    g1 = RandomShiftGenerator(random_generator=default_rng(0))
    g2 = RandomShiftGenerator(random_generator=default_rng(0))
    Xt_none = g1.generate(X.copy(), shift_std=0.1, selector=None)
    Xt_default = g2.generate(
        X.copy(), shift_std=0.1, selector=AlwaysSelector(range(X.shape[1])))
    np.testing.assert_array_equal(Xt_none, Xt_default)


def test_fixed_shift__selector_none_matches_whole_data_shift(tabular_data):
    """selector=None keeps the backward-compatible behavior: every row shifts."""
    X = np.asarray(tabular_data, dtype=float)
    generator = FixedShiftGenerator(random_generator=np.random.default_rng(0))
    Xt = generator.generate(X.copy(), shift=1.0, selector=None)
    delta = Xt - X
    np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=1e-8)


def test_fixed_shift__selector_restricts_shifted_rows(tabular_data):
    """Only the rows fired by the selector receive the exact shift."""
    X = np.asarray(tabular_data, dtype=float)
    selector = TopQuantileSelector([0], 0, 0.8)
    generator = FixedShiftGenerator(random_generator=np.random.default_rng(0))
    Xt = generator.generate(X.copy(), shift=1.0, selector=selector)
    fired = selector(X)[0]
    assert fired.any() and not fired.all()
    np.testing.assert_allclose(Xt[~fired], X[~fired], atol=1e-12)
    # fired rows moved by exactly one std per column
    delta = Xt[fired] - X[fired]
    expected = 1.0 * X.std(axis=0, ddof=0)
    np.testing.assert_allclose(delta, np.broadcast_to(expected, delta.shape), atol=1e-8)


def test_fixed_shift__deterministic_with_selector(tabular_data):
    """FixedShiftGenerator with a selector stays reproducible."""
    X = np.asarray(tabular_data, dtype=float)
    generator = FixedShiftGenerator(random_generator=np.random.default_rng(0))
    Xt1 = generator.generate(X.copy(), shift=1.0, selector=BernoulliSelector([0], rate=0.5, rng=default_rng(1)))
    Xt2 = generator.generate(X.copy(), shift=1.0, selector=BernoulliSelector([0], rate=0.5, rng=default_rng(1)))
    np.testing.assert_array_equal(Xt1, Xt2)


def test_fixed_shift__selector_none_is_relevant_default_selector(tabular_data):
    """selector=None IS a relevant default selector: AlwaysSelector over
    every column. The explicit and default paths are identical."""
    X = np.asarray(tabular_data, dtype=float)
    generator = FixedShiftGenerator(random_generator=np.random.default_rng(0))
    Xt_none = generator.generate(X.copy(), shift=1.0, selector=None)
    Xt_default = generator.generate(
        X.copy(), shift=1.0, selector=AlwaysSelector(range(X.shape[1])))
    np.testing.assert_array_equal(Xt_none, Xt_default)


def test_random_shift__preserves_shape_and_modifies_data(tabular_data):
    """RandomShiftGenerator with scalar shift_std preserves shape and modifies data."""
    X = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=0.1)
    assert len(X) == len(Xt)
    X_np = np.asarray(X)
    Xt_np = np.asarray(Xt)
    assert not np.allclose(X_np, Xt_np)


def test_random_shift__zero_shift_preserves_data(tabular_data):
    """RandomShiftGenerator with shift_std=0 produces identical output."""
    X = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=0.0)
    X_np = np.asarray(X)
    Xt_np = np.asarray(Xt)
    np.testing.assert_allclose(X_np, Xt_np)


def test_random_shift__preserves_shape_array(tabular_data):
    """RandomShiftGenerator with array shift_std preserves shape and modifies data."""
    X = tabular_data
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    shift_std = np.array([0.1]) if n_features == 1 else np.linspace(0.1, 1, n_features)

    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=shift_std)
    assert len(X) == len(Xt)
    assert not np.allclose(X_np, np.asarray(Xt))


def test_random_shift__signed_array_shift_std(tabular_data):
    """generate(X=..., shift_std=negative-array) uses sign as shift direction."""
    X = tabular_data
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    # signed magnitudes: alternate positive/negative shift directions
    signed = np.array([2.0 if i % 2 == 0 else -1.0 for i in range(n_features)])
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X=X_np.copy(), shift_std=signed)
    assert np.asarray(Xt).shape == X_np.shape


def test_random_shift_classes__preserves_shape_and_modifies_data(tabular_data_labeled):
    """RandomShiftClassesGenerator with scalar shift_std preserves shape and modifies data."""
    Xc = np.asarray(tabular_data_labeled)
    classes = Xc[:, -1]
    selector = ClassSelector([0], class_column=Xc.shape[1] - 1, proportions={c: 0.5 for c in np.unique(classes)}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=0.1)
    assert len(Xc) == len(Xt)
    assert not np.allclose(np.asarray(Xc), np.asarray(Xt))


def test_random_shift_classes__selector_none_raises(tabular_data_labeled):
    """A per-class rule has no class-free default: selector=None raises."""
    Xc = np.asarray(tabular_data_labeled)
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="selector"):
        generator.generate(Xc.copy(), shift_std=0.1)


def test_random_shift_classes__class_column_not_shifted(tabular_data_labeled):
    """Default: feature columns shift, the class column stays untouched."""
    Xc = np.asarray(tabular_data_labeled)
    classes = Xc[:, -1]
    selector = ClassSelector([0], class_column=Xc.shape[1] - 1, proportions={c: 1.0 for c in np.unique(classes)}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0)
    Xt = np.asarray(Xt)
    assert np.allclose(Xt[:, -1], Xc[:, -1])          # class column untouched
    assert not np.allclose(Xt[:, :-1], Xc[:, :-1])    # feature columns shifted


def test_random_shift_classes__explicit_variables(tabular_data_labeled):
    """variables= restricts the shifted columns (int positions)."""
    Xc = np.asarray(tabular_data_labeled)
    classes = Xc[:, -1]
    selector = ClassSelector([0], class_column=Xc.shape[1] - 1, proportions={c: 1.0 for c in np.unique(classes)}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0, variables=[0])
    Xt = np.asarray(Xt)
    assert not np.allclose(Xt[:, 0], Xc[:, 0])        # column 0 shifted
    assert np.allclose(Xt[:, 1:], Xc[:, 1:])          # everything else untouched


def test_random_shift_classes__selector_without_class_column_raises(tabular_data_labeled):
    """The selector owns the class information: a selector that does not
    expose a class_column (e.g. BernoulliSelector) raises instead of the old
    "all fired rows are one class" fallback."""
    Xc = np.asarray(tabular_data_labeled)
    selector = BernoulliSelector([0], rate=1.0, rng=default_rng(1))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="class_column"):
        generator.generate(Xc.copy(), selector=selector, shift_std=1.0)


def test_random_shift_classes__selector_out_of_range_class_column_raises(tabular_data_labeled):
    """A selector whose class_column is out of range raises instead of the
    old one-class fallback."""
    Xc = np.asarray(tabular_data_labeled)
    selector = ClassSelector([0], class_column=Xc.shape[1], proportions={0: 1.0}, rng=default_rng(1))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="not a valid column position"):
        generator.generate(Xc.copy(), selector=selector, shift_std=1.0)


def test_random_shift_classes__per_class_dict_shift_std(tabular_data_labeled):
    """Dict shift_std shifts only the listed classes, one draw per class;
    a fired class value absent from the dict raises."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    # fire every row (all classes), magnitudes only for the first class
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    shift_std = {classes[0]: 0.5}
    with pytest.raises(ValueError, match="no entry for the fired class"):
        generator.generate(Xc.copy(), selector=selector, shift_std=shift_std)
    # with an entry for every fired class only the listed class is shifted
    full = {c: 0.5 if c == classes[0] else 0.0 for c in classes}
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=full)
    fired_cls0 = Xc[:, cls_col] == classes[0]
    other = Xc[:, cls_col] != classes[0]
    assert not np.allclose(np.asarray(Xt)[fired_cls0], Xc[fired_cls0])
    assert np.allclose(np.asarray(Xt)[other], Xc[other])


def test_random_shift_classes__dict_int_keys_float_column(tabular_data_labeled):
    """Plain-int dict keys work for a float-encoded class column (dtype
    normalization)."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    assert Xc[:, cls_col].dtype == float and set(classes) == {0.0, 1.0, 2.0, 3.0, 4.0}
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std={int(c): 0.5 for c in classes})
    assert not np.allclose(np.asarray(Xt), Xc)


def test_random_shift_classes__dict_wrong_kind_keys_raise(tabular_data_labeled):
    """Keys of a wrong kind (strings for numeric classes) raise instead of
    silently skipping every class."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="cannot be compared"):
        generator.generate(
            Xc.copy(), selector=selector,
            shift_std={"class_0": 0.5, "class_1": 0.5, "class_2": 0.5, "class_3": 0.5})


def test_random_shift_classes__signed_shift_std_is_direction(tabular_data_labeled):
    """Negative shift_std entries drift the class in the OPPOSITE direction
    with the SAME magnitude: the mirror of the positive-std result (the rng
    draws are identical, only the sign differs)."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    Xp = Xc.copy()
    Xn = Xc.copy()
    gp = RandomShiftClassesGenerator(random_generator=default_rng(0))
    gn = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt_pos = gp.generate(Xp, selector=selector, shift_std={c: 4.0 for c in classes})
    Xt_neg = gn.generate(Xn, selector=selector, shift_std={c: -4.0 for c in classes})
    np.testing.assert_allclose(np.asarray(Xn), 2 * Xc - np.asarray(Xp))


def test_fixed_shift__deterministic_and_exact(tabular_data):
    """FixedShiftGenerator shifts each column by exactly shift, reproducibly."""
    X = tabular_data
    X_np = np.asarray(X, dtype=float)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    shift = np.linspace(1.0, -1.0, n_features)
    generator = FixedShiftGenerator(random_generator=np.random.default_rng(0))
    Xt1 = generator.generate(X=X_np.copy(), shift=shift)
    Xt2 = generator.generate(X=X_np.copy(), shift=shift)
    np.testing.assert_array_equal(Xt1, Xt2)  # deterministic
    delta = Xt1 - X_np
    expected = shift * X_np.std(axis=0, ddof=0)  # StandardScaler uses population std
    np.testing.assert_allclose(delta, np.broadcast_to(expected, delta.shape), atol=1e-8)


def test_fixed_shift_classes__deterministic_and_per_class(tabular_data_labeled):
    """FixedShiftClassesGenerator: same input -> same output; one exact shift per class."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in np.unique(Xc[:, -1])}, rng=default_rng(0))
    generator = FixedShiftClassesGenerator(random_generator=np.random.default_rng(0))
    Xt1 = generator.generate(X=Xc.copy(), selector=selector, shift=1.0)
    Xt2 = generator.generate(X=Xc.copy(), selector=selector, shift=1.0)
    np.testing.assert_array_equal(Xt1, Xt2)  # deterministic
    assert np.allclose(np.asarray(Xt1)[:, cls_col], Xc[:, cls_col])  # class col untouched
    # per-class shifts are exact: scalar shift 1.0 -> exactly 1 std per feature column
    delta = np.asarray(Xt1) - Xc
    for c in np.unique(Xc[:, -1]):
        rows = Xc[:, cls_col] == c
        expected = 1.0 * Xc[:, :-1].std(axis=0, ddof=0)
        np.testing.assert_allclose(
            delta[rows][:, :-1], np.broadcast_to(expected, delta[rows][:, :-1].shape), atol=1e-8
        )


def test_fixed_shift_classes__dict_shift_per_class(tabular_data_labeled):
    """Dict shift applies exact shifts only to the listed classes; a fired
    class value absent from the dict raises."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    generator = FixedShiftClassesGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="no entry for the fired class"):
        generator.generate(X=Xc.copy(), selector=selector, shift={classes[0]: 1.0})
    # with an entry for every fired class only the listed class is shifted
    Xt = generator.generate(X=Xc.copy(), selector=selector, shift={c: 1.0 if c == classes[0] else 0.0 for c in classes})
    fired = Xc[:, cls_col] == classes[0]
    other = ~fired
    assert not np.allclose(np.asarray(Xt)[fired], Xc[fired])
    assert np.allclose(np.asarray(Xt)[other], Xc[other])


def test_fixed_shift_classes__selector_without_class_column_raises(tabular_data_labeled):
    """The selector owns the class information: a selector that does not
    expose a class_column raises instead of the old one-class fallback."""
    Xc = np.asarray(tabular_data_labeled)
    selector = BernoulliSelector([0], rate=1.0, rng=default_rng(1))
    generator = FixedShiftClassesGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="class_column"):
        generator.generate(Xc.copy(), selector=selector, shift=1.0)


def test_fixed_shift_classes__dict_int_keys_float_column(tabular_data_labeled):
    """Plain-int dict keys work for a float-encoded class column (dtype
    normalization)."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    assert Xc[:, cls_col].dtype == float and set(classes) == {0.0, 1.0, 2.0, 3.0, 4.0}
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    generator = FixedShiftClassesGenerator(random_generator=np.random.default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift={int(c): 0.5 for c in classes})
    assert not np.allclose(np.asarray(Xt), Xc)


def test_fixed_shift_classes__dict_wrong_kind_keys_raise(tabular_data_labeled):
    """Keys of a wrong kind (strings for numeric classes) raise instead of
    silently skipping every class."""
    Xc = np.asarray(tabular_data_labeled)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(Xc[:, -1])
    selector = ClassSelector([0], class_column=cls_col, proportions={c: 1.0 for c in classes}, rng=default_rng(0))
    generator = FixedShiftClassesGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="cannot be compared"):
        generator.generate(
            Xc.copy(), selector=selector,
            shift={"class_0": 0.5, "class_1": 0.5, "class_2": 0.5, "class_3": 0.5})
