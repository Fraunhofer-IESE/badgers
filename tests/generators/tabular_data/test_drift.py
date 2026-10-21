import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import BernoulliSelector, ClassSelector
from badgers.generators.tabular_data.drift.drift import (
    FixedShiftClassesGenerator,
    FixedShiftGenerator,
    RandomShiftGenerator,
    RandomShiftClassesGenerator,
)


def test_random_shift__preserves_shape_and_modifies_data(tabular_data):
    """RandomShiftGenerator with scalar shift_std preserves shape and modifies data."""
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=0.1)
    assert len(X) == len(Xt)
    X_np = np.asarray(X)
    Xt_np = np.asarray(Xt)
    assert not np.allclose(X_np, Xt_np)


def test_random_shift__zero_shift_preserves_data(tabular_data):
    """RandomShiftGenerator with shift_std=0 produces identical output."""
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=0.0)
    X_np = np.asarray(X)
    Xt_np = np.asarray(Xt)
    np.testing.assert_allclose(X_np, Xt_np)


def test_random_shift__preserves_shape_array(tabular_data):
    """RandomShiftGenerator with array shift_std preserves shape and modifies data."""
    X, y = tabular_data
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    shift_std = np.array([0.1]) if n_features == 1 else np.linspace(0.1, 1, n_features)

    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), shift_std=shift_std)
    assert len(X) == len(Xt)
    assert not np.allclose(X_np, np.asarray(Xt))


def test_random_shift__signed_array_shift_std(tabular_data):
    """generate(X=..., shift_std=negative-array) uses sign as shift direction."""
    X, y = tabular_data
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    # signed magnitudes: alternate positive/negative shift directions
    signed = np.array([2.0 if i % 2 == 0 else -1.0 for i in range(n_features)])
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X=X_np.copy(), shift_std=signed)
    assert np.asarray(Xt).shape == X_np.shape


def test_random_shift_classes__preserves_shape_and_modifies_data(tabular_data_labeled):
    """RandomShiftClassesGenerator with scalar shift_std preserves shape and modifies data."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)])
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = Xc.shape[1] - 1
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 0.5 for c in np.unique(y)}}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=0.1)
    assert len(Xc) == len(Xt)
    assert not np.allclose(np.asarray(Xc), np.asarray(Xt))


def test_random_shift_classes__selector_none_raises(tabular_data_labeled):
    """A per-class rule has no class-free default: selector=None raises."""
    X, y = tabular_data_labeled
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="selector"):
        generator.generate(X.copy(), shift_std=0.1)


def test_random_shift_classes__class_column_not_shifted(tabular_data_labeled):
    """Default: feature columns shift, the class column stays untouched."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = Xc.shape[1] - 1
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 1.0 for c in np.unique(y)}}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0)
    Xt = np.asarray(Xt)
    assert np.allclose(Xt[:, -1], Xc[:, -1])          # class column untouched
    assert not np.allclose(Xt[:, :-1], Xc[:, :-1])    # feature columns shifted


def test_random_shift_classes__explicit_variables(tabular_data_labeled):
    """variables= restricts the shifted columns (names via column_mapping)."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = Xc.shape[1] - 1
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 1.0 for c in np.unique(y)}}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0, variables=["f0"])
    Xt = np.asarray(Xt)
    assert not np.allclose(Xt[:, 0], Xc[:, 0])        # f0 shifted
    assert np.allclose(Xt[:, 1:], Xc[:, 1:])          # everything else untouched


def test_random_shift_classes__explicit_class_column_int(tabular_data_labeled):
    """class_column as int works even with a selector without column_mapping."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    selector = BernoulliSelector(["rows"], rate=1.0, rng=default_rng(1))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0, class_column=Xc.shape[1] - 1)
    Xt = np.asarray(Xt)
    assert np.allclose(Xt[:, -1], Xc[:, -1])          # class column untouched
    assert not np.allclose(Xt[:, :-1], Xc[:, :-1])    # feature columns shifted


def test_random_shift_classes__explicit_class_column_name(tabular_data_labeled):
    """class_column as name is resolved through selector.column_mapping."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = Xc.shape[1] - 1
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 1.0 for c in np.unique(y)}}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=2.0, class_column="cls")
    assert np.allclose(np.asarray(Xt)[:, -1], Xc[:, -1])


def test_random_shift_classes__explicit_class_column_bad_name_raises(tabular_data_labeled):
    """An unresolvable class_column name raises ValueError."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    # selector without column_mapping: a name cannot be resolved
    selector = BernoulliSelector(["rows"], rate=1.0, rng=default_rng(1))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="class column"):
        generator.generate(Xc.copy(), selector=selector, shift_std=1.0, class_column="cls")


def test_random_shift_classes__per_class_dict_shift_std(tabular_data_labeled):
    """Dict shift_std shifts only the listed classes, one draw per class."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)])
    cls_col = Xc.shape[1] - 1
    classes = np.unique(np.asarray(y))
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = cls_col
    # fire every row (all classes), magnitudes only for the first class
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 1.0 for c in classes}}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    shift_std = {classes[0]: 0.5}
    Xt = generator.generate(Xc.copy(), selector=selector, shift_std=shift_std)
    fired_cls0 = Xc[:, cls_col] == classes[0]
    other = Xc[:, cls_col] != classes[0]
    assert not np.allclose(Xt[fired_cls0], Xc[fired_cls0])
    assert np.allclose(Xt[other], Xc[other])


def test_random_shift_classes__signed_shift_std_is_direction(tabular_data_labeled):
    """Negative shift_std entries drift the class in the OPPOSITE direction
    with the SAME magnitude: the mirror of the positive-std result (the rng
    draws are identical, only the sign differs)."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(np.asarray(y))
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = cls_col
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 1.0 for c in classes}}, rng=default_rng(0))
    Xp = Xc.copy()
    Xn = Xc.copy()
    gp = RandomShiftClassesGenerator(random_generator=default_rng(0))
    gn = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt_pos = gp.generate(Xp, selector=selector, shift_std={c: 4.0 for c in classes})
    Xt_neg = gn.generate(Xn, selector=selector, shift_std={c: -4.0 for c in classes})
    np.testing.assert_allclose(np.asarray(Xn), 2 * Xc - np.asarray(Xp))


def test_fixed_shift__deterministic_and_exact(tabular_data):
    """FixedShiftGenerator shifts each column by exactly shift_std, reproducibly."""
    X, y = tabular_data
    X_np = np.asarray(X, dtype=float)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    shift = np.linspace(1.0, -1.0, n_features)
    generator = FixedShiftGenerator()
    Xt1 = generator.generate(X=X_np.copy(), shift_std=shift)
    Xt2 = generator.generate(X=X_np.copy(), shift_std=shift)
    np.testing.assert_array_equal(Xt1, Xt2)  # deterministic
    delta = Xt1 - X_np
    expected = shift * X_np.std(axis=0, ddof=0)  # StandardScaler uses population std
    np.testing.assert_allclose(delta, np.broadcast_to(expected, delta.shape), atol=1e-8)


def test_fixed_shift_classes__deterministic_and_per_class(tabular_data_labeled):
    """FixedShiftClassesGenerator: same input -> same output; one exact shift per class."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    cls_col = Xc.shape[1] - 1
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = cls_col
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 1.0 for c in np.unique(y)}}, rng=default_rng(0))
    generator = FixedShiftClassesGenerator()
    Xt1 = generator.generate(X=Xc.copy(), selector=selector, shift=1.0)
    Xt2 = generator.generate(X=Xc.copy(), selector=selector, shift=1.0)
    np.testing.assert_array_equal(Xt1, Xt2)  # deterministic
    assert np.allclose(np.asarray(Xt1)[:, cls_col], Xc[:, cls_col])  # class col untouched
    # per-class shifts are exact: scalar shift 1.0 -> exactly 1 std per feature column
    delta = np.asarray(Xt1) - Xc
    for c in np.unique(y):
        rows = Xc[:, cls_col] == c
        expected = 1.0 * Xc[:, :-1].std(axis=0, ddof=0)
        np.testing.assert_allclose(
            delta[rows][:, :-1], np.broadcast_to(expected, delta[rows][:, :-1].shape), atol=1e-8
        )


def test_fixed_shift_classes__dict_shift_per_class(tabular_data_labeled):
    """Dict shift applies exact shifts only to the listed classes."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)]).astype(float)
    cls_col = Xc.shape[1] - 1
    classes = np.unique(np.asarray(y))
    selector = ClassSelector("cls", {f"f{i}": i for i in range(Xc.shape[1])} | {"cls": cls_col},
                             {"cls": {c: 1.0 for c in classes}}, rng=default_rng(0))
    generator = FixedShiftClassesGenerator()
    Xt = generator.generate(X=Xc.copy(), selector=selector, shift={classes[0]: 1.0})
    fired = Xc[:, cls_col] == classes[0]
    other = ~fired
    assert not np.allclose(np.asarray(Xt)[fired], Xc[fired])
    assert np.allclose(np.asarray(Xt)[other], Xc[other])