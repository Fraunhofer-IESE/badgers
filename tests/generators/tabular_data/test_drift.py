import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import ClassSelector
from badgers.generators.tabular_data.drift.drift import RandomShiftGenerator, RandomShiftClassesGenerator


def test_random_shift__preserves_shape_and_modifies_data(tabular_data):
    """RandomShiftGenerator with scalar shift_std preserves shape and modifies data."""
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=0.1)
    assert len(X) == len(Xt)
    X_np = np.asarray(X)
    Xt_np = np.asarray(Xt)
    assert not np.allclose(X_np, Xt_np)


def test_random_shift__zero_shift_preserves_data(tabular_data):
    """RandomShiftGenerator with shift_std=0 produces identical output."""
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=0.0)
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
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)
    assert not np.allclose(X_np, np.asarray(Xt))


def test_random_shift_classes__preserves_shape_and_modifies_data(tabular_data_labeled):
    """RandomShiftClassesGenerator with scalar shift_std preserves shape and modifies data."""
    X, y = tabular_data_labeled
    Xc = np.column_stack([np.asarray(X), np.asarray(y)])
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = Xc.shape[1] - 1
    selector = ClassSelector("cls", column_mapping, {"cls": {c: 0.5 for c in np.unique(y)}}, rng=default_rng(0))
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(Xc.copy(), y, selector=selector, shift_std=0.1)
    assert len(Xc) == len(Xt)
    assert not np.allclose(np.asarray(Xc), np.asarray(Xt))


def test_random_shift_classes__selector_none_raises(tabular_data_labeled):
    """A per-class rule has no class-free default: selector=None raises."""
    X, y = tabular_data_labeled
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="selector"):
        generator.generate(X.copy(), y, shift_std=0.1)


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
    Xt, _ = generator.generate(Xc.copy(), y, selector=selector, shift_std=shift_std)
    fired_cls0 = Xc[:, cls_col] == classes[0]
    other = Xc[:, cls_col] != classes[0]
    assert not np.allclose(Xt[fired_cls0], Xc[fired_cls0])
    assert np.allclose(Xt[other], Xc[other])