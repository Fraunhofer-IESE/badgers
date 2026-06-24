import numpy as np
from numpy.random import default_rng

from badgers.generators.tabular_data.drift import RandomShiftGenerator, RandomShiftClassesGenerator


def test_random_shift__preserves_shape_number(tabular_data):
    """RandomShiftGenerator with scalar shift_std preserves input shape."""
    X, y = tabular_data
    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=0.1)
    assert len(X) == len(Xt)


def test_random_shift__preserves_shape_array(tabular_data):
    """RandomShiftGenerator with array shift_std preserves input shape."""
    X, y = tabular_data
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    shift_std = np.array([0.1]) if n_features == 1 else np.linspace(0.1, 1, n_features)

    generator = RandomShiftGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)


def test_random_shift_classes__preserves_shape_number(tabular_data_labeled):
    """RandomShiftClassesGenerator with scalar shift_std preserves input shape."""
    X, y = tabular_data_labeled
    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt, yt = generator.generate(X.copy(), y, shift_std=0.1)
    assert len(X) == len(Xt)


def test_random_shift_classes__preserves_shape_array(tabular_data_labeled):
    """RandomShiftClassesGenerator with array shift_std preserves input shape."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    n_classes = len(np.unique(y))
    shift_std = 0.1 if n_features == 1 else np.linspace(0.1, 1, n_classes)

    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)


def test_random_shift_classes__preserves_shape_2d_array(tabular_data_labeled):
    """RandomShiftClassesGenerator with 2D shift_std array preserves input shape."""
    X, y = tabular_data_labeled
    X_np = np.asarray(X)
    if X_np.ndim == 1 or X_np.shape[1] < 2:
        return

    n_features = X_np.shape[1]
    shift_std = np.array([[i] * n_features for i in np.linspace(0.1, 1, 5)])

    generator = RandomShiftClassesGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, shift_std=shift_std)
    assert len(X) == len(Xt)