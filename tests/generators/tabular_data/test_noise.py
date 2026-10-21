import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import ClassSelector
from badgers.generators.tabular_data.noise.noise import (
    GaussianNoiseClassesGenerator,
)


def _selector(y_values, proportions=None, rng=None):
    Xc = np.column_stack([np.zeros(len(y_values)), np.asarray(y_values)])
    column_mapping = {f"f{i}": i for i in range(Xc.shape[1])}
    column_mapping["cls"] = Xc.shape[1] - 1
    props = proportions if proportions is not None else {c: 1.0 for c in np.unique(y_values)}
    return ClassSelector("cls", column_mapping, {"cls": props}, rng=rng or default_rng(0)), Xc


def test_gaussian_noise_classes__preserves_shape(tabular_data_labeled):
    X, y = tabular_data_labeled
    selector, Xc = _selector(y)
    noise_std_per_class = {label: 0.1 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator()
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    assert len(Xc) == len(Xt)


def test_gaussian_noise_classes__modifies_data(tabular_data_labeled):
    """With non-zero noise_std, data should be modified."""
    X, y = tabular_data_labeled
    selector, Xc = _selector(y)
    noise_std_per_class = {label: 0.5 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator()
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    assert not np.allclose(Xt, Xc)


def test_gaussian_noise_classes__per_class_magnitude(tabular_data_labeled):
    """Only classes listed in noise_std_per_class receive noise."""
    X, y = tabular_data_labeled
    classes = np.unique(np.asarray(y))
    selector, Xc = _selector(y)
    cls_col = Xc.shape[1] - 1
    generator = GaussianNoiseClassesGenerator(random_generator=default_rng(0))
    noise_std_per_class = {classes[0]: 0.5}
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    fired_cls0 = Xc[:, cls_col] == classes[0]
    other = Xc[:, cls_col] != classes[0]
    assert not np.allclose(Xt[fired_cls0], Xc[fired_cls0])
    assert np.allclose(Xt[other], Xc[other])


def test_gaussian_noise_classes__selector_none_raises(tabular_data_labeled):
    X, y = tabular_data_labeled
    generator = GaussianNoiseClassesGenerator()
    with pytest.raises(ValueError, match="selector"):
        generator.generate(X.copy(), noise_std_per_class={0: 0.1})
