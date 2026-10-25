import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import ClassSelector
from badgers.generators.tabular_data.noise.noise import (
    GaussianNoiseClassesGenerator,
)


def _selector(y_values, proportions=None, rng=None):
    Xc = np.column_stack([np.zeros(len(y_values)), np.asarray(y_values)])
    props = proportions if proportions is not None else {c: 1.0 for c in np.unique(y_values)}
    return (
        ClassSelector([0], class_column=Xc.shape[1] - 1, proportions=props,
                      rng=rng or default_rng(0)),
        Xc,
    )


def test_gaussian_noise_classes__preserves_shape(tabular_data_labeled):
    Xc = np.asarray(tabular_data_labeled)
    y = Xc[:, -1]
    selector, Xc = _selector(y)
    noise_std_per_class = {label: 0.1 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator(random_generator=np.random.default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    assert len(Xc) == len(Xt)


def test_gaussian_noise_classes__modifies_data(tabular_data_labeled):
    """With non-zero noise_std, data should be modified."""
    Xc = np.asarray(tabular_data_labeled)
    y = Xc[:, -1]
    selector, Xc = _selector(y)
    noise_std_per_class = {label: 0.5 for label in np.unique(y)}
    generator = GaussianNoiseClassesGenerator(random_generator=np.random.default_rng(0))
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    assert not np.allclose(Xt, Xc)


def test_gaussian_noise_classes__per_class_magnitude(tabular_data_labeled):
    """Only the class listed in noise_std_per_class receives noise; a class
    value fired but absent from the dict raises."""
    Xc_full = np.asarray(tabular_data_labeled)
    y = Xc_full[:, -1]
    classes = np.unique(y)
    selector, Xc = _selector(y)
    cls_col = Xc.shape[1] - 1
    generator = GaussianNoiseClassesGenerator(random_generator=default_rng(0))
    noise_std_per_class = {classes[0]: 0.5}
    with pytest.raises(ValueError, match="no entry for the fired class"):
        generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    # with an entry for every fired class only the listed class is noised
    full = {c: 0.5 if c == classes[0] else 0.0 for c in classes}
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=full)
    fired_cls0 = Xc[:, cls_col] == classes[0]
    other = Xc[:, cls_col] != classes[0]
    assert not np.allclose(Xt[fired_cls0], Xc[fired_cls0])
    assert np.allclose(Xt[other], Xc[other])


def test_gaussian_noise_classes__int_keys_float_column(tabular_data_labeled):
    """Plain-int dict keys work for a float-encoded class column (dtype
    normalization)."""
    Xc_full = np.asarray(tabular_data_labeled)
    y = Xc_full[:, -1]
    selector, Xc = _selector(y)
    classes = np.unique(y)
    assert Xc[:, -1].dtype == float and set(classes) == {0.0, 1.0, 2.0, 3.0, 4.0}
    generator = GaussianNoiseClassesGenerator(random_generator=default_rng(0))
    noise_std_per_class = {int(c): 0.5 for c in classes}
    Xt = generator.generate(Xc.copy(), selector=selector, noise_std_per_class=noise_std_per_class)
    assert not np.allclose(Xt, Xc)


def test_gaussian_noise_classes__wrong_kind_keys_raise(tabular_data_labeled):
    """Keys of a wrong kind (strings for numeric classes) raise instead of
    silently skipping every class."""
    Xc_full = np.asarray(tabular_data_labeled)
    y = Xc_full[:, -1]
    selector, Xc = _selector(y)
    generator = GaussianNoiseClassesGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="cannot be compared"):
        generator.generate(
            Xc.copy(), selector=selector,
            noise_std_per_class={"class_0": 0.5, "class_1": 0.5,
                                 "class_2": 0.5, "class_3": 0.5})


def test_gaussian_noise_classes__selector_none_raises(tabular_data_labeled):
    Xc = np.asarray(tabular_data_labeled)
    generator = GaussianNoiseClassesGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="selector"):
        generator.generate(Xc.copy(), noise_std_per_class={0: 0.1})
