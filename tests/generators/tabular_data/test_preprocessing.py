import numpy as np
import pandas as pd
import pytest

from badgers.core.decorators.tabular_data import preprocess_inputs


class XOnlyGenerator:
    @preprocess_inputs
    def generate(self, X, **kwargs):
        return X, kwargs


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        (np.array([1, 2]), [[1], [2]]),
        (pd.Series([1, 2]), [[1], [2]]),
        (pd.DataFrame({"a": [1, 2], "b": [3, 4]}), [[1, 3], [2, 4]]),
    ],
)
def test_preprocess_inputs_forwards_only_preprocessed_X(X, expected):
    result, kwargs = XOnlyGenerator().generate(X, option="kept")

    np.testing.assert_array_equal(result, expected)
    assert result.ndim == 2
    assert kwargs == {"option": "kept"}


def test_preprocess_inputs_makes_read_only_X_writable():
    X = np.array([[1, 2], [3, 4]])
    X.setflags(write=False)

    result, _ = XOnlyGenerator().generate(X)

    assert result.flags.writeable
    assert not X.flags.writeable


def test_preprocess_inputs_rejects_more_than_two_dimensions():
    X = np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="more than 2 dimensions"):
        XOnlyGenerator().generate(X)


def test_preprocess_inputs_rejects_unsupported_X_type():
    with pytest.raises(ValueError, match="X must be a list"):
        XOnlyGenerator().generate("not tabular data")


class _FakeGen:
    @preprocess_inputs
    def generate(self, X, selector=None, value=0.0):
        return X, selector, value


def test_preprocess_accepts_dataframe_and_keeps_positional_args():
    gen = _FakeGen()
    X, selector, value = gen.generate(pd.DataFrame({"a": [1.0, 2.0]}), "SELECTOR", 7.5)
    assert isinstance(X, np.ndarray)
    assert selector == "SELECTOR"
    assert value == 7.5


def test_preprocess_forwards_kwargs_and_readonly_copy():
    gen = _FakeGen()
    ro = np.arange(4).reshape(2, 2)
    ro.flags.writeable = False
    X, selector, value = gen.generate(ro, selector="S", value=1.5)
    assert X.flags.writeable
    assert (X == np.arange(4).reshape(2, 2)).all()
    assert selector == "S" and value == 1.5
