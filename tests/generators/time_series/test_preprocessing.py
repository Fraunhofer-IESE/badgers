import numpy as np
import pandas as pd
import pytest

from badgers.core.decorators.time_series import preprocess_inputs


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
