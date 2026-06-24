import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["numpy_1D", "numpy_2D", "pandas_1D", "pandas_2D"])
def tabular_data(request, rng):
    """Parametrized fixture yielding (X, y=None) for each input type."""
    if request.param == "numpy_1D":
        return rng.normal(size=100).reshape(-1, 1), None
    elif request.param == "numpy_2D":
        return rng.normal(size=(100, 10)), None
    elif request.param == "pandas_1D":
        return pd.DataFrame(
            rng.normal(size=100).reshape(-1, 1), columns=["col"]
        ), None
    elif request.param == "pandas_2D":
        return pd.DataFrame(
            rng.normal(size=(100, 10)),
            columns=[f"col{i}" for i in range(10)],
        ), None


@pytest.fixture(params=["numpy_1D", "numpy_2D", "pandas_1D", "pandas_2D"])
def tabular_data_labeled(request, rng):
    """Parametrized fixture yielding (X, y) with 5-class classification labels."""
    if request.param == "numpy_1D":
        return (
            rng.normal(size=100).reshape(-1, 1),
            np.array([i % 5 for i in range(100)]),
        )
    elif request.param == "numpy_2D":
        return (
            rng.normal(size=(100, 10)),
            np.array([0, 1, 2, 3, 4] * 20),
        )
    elif request.param == "pandas_1D":
        return (
            pd.DataFrame(
                rng.normal(size=100).reshape(-1, 1), columns=["col"]
            ),
            pd.Series([i % 5 for i in range(100)]),
        )
    elif request.param == "pandas_2D":
        return (
            pd.DataFrame(
                rng.normal(size=(100, 10)),
                columns=[f"col{i}" for i in range(10)],
            ),
            pd.Series([0, 1, 2, 3, 4] * 20),
        )
