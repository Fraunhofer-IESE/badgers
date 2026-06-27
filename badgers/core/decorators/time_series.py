import functools

import numpy as np
import pandas as pd


def preprocess_inputs(generate_func):
    """
    Validates and converts X and y for time series data generators.

    Preprocessing:
    X is converted to a 2D numpy array.
    y is converted to a 1D numpy array (or left as None).
    """

    @functools.wraps(generate_func)
    def wrapper(self, X, y, **kwargs):
        # Validate and preprocess X
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError(
                f"X must be a list, numpy array, pandas Series, or pandas DataFrame\n"
                f"X is: {type(X)}"
            )

        # Validate dimensionality of X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim > 2:
            raise ValueError(
                "X has more than 2 dimensions where it is expected to have either 1 or 2!"
            )

        # Validate and preprocess y
        if y is not None:
            if isinstance(y, list):
                y = np.array(y)
            elif isinstance(y, pd.Series):
                y = y.values
            elif isinstance(y, pd.DataFrame):
                y = y.values.ravel()
            elif isinstance(y, np.ndarray):
                pass
            else:
                raise ValueError(
                    f"y must be a list, numpy array, pandas Series, or pandas DataFrame\n"
                    f"y is: {type(y)}"
                )

        # Call the original function with the preprocessed inputs
        return generate_func(self, X, y, **kwargs)

    return wrapper
