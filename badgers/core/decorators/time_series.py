import functools

import numpy as np
import pandas as pd


def preprocess_inputs(generate_func):
    """
    Validates and convert X and y for time series data generators

    Preprocessing:
    X is converted to a pandas DataFrame
    y is converted to a pandas Series
    """

    @functools.wraps(generate_func)
    def wrapper(self, X, y, **kwargs):
        # Validate and preprocess X
        if isinstance(X, list):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            # if it is a numpy array first check the dimensionality,
            # if dimension is 1 then reshape, if the dimension > 2 then raise error
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.ndim > 2:
                raise ValueError(
                    "X has more than 2 dimensions where it is expected to have either 1 or 2!"
                )
            X = pd.DataFrame(data=X)
        elif isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            # do nothing here
            pass
        else:
            raise ValueError(f"X must be a list, numpy array, pandas Series, or pandas DataFrame\nX is: {type(X)}")

        # Validate and preprocess y
        if y is not None:
            if isinstance(y, list):
                y = pd.Series(y)
            elif isinstance(y, np.ndarray):
                y = pd.Series(y)
            elif isinstance(y, pd.Series):
                pass
            else:
                raise ValueError("y must be a list, numpy array, or pandas Series")

        # Call the original function with the preprocessed inputs
        return generate_func(self, X, y, **kwargs)

    return wrapper
