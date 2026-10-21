import functools

import numpy as np
import pandas as pd


def preprocess_inputs(generate_func):
    """
    Validates and converts X for tabular data generators.

    Preprocessing:
    X is converted to a 2D numpy array.
    """

    @functools.wraps(generate_func)
    def wrapper(self, X, **kwargs):
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

        # Ensure X is writable: pandas 3.0 (Copy-on-Write) and some inputs
        # yield read-only arrays, but generators mutate X in place.
        if not X.flags.writeable:
            X = X.copy()

        # Validate dimensionality of X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim > 2:
            raise ValueError(
                "X has more than 2 dimensions where it is expected to have either 1 or 2!"
            )

        # Call the original function with the preprocessed input
        return generate_func(self, X, **kwargs)

    return wrapper
