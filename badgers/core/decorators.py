import functools
from enum import Enum

import pandas as pd


class TabularDataType(Enum):
    NUMPY_ARRAY = 1
    PANDAS_DATAFRAME = 2


def numpy_API(generate_func):
    """
    Ensures X can be worked on using the numpy API (useful for indexing!).
    If X is an object that does not strictly follow the numpy API (like pandas.DataFrame),
    then it internally stores the metadata (like columns), casts X to a numpy array, calls the generate function,
    and finally restore and restores the original type.

    @TODO check y too!
    """

    @functools.wraps(generate_func)
    def wrapper(self, X, y, **params):
        X_data_type = None
        if isinstance(X, pd.DataFrame):
            # when X is a pandas DataFrame, then locally save the columns and make X a numpy array
            X_data_type = TabularDataType.PANDAS_DATAFRAME
            columns = X.columns
            X = X.to_numpy()
        # call to generate function
        Xt, yt = generate_func(self, X, y, **params)
        if X_data_type == TabularDataType.PANDAS_DATAFRAME:
            Xt = pd.DataFrame(Xt, columns=columns)
        return Xt, yt

    return wrapper
