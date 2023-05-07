import functools

import pandas as pd


def input_checker(transform_func):
    """
    A decorator that checks whether the input is a pd.DataFrame
    and if so recreate a pd.DataFrame after the transformation if necessary.

    Does not perform any input validation! This is left to the transformer itself

    :param transform_func: the transform function that is supposed to be decorated
    :return: the transformed X {np.array|pd.DataFrame}
    """
    @functools.wraps(transform_func)
    def checker(X):
        cols = None
        input_type = None
        if isinstance(X, pd.DataFrame):
            cols = X.columns
            input_type = 'pandas'
        X = transform_func(X)
        if input_type == 'pandas' and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X, columns=cols)
        return X

    return checker
