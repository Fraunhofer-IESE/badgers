"""Functional checks applicable to all data types."""
import numpy as np
import pandas as pd
from benchmarks.models import FunctionalCheck


def _check_same_shape(X, y, Xt, yt, params):
    if hasattr(X, "shape") and hasattr(Xt, "shape"):
        return X.shape == Xt.shape
    if isinstance(X, list) and isinstance(Xt, list):
        return len(X) == len(Xt)
    return True


def _check_no_nans(X, y, Xt, yt, params):
    if isinstance(Xt, pd.DataFrame):
        return not Xt.isna().any().any()
    if isinstance(Xt, np.ndarray):
        return not np.isnan(Xt).any()
    return True


CHECK_SAME_SHAPE = FunctionalCheck(
    name="same_shape",
    description="Output has the same shape as input",
    check=_check_same_shape,
)

CHECK_NO_NANS = FunctionalCheck(
    name="no_nans",
    description="Output contains no NaN values (unless generating missingness)",
    check=_check_no_nans,
)