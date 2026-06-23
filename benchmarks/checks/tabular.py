"""Functional checks for tabular data generators."""
import numpy as np
from benchmarks.models import FunctionalCheck


def _check_increased_variance(X, y, Xt, yt, params):
    var_X = np.var(X, axis=0).mean()
    var_Xt = np.var(Xt, axis=0).mean()
    return var_Xt > var_X


def _check_outlier_count(X, y, Xt, yt, params):
    n_outliers = params.get("n_outliers", 0)
    return Xt.shape[0] == n_outliers


CHECK_INCREASED_VARIANCE = FunctionalCheck(
    name="increased_variance",
    description="Output has greater variance than input (noise was added)",
    check=_check_increased_variance,
)

CHECK_OUTLIER_COUNT = FunctionalCheck(
    name="outlier_count",
    description="Correct number of outliers were generated",
    check=_check_outlier_count,
)