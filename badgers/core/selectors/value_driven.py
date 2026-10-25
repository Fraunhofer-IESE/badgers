"""Value-driven selector presets: rules that read column values.

These presets READ values from ``X``, so they are constructed with the
read column position(s) as bare ints. The column positions they COVER
(``columns``) are their scope: the returned dict has one key per target.
"""

from collections.abc import Sequence

import numpy as np

from badgers.core.utils import normalize_proba

from .base import (
    _require_rng,
    _resolve_targets,
    _validate_columns,
    _validate_rate,
)

__all__ = [
    "TopQuantileSelector",
    "ClassSelector",
    "ClassCountSelector",
    "ProportionalSelector",
]


class TopQuantileSelector:
    """Selector: fires rows where the column value is >= its quantile.

    Deterministic rule: threshold = ``np.quantile(X[:, driver], quantile)``
    (``driver=None``: the column's own threshold). A row fires when the
    driver value is ``>= threshold``; the same mask is returned under
    every target.

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    column : int
        Required read column; unused when ``driver`` is set. In self
        mode (``driver=None``) the threshold is compared against this
        column (``column == driver``).
    quantile : float, in [0, 1]
        The quantile forming the threshold (``>=`` comparison).
    driver : int or None, default=None
        Driver column position whose values form the threshold and mask.
        ``None`` = self mode (driver is ``column``).
    """

    def __init__(self, columns: int | Sequence[int], column: int,
                 quantile: float, driver: int | None = None):
        self.columns = _validate_columns(columns)
        if not isinstance(column, (int, np.integer)):
            raise ValueError(f"column must be an int position, got {column!r}")
        self.column = int(column)
        self.driver = self.column if driver is None else int(driver)
        self.quantile = _validate_rate(quantile, "quantile")

    def __call__(self, X: np.ndarray, targets=None):
        targets = _resolve_targets(self, X, targets)
        dcol = X[:, self.driver]
        threshold = np.quantile(dcol, self.quantile)
        mask = dcol >= threshold
        return {col: mask for col in targets}

    def __repr__(self) -> str:
        return (f"TopQuantileSelector(columns={self.columns!r}, "
                f"column={self.column!r}, quantile={self.quantile!r}, "
                f"driver={self.driver!r})")


class ClassSelector:
    """Selector: fires a fixed share of the rows of each listed class.

    Reads the class labels from ``class_column`` ONCE; for every class
    value in ``proportions`` exactly ``min(int(share * n_rows),
    n_rows_in_class)`` rows of that class are drawn WITHOUT REPLACEMENT
    (uniform). The class values are discovered from ``X``; a proportion
    for an absent class raises ``ValueError`` (typo protection). The
    resulting mask is returned under EVERY target column (one read,
    broadcast across targets).

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    class_column : int
        Position of the column holding the class labels.
    proportions : dict mapping class_value -> float in [0, 1]
        Share of the TOTAL row count fired per class value.
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws.
    """

    def __init__(self, columns: int | Sequence[int], class_column: int,
                 proportions: dict, rng: np.random.Generator | None = None):
        self.columns = _validate_columns(columns)
        if not isinstance(class_column, (int, np.integer)):
            raise ValueError(
                f"class_column must be an int position, got {class_column!r}")
        self.class_column = int(class_column)
        for value, share in proportions.items():
            _validate_rate(share, f"proportions[{value!r}]")
        self.proportions = proportions
        self.rng = rng

    def __call__(self, X: np.ndarray, targets=None):
        _require_rng(self)
        if not (0 <= self.class_column < X.shape[1]):
            raise ValueError(
                f"the selector's class_column {self.class_column} is not a "
                f"valid column position in X with {X.shape[1]} columns")
        targets = _resolve_targets(self, X, targets)
        n_rows = X.shape[0]
        col = X[:, self.class_column]
        values = set(np.unique(col).tolist())
        unknown = [v for v in self.proportions if v not in values]
        if unknown:
            raise ValueError(
                f"class value(s) {unknown!r} not found in column "
                f"{self.class_column} (found {sorted(values)!r})")
        mask = np.zeros(n_rows, dtype=bool)
        for class_value, share in self.proportions.items():
            rows = np.flatnonzero(col == class_value)
            # Share of the TOTAL row count, clipped to the class size.
            count = min(int(share * n_rows), len(rows))
            if count:
                # Sampling without replacement: sequential by nature.
                idx = self.rng.choice(rows, size=count, replace=False)
                mask[idx] = True
        return {col: mask for col in targets}

    def __repr__(self) -> str:
        return (f"ClassSelector(columns={self.columns!r}, "
                f"class_column={self.class_column!r}, "
                f"proportions={self.proportions!r})")


class ClassCountSelector:
    """Selector: exactly ``counts[class_value]`` rows per listed class.

    The value-driven exact-count preset (the class-aware sibling of
    :class:`~badgers.core.selectors.value_free.CountSelector`). Reads the
    class labels from ``class_column`` ONCE; for every class value in
    ``counts`` exactly that many rows of the class are drawn WITHOUT
    REPLACEMENT (uniform) and the union mask is returned under every
    target. Classes absent from ``counts`` fire nothing; a count for a
    class value missing from ``X`` raises (typo protection), and a count
    above the class size raises (an explicit count is a request, not a
    share -- unlike :class:`ClassSelector`'s share clipping).

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    class_column : int
        Position of the column holding the class labels.
    counts : dict mapping class_value -> int
        Exact number of rows fired per class value (ints >= 0).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the row draws.
    """

    def __init__(self, columns: int | Sequence[int], class_column: int,
                 counts: dict, rng: np.random.Generator | None = None):
        self.columns = _validate_columns(columns)
        if not isinstance(class_column, (int, np.integer)):
            raise ValueError(
                f"class_column must be an int position, got {class_column!r}")
        self.class_column = int(class_column)
        for value, count in counts.items():
            if not isinstance(count, (int, np.integer)) or isinstance(count, bool):
                raise ValueError(
                    f"counts[{value!r}] must be an int, got {count!r}")
            if count < 0:
                raise ValueError(f"counts[{value!r}] must be >= 0, got {count}")
        self.counts = counts
        self.rng = rng

    def __call__(self, X: np.ndarray, targets=None):
        _require_rng(self)
        targets = _resolve_targets(self, X, targets)
        n_rows = X.shape[0]
        labels = X[:, self.class_column]
        values = set(np.unique(labels).tolist())
        unknown = [v for v in self.counts if v not in values]
        if unknown:
            raise ValueError(
                f"class value(s) {unknown!r} not found in column "
                f"{self.class_column} (found {sorted(values)!r})")
        mask = np.zeros(n_rows, dtype=bool)
        for class_value, count in self.counts.items():
            rows = np.flatnonzero(labels == class_value)
            if count > len(rows):
                raise ValueError(
                    f"counts[{class_value!r}]={count} exceeds the number "
                    f"of rows of class {class_value!r} ({len(rows)})")
            if count:
                # Sampling without replacement: sequential by nature.
                idx = self.rng.choice(rows, size=count, replace=False)
                mask[idx] = True
        return {t: mask for t in targets}

    def __repr__(self) -> str:
        return (f"ClassCountSelector(columns={self.columns!r}, "
                f"class_column={self.class_column!r}, "
                f"counts={self.counts!r})")


class ProportionalSelector:
    """Selector: the deficit probability is proportional to a column's value.

    Value-driven, FRACTIONAL rule: the per-row score is the min-max
    normalized value of ``column`` rescaled to sum to 1 (a constant
    column yields the uniform score). Deterministic (no rng); the
    consumer decides how the probabilities are realized.

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    column : int
        Position of the column whose values drive the scores.
    """

    def __init__(self, columns: int | Sequence[int], column: int):
        self.columns = _validate_columns(columns)
        if not isinstance(column, (int, np.integer)):
            raise ValueError(f"column must be an int position, got {column!r}")
        self.column = int(column)

    def __call__(self, X: np.ndarray, targets=None):
        targets = _resolve_targets(self, X, targets)
        col = X[:, self.column].astype(float)
        span = col.max() - col.min()
        unit = (col - col.min()) / span if span > 0 else np.ones_like(col)
        scores = normalize_proba(unit)
        return {c: scores for c in targets}

    def __repr__(self) -> str:
        return (f"ProportionalSelector(columns={self.columns!r}, "
                f"column={self.column!r})")
