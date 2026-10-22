"""Value-driven selector presets: rules that read column values.

These presets READ values from ``X``, so they are constructed with a
``column_mapping`` binding variable names to column positions. The
deterministic ones need no rng; ``ClassSelector`` draws without replacement
and therefore binds one.
"""

from collections.abc import Sequence

import numpy as np

from badgers.core.utils import normalize_proba

from .base import _require_rng, _validate_column_mapping, _validate_rate, _validate_variables

__all__ = [
    "TopQuantileSelector",
    "ClassSelector",
    "ProportionalSelector",
]


class TopQuantileSelector:
    """Selector: fires rows above the ``quantile`` of a column.

    Deterministic, value-driven rule: for each of the selector's variables
    the threshold
    is ``quantile(col, quantile)`` and a row fires when the compared value
    is ``>= threshold``. E.g. ``quantile=0.75`` fires every row whose value
    is at or above the third quartile (Q3) -- "I keep everything above
    Q3". Like the other presets, the class name describes the RULE
    (quantile thresholding), never the deficit that follows.

    Two modes:

    - **self** (``drivers=None``): each variable fires rows where its OWN
      value is above its own ``quantile``.
    - **driver** (``drivers=...``): each variable fires rows where a DRIVER
      column's value is above the driver's ``quantile``. A single string is
      broadcast to all variables (one driver decides everything); a sequence
      gives one driver per variable (same length as ``variables``).

    Because the rule is deterministic, no ``rng`` is needed.

    Parameters
    ----------
    variables : str or sequence of str
        Names of the variables whose affected rows the selector decides
        (the returned dict's keys).
    column_mapping : dict mapping str -> int
        Maps each variable/driver name to its column index in ``X``.
        Required because this preset READS values (unlike the other
        presets, which are value-blind).
    quantile : float, in [0, 1]
        The quantile of the (driver) column that forms the threshold.
        The comparison is ``>=``, so with ``quantile=1.0`` rows equal to
        the column maximum still fire. Baked at construction: a configured
        selector is self-contained.
    drivers : str, sequence of str, or None, default=None
        The driver columns as described above.

    Calling
    -------
    ``__call__(X)``: one mask per selector variable; a row fires
    when the (driver) column's value is ``>=`` the column's quantile
    threshold computed on the passed rows.
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        column_mapping: dict[str, int],
        quantile: float,
        drivers: str | Sequence[str] | None = None,
    ):
        _validate_rate(quantile, "quantile")
        self.quantile = quantile
        self.variables = _validate_variables(variables)
        _validate_column_mapping(self.variables, column_mapping)
        if isinstance(drivers, str):
            drivers = (drivers,) * len(self.variables)
        elif drivers is not None:
            drivers = tuple(drivers)
            if len(drivers) != len(self.variables):
                raise ValueError(
                    f"drivers must be a single str or one driver per "
                    f"variable ({len(self.variables)}), got {len(drivers)}"
                )
        if drivers is not None:
            for name in drivers:
                if name not in column_mapping:
                    raise ValueError(
                        f"driver {name!r} not in column_mapping"
                    )
        self.column_mapping = column_mapping
        self.drivers = drivers

    def __call__(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        masks = {}
        for name, driver in zip(self.variables, self.drivers or self.variables):
            dcol = self.column_mapping[driver]
            threshold = np.quantile(X[:, dcol], self.quantile)
            masks[name] = X[:, dcol] >= threshold
        return masks

    def __repr__(self) -> str:
        return (
            f"TopQuantileSelector(variables={self.variables!r}, "
            f"quantile={self.quantile!r}, "
            f"drivers={self.drivers!r})"
        )


class ClassSelector:
    """Selector: fires a fixed share of the rows of each class of a column.

    For each selector variable (a column of ``X`` intended to hold DISCRETE
    values -- classes), the distinct values are discovered from ``X`` and
    for every class value listed in ``proportions`` exactly
    ``min(int(prop * n_rows), n_rows_in_class)`` rows OF THAT CLASS are
    drawn WITHOUT REPLACEMENT (uniform) -- the proportion is a share of
    the TOTAL row count, clipped to the rows available in the class.
    Class values present in the column but absent from the
    mapping are NOT fired (kept as-is) -- partial mappings are legitimate
    ("only rarefy the rare class"). The FIRE INTENSITY (the proportions)
    is a constructor argument: a sweep means constructing a new selector.

    Like the other presets, the class name describes the RULE (per-class
    fixed-count sampling), never the deficit that follows. The consuming
    generator decides what firing means -- e.g. the imbalance generators
    DROP the fired rows (subsampling deficit).

    Parameters
    ----------
    variables : str or sequence of str
        Names of the class columns whose affected rows this selector
        decides (the returned dict's keys). Each must be a key of
        ``column_mapping``.
    column_mapping : dict mapping str -> int
        Maps each variable name to its column index in ``X`` (standard
        binding; this preset READS values).
    proportions : dict mapping str -> dict
        For each variable name, a mapping ``{class_value: proportion}``
        with proportions in ``[0, 1]``. Proportions are shares of the
        INPUT row count. A class value in the mapping that is absent from
        the column raises ``ValueError`` (typo protection).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws. The consuming generator binds
        its own generator here (single RNG stream). Calling without a
        bound generator raises ``ValueError``.

    Calling
    -------
    ``__call__(X)``: one mask per class column; a row fires if it
    belongs to a fired class. With several class columns the masks are
    INDEPENDENT per column -- consumers typically OR them (a row dropped
    if fired in any column). Selection data, including class columns,
    belongs in ``X``; selectors do not accept a separate target array.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float).reshape(-1, 1)
    >>> mapping = {"cls": 0}
    >>> selector = ClassSelector("cls", mapping, {"cls": {0: 2 / 3}},
    ...                          rng=np.random.default_rng(0))
    >>> mask = selector(X)["cls"]
    >>> int(mask.sum())
    2
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        column_mapping: dict[str, int],
        proportions: dict[str, dict],
        rng: np.random.Generator | None = None,
    ):
        self.variables = _validate_variables(variables)
        _validate_column_mapping(self.variables, column_mapping)
        for name in self.variables:
            if name not in proportions:
                raise ValueError(
                    f"variable {name!r} has no entry in proportions"
                )
            for class_value, prop in proportions[name].items():
                _validate_rate(prop, f"proportions[{name!r}][{class_value!r}]")
        self.column_mapping = column_mapping
        self.proportions = proportions
        self.rng = rng

    def __call__(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        _require_rng(self)
        n_rows = X.shape[0]
        masks = {}
        for name in self.variables:
            col = X[:, self.column_mapping[name]]
            values = set(np.unique(col).tolist())
            unknown = [v for v in self.proportions[name] if v not in values]
            if unknown:
                raise ValueError(
                    f"class value(s) {unknown!r} not found in column "
                    f"{name!r} (found {sorted(values)!r})"
                )
            mask = np.zeros(n_rows, dtype=bool)
            for class_value, prop in self.proportions[name].items():
                rows = np.flatnonzero(col == class_value)
                # Shares are of the TOTAL row count; clipped to the number
                # of rows actually in the class (a mask cannot fire a row
                # twice, so no replacement is possible).
                count = min(int(prop * n_rows), len(rows))
                # Sampling without replacement: sequential by nature -- the
                # documented exception to the vectorization rule.
                idx = self.rng.choice(rows, size=count, replace=False)
                mask[idx] = True
            masks[name] = mask
        return masks

    def __repr__(self) -> str:
        return (
            f"ClassSelector(variables={self.variables!r}, "
            f"proportions={self.proportions!r})"
        )


class ProportionalSelector:
    """Selector: the deficit probability is proportional to a column's value.

    Value-driven, FRACTIONAL rule: for each of the selector's variables the
    per-row
    deficit probability is derived from the column's values, rescaled to
    ``[0, 1]`` (min-max) and then normalized to sum to 1. A constant
    column yields the uniform probability. Deterministic (no ``rng``); the
    consumer decides how the probabilities are realized (weights in a
    weighted draw, Bernoulli firing, ...). "Proportional" describes the
    RULE, never the deficit that follows.

    Parameters
    ----------
    variables : str or sequence of str
        Names of the variables whose affected rows the selector decides
        (the returned dict's keys). Each must be a key of ``column_mapping``.
    column_mapping : dict mapping str -> int
        Maps each variable name to its column index in ``X`` (standard
        binding; this preset READS values).

    Calling
    -------
    ``__call__(X)``: one float score array per variable (the
    normalized column values, summing to 1).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1.0, 2.0, 4.0]).reshape(-1, 1)
    >>> ProportionalSelector("f0", {"f0": 0})(X)["f0"]
    array([0.        , 0.33333333, 0.66666667])
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        column_mapping: dict[str, int],
    ):
        self.variables = _validate_variables(variables)
        _validate_column_mapping(self.variables, column_mapping)
        self.column_mapping = column_mapping

    def __call__(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        scores = {}
        for name in self.variables:
            col = X[:, self.column_mapping[name]].astype(float)
            span = col.max() - col.min()
            unit = (col - col.min()) / span if span > 0 else np.ones_like(col)
            scores[name] = normalize_proba(unit)
        return scores

    def __repr__(self) -> str:
        return (
            f"ProportionalSelector(variables={self.variables!r})"
        )
