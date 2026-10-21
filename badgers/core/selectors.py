"""Selectors: the observation-side contract for selecting deficit rows.

A SELECTOR decides which rows of the input get a data quality deficit

    Selector = Callable[[X, y], dict[str, bool-mask | score]]

- keys   = variable names (the subscript in ``R_X``: which column the mask gates)
- values = per-row boolean masks or float SCORES, ``len(mask) == X.shape[0]``
- polarity: ``True``/``1.0`` = ``R_X = 1`` = deficit applied; ``False``/``0.0`` = record as-is

Boolean masks are the common case, but a selector may return FLOAT scores in
``[0, 1]``: the PROBABILITY that the deficit is applied to that row (the
deterministic ``1.0``/``0.0``/``bool`` values being the degenerate cases). A
selector itself never draws randomness for these rates -- realization belongs
to the consuming generator, which binds a single rng stream: a GATING consumer
(noise, missingness) realizes ``p = 0.3`` as a Bernoulli firing; a RESAMPLING
consumer (imbalance) uses it as a selection weight. Being in ``core``, the
module is modality-agnostic: time-series / graph / text generators can reuse
the same contract with masks over their own axes.

The presets are small callable objects so their configuration (which
variables, which rng, which column mapping, which intensity) stays
introspectable (``repr`` and stored attributes). The FIRE INTENSITY is part
of the preset configuration where it exists: ``rate`` for the stochastic
presets (``BernoulliSelector``/``ExactCountSelector``), ``quantile`` for
the deterministic ``TopQuantileSelector``, and ``proportions`` for the
per-class ``ClassSelector`` are constructor arguments, so an
intensity sweep means constructing a new selector (nothing is swept on the
call). The trivial ``AlwaysSelector`` has no intensity knob at all. There
is deliberately no
base class and no
registry -- conformance to the :class:`Selector` protocol is defined by the
call signature alone. Each
preset is self-contained: its mask logic is inlined in ``__call__`` (no free
helper functions), so a preset depends on nothing beyond numpy and the
consuming generators simply construct one and call it.

GATING consumers preserve the input's shape and only ALTER the fired cells
(missingness, noise, outliers, ...); SUBSAMPLING consumers (imbalance) DROP
the fired rows, so their output is smaller than the input. The contract
itself is only "which rows fire".
"""

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from badgers.core.utils import normalize_proba

__all__ = [
    "Selector",
    "AlwaysSelector",
    "BernoulliSelector",
    "ExactCountSelector",
    "TopQuantileSelector",
    "ClassSelector",
    "ProportionalSelector",
]


class AlwaysSelector:
    """Selector: every row fires, for every gated variable.

    The trivial, value- and rng-free rule — the degenerate gate "apply the
    deficit everywhere". The default when a causal generator receives
    ``selector=None`` (noise, drift, edge perturbation).

    Parameters
    ----------
    variables : str or sequence of str
        Names of the gated variables (the returned dict's keys).

    Calling
    -------
    ``__call__(X, y=None)``: returns an all-True mask per variable. No
    intensity kwarg — there is nothing to sweep (any intensity knob lives
    in the DEFICIT itself, e.g. noise_std).
    """

    def __init__(self, variables: str | Sequence[str]):
        self.variables = _validate_variables(variables)

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        return {
            name: np.ones(X.shape[0], dtype=bool)
            for name in self.variables
        }

    def __repr__(self) -> str:
        return f"AlwaysSelector(variables={self.variables!r})"


@runtime_checkable
class Selector(Protocol):
    """Protocol for selectors: callables producing per-variable deficit masks.

    Implementations (preset objects or user callables) must accept
    ``(X, y=None)`` and return a ``dict[str, numpy.ndarray]`` mapping each
    gated variable name to a boolean mask with one entry per input row,
    ``True`` meaning the deficit is applied to that row.

    Note: ``isinstance`` checks against this (runtime-checkable) protocol
    only verify that the object is callable, not the full signature. All
    intensity configuration lives in the constructor. Return values may be
    boolean masks (deterministic fire) or float SCORES in ``[0, 1]`` (the
    probability that the deficit is applied); the consuming generator
    realizes fractional scores appropriately (Bernoulli firing for gating
    consumers, selection weights for resampling consumers).
    """

    def __call__(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        ...  # pragma: no cover


class BernoulliSelector:
    """Selector: iid per-row, per-variable deficit indicator draws.

    For each requested variable, whether a row is affected (``R = 1``) is an
    independent Bernoulli draw with probability ``rate``, independent of the
    data values and of ``y``. In the missingness literature this pattern is
    called "completely at random"; the class name here describes the RULE
    (iid Bernoulli firing), not the deficit, because the same rule gates any
    observation-side deficit form (missing values, noise, out-of-distribution
    values, ...).

    Parameters
    ----------
    variables : str or sequence of str
        Name(s) of the variables this selector gates: one dict key and one
        INDEPENDENT mask per name (a single vectorized 2D draw). Names are
        graph-node names -- ``X``/``y`` are positional numpy arrays, so the
        consuming generator resolves names to column positions via its
        column_mapping (the same binding as for ``rng``). This selector never
        reads column values (value-free rule).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws. The consuming generator binds
        its own generator here (single RNG stream). Calling without a bound
        generator raises ``ValueError``.

    Calling
    -------
    ``__call__(X, y=None)``: returns one mask per variable.

    Examples
    --------
    >>> import numpy as np
    >>> selector = BernoulliSelector("temperature", rate=0.3,
    ...                              rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> masks = selector(X, y=None)
    >>> list(masks) == ["temperature"]
    True
    >>> masks["temperature"].dtype
    dtype('bool')

    A list of variables yields one independent mask per variable:

    >>> selector = BernoulliSelector(["a", "b"], rate=0.5,
    ...                              rng=np.random.default_rng(0))
    >>> sorted(selector(X, y=None))
    ['a', 'b']
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        rate: float,
        rng: np.random.Generator | None = None,
    ):
        self.variables = _validate_variables(variables)
        self.rate = _validate_rate(rate, "rate")
        self.rng = rng

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if self.rng is None:
            raise ValueError(
                f"{self!r} has no random generator bound: construct it with "
                "rng=numpy.random.Generator (the consuming generator binds "
                "its own) or set selector.rng before calling."
            )
        # One vectorized iid draw: shape (n_rows, k), True = deficit applied.
        draws = (
            self.rng.random(size=(X.shape[0], len(self.variables)))
            < self.rate
        )
        return {name: draws[:, i] for i, name in enumerate(self.variables)}

    def __repr__(self) -> str:
        return (
            f"BernoulliSelector(variables={self.variables!r}, "
            f"rate={self.rate!r})"
        )


def _validate_variables(variables) -> tuple[str, ...]:
    """Shared variable-name validation for all presets."""
    if isinstance(variables, str):
        variables = (variables,)
    variables = tuple(variables)
    if len(variables) == 0:
        raise ValueError("variables must not be empty")
    for name in variables:
        if not isinstance(name, str):
            raise ValueError(
                f"variable names must be str, got "
                f"{type(name).__name__} for {name!r}"
            )
    if len(set(variables)) != len(variables):
        raise ValueError(
            f"variable names must be unique, got {variables}"
        )
    return variables


def _validate_rate(rate: float, name: str) -> float:
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {rate}")
    return rate


class ExactCountSelector:
    """Selector: exactly ``int(rate * n_rows)`` rows per variable, drawn
    without replacement (uniform, or weighted by per-row probabilities).

    The exact-count sibling of :class:`BernoulliSelector` (random count).
    This is the rule behind the classic missingness generators: per
    variable, a fixed number of affected rows, optionally with per-row
    selection probabilities ("soft" coupling -- the probability of being
    affected varies smoothly with the data instead of a hard quantile
    threshold).

    Parameters
    ----------
    variables : str or sequence of str
        Variable names gated by this selector; one INDEPENDENT mask per
        name (rows are drawn separately per variable).
    weights : None, (n_rows, len(variables)) array, or callable
        Per-row selection probabilities. ``None`` = uniform. An array
        must have shape ``(X.shape[0], len(variables))`` with non-negative
        entries summing to (approximately) 1 per column -- pass them
        through :func:`badgers.core.utils.normalize_proba` first if
        needed. A callable ``(X, y=None) -> (n_rows, k)`` array is
        evaluated per call -- this is how data-driven coupling is
        expressed (the caller binds any randomness, e.g. rng, inside the
        callable).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws. The consuming generator
        binds its own generator here (single RNG stream). Calling without
        a bound generator raises ``ValueError``.

    Calling
    -------
    ``__call__(X, y=None)``: the affected count is ``int(rate * n_rows)``
    (floor, classic semantics).

    Examples
    --------
    >>> import numpy as np
    >>> selector = ExactCountSelector("temperature", rate=0.4,
    ...                               rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> selector(X, y=None)["temperature"].sum()
    4
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        rate: float,
        weights: np.ndarray
        | Callable[[np.ndarray, np.ndarray | None], np.ndarray]
        | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.variables = _validate_variables(variables)
        self.rate = _validate_rate(rate, "rate")
        self.weights = weights
        self.rng = rng

    def _resolve_weights(
        self, X: np.ndarray, y: np.ndarray | None
    ) -> np.ndarray | None:
        k = len(self.variables)
        if self.weights is None:
            return None
        w = self.weights(X, y) if callable(self.weights) else self.weights
        w = np.asarray(w, dtype=float)
        if w.shape != (X.shape[0], k):
            raise ValueError(
                f"weights must have shape ({X.shape[0]}, {k}), got {w.shape}"
            )
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        return w

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if self.rng is None:
            raise ValueError(
                f"{self!r} has no random generator bound: construct it with "
                "rng=numpy.random.Generator (the consuming generator binds "
                "its own) or set selector.rng before calling."
            )
        n_rows = X.shape[0]
        count = int(self.rate * n_rows)
        # Resolve the weights BEFORE the row draws: data-driven coupling
        # may consume rng draws itself (e.g. choosing driver columns), and
        # those must precede the per-variable row draws (draw-order parity
        # with the classic generators).
        weights = self._resolve_weights(X, y)
        masks = {}
        for i, name in enumerate(self.variables):
            # Sampling without replacement: sequential by nature -- the
            # documented exception to the vectorization rule.
            p = None if weights is None else weights[:, i]
            idx = self.rng.choice(n_rows, size=count, replace=False, p=p)
            mask = np.zeros(n_rows, dtype=bool)
            mask[idx] = True
            masks[name] = mask
        return masks

    def __repr__(self) -> str:
        return (
            f"ExactCountSelector(variables={self.variables!r}, "
            f"rate={self.rate!r}, "
            f"weights={'callable' if callable(self.weights) else self.weights!r})"
        )


class TopQuantileSelector:
    """Selector: fires rows above the ``quantile`` of a column.

    Deterministic, value-driven rule: for each gated variable the threshold
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
      broadcast to all variables (one driver gates everything); a sequence
      gives one driver per variable (same length as ``variables``).

    Because the rule is deterministic, no ``rng`` is needed.

    Parameters
    ----------
    variables : str or sequence of str
        Names of the gated variables (the returned dict's keys).
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
    ``__call__(X, y=None)``: one mask per gated variable; a row fires
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
        for name in self.variables:
            if name not in column_mapping:
                raise ValueError(
                    f"variable {name!r} not in column_mapping"
                )
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
        y: np.ndarray | None = None,
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

    For each gated variable (a column of ``X`` intended to hold DISCRETE
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
        Names of the class columns gated by this selector (the returned
        dict's keys). Each must be a key of ``column_mapping``.
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
    ``__call__(X, y=None)``: one mask per class column; a row fires if it
    belongs to a fired class. With several class columns the masks are
    INDEPENDENT per column -- consumers typically OR them (a row dropped
    if fired in any column). ``y`` is unused: class columns are columns
    of ``X`` (the target is being deprecated as a data carrier).

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
        for name in self.variables:
            if name not in column_mapping:
                raise ValueError(
                    f"variable {name!r} not in column_mapping"
                )
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
        y: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if self.rng is None:
            raise ValueError(
                f"{self!r} has no random generator bound: construct it with "
                "rng=numpy.random.Generator (the consuming generator binds "
                "its own) or set selector.rng before calling."
            )
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

    Value-driven, FRACTIONAL rule: for each gated variable the per-row
    deficit probability is derived from the column's values, rescaled to
    ``[0, 1]`` (min-max) and then normalized to sum to 1. A constant
    column yields the uniform probability. Deterministic (no ``rng``); the
    consumer decides how the probabilities are realized (weights in a
    weighted draw, Bernoulli firing, ...). "Proportional" describes the
    RULE, never the deficit that follows.

    Parameters
    ----------
    variables : str or sequence of str
        Names of the gated variables (the returned dict's keys). Each must
        be a key of ``column_mapping``.
    column_mapping : dict mapping str -> int
        Maps each variable name to its column index in ``X`` (standard
        binding; this preset READS values).

    Calling
    -------
    ``__call__(X, y=None)``: one float score array per variable (the
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
        for name in self.variables:
            if name not in column_mapping:
                raise ValueError(
                    f"variable {name!r} not in column_mapping"
                )
        self.column_mapping = column_mapping

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
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
