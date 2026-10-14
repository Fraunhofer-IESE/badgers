"""Selectors: the observation-side contract for selecting deficit rows.

A SELECTOR decides which rows of the input get a data quality deficit

    Selector = Callable[[X, y], dict[str, bool-mask]]

- keys   = variable names (the subscript in ``R_X``: which column the mask gates)
- values = per-row boolean masks, ``len(mask) == X.shape[0]``
- polarity: ``True`` = ``R_X = 1`` = deficit applied; ``False`` = record as-is

Selectors belong to the observation family of generators (missingness and the
observation-side defects). Structure-side generators (drift / relationship /
noise / outliers / imbalance) never consume them. Being in ``core``, the
module is modality-agnostic: time-series / graph / text generators can reuse
the same contract with masks over their own axes.

The presets are small callable objects so their configuration (which
variables, which rng, which column mapping) stays introspectable (``repr``
and stored attributes). The FIRE INTENSITY knob -- ``rate`` (the stochastic
presets) or ``quantile`` (the deterministic one) -- is NOT stored: it is a
required keyword argument of ``__call__``, so a single configured selector
can be swept over intensities without re-constructing it, and the knob is
visible right at the call site. There is deliberately no base class and no
registry -- conformance to the :class:`Selector` protocol is defined by the
call signature alone (presets extend it with the intensity kwarg). Each
preset is self-contained: its mask logic is inlined in ``__call__`` (no free
helper functions), so a preset depends on nothing beyond numpy and the
consuming generators simply construct one and call it.
"""

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Selector",
    "BernoulliSelector",
    "ExactCountSelector",
    "TopQuantileSelector",
]


@runtime_checkable
class Selector(Protocol):
    """Protocol for selectors: callables producing per-variable deficit masks.

    Implementations (preset objects or user callables) must accept
    ``(X, y=None)`` and return a ``dict[str, numpy.ndarray]`` mapping each
    gated variable name to a boolean mask with one entry per input row,
    ``True`` meaning the deficit is applied to that row.

    Note: ``isinstance`` checks against this (runtime-checkable) protocol
    only verify that the object is callable, not the full signature. The
    library presets extend the signature with a required keyword-only
    intensity argument (``rate=`` / ``quantile=``).
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
    ``__call__(X, y=None, rate)``: ``rate`` is the fire probability in
    [0, 1] -- a REQUIRED keyword argument, validated per call. Passing the
    same object with different ``rate`` values sweeps the intensity without
    re-constructing the selector (the rng stream simply continues).

    Examples
    --------
    >>> import numpy as np
    >>> selector = BernoulliSelector("temperature",
    ...                              rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> masks = selector(X, y=None, rate=0.3)
    >>> list(masks) == ["temperature"]
    True
    >>> masks["temperature"].dtype
    dtype('bool')

    A list of variables yields one independent mask per variable:

    >>> selector = BernoulliSelector(["a", "b"], rng=np.random.default_rng(0))
    >>> sorted(selector(X, y=None, rate=0.5))
    ['a', 'b']
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        rng: np.random.Generator | None = None,
    ):
        self.variables = _validate_variables(variables)
        self.rng = rng

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        rate: float,
    ) -> dict[str, np.ndarray]:
        _validate_rate(rate, "rate")
        if self.rng is None:
            raise ValueError(
                f"{self!r} has no random generator bound: construct it with "
                "rng=numpy.random.Generator (the consuming generator binds "
                "its own) or set selector.rng before calling."
            )
        # One vectorized iid draw: shape (n_rows, k), True = deficit applied.
        draws = self.rng.random(size=(X.shape[0], len(self.variables))) < rate
        return {name: draws[:, i] for i, name in enumerate(self.variables)}

    def __repr__(self) -> str:
        return f"BernoulliSelector(variables={self.variables!r})"


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
    ``__call__(X, y=None, rate)``: ``rate`` is the affected FRACTION per
    variable in [0, 1]; the affected count is ``int(rate * n_rows)``
    (floor, classic semantics). A REQUIRED keyword argument, validated
    per call.

    Examples
    --------
    >>> import numpy as np
    >>> selector = ExactCountSelector("temperature",
    ...                               rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> selector(X, y=None, rate=0.4)["temperature"].sum()
    4
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        weights: np.ndarray
        | Callable[[np.ndarray, np.ndarray | None], np.ndarray]
        | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.variables = _validate_variables(variables)
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
        *,
        rate: float,
    ) -> dict[str, np.ndarray]:
        _validate_rate(rate, "rate")
        if self.rng is None:
            raise ValueError(
                f"{self!r} has no random generator bound: construct it with "
                "rng=numpy.random.Generator (the consuming generator binds "
                "its own) or set selector.rng before calling."
            )
        n_rows = X.shape[0]
        count = int(rate * n_rows)
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
    drivers : str, sequence of str, or None, default=None
        The driver columns as described above.

    Calling
    -------
    ``__call__(X, y=None, quantile, stats=None)``:

    - ``quantile`` (REQUIRED keyword): the quantile of the (driver) column
      that forms the threshold, in [0, 1], validated per call. The
      comparison is ``>=``, so with ``quantile=1.0`` rows equal to the
      column maximum still fire.
    - ``stats`` (optional): the row set on which the QUANTILE THRESHOLDS
      are computed; comparisons are always against ``X``. Defaults to
      ``X`` itself. This supports generators that gate expanded rows
      against the original columns' statistics (e.g. a row-duplication
      noise generator): pass ``stats=original_rows``.
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        column_mapping: dict[str, int],
        drivers: str | Sequence[str] | None = None,
    ):
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
        *,
        quantile: float,
        stats: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        _validate_rate(quantile, "quantile")
        stats = X if stats is None else stats
        masks = {}
        for name, driver in zip(self.variables, self.drivers or self.variables):
            dcol = self.column_mapping[driver]
            threshold = np.quantile(stats[:, dcol], quantile)
            masks[name] = X[:, dcol] >= threshold
        return masks

    def __repr__(self) -> str:
        return (
            f"TopQuantileSelector(variables={self.variables!r}, "
            f"drivers={self.drivers!r})"
        )
