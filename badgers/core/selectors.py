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

The presets are small callable objects so their parameters stay introspectable
(``repr`` and stored attributes: "which selector produced this mask?" is
answerable). There is deliberately no base class and no registry -- conformance
to the :class:`Selector` protocol is defined by the call signature alone. Each
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
    only verify that the object is callable, not the full signature.
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
    rate : float
        Probability that a row's variable is affected, in [0, 1].
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

    Examples
    --------
    >>> import numpy as np
    >>> selector = BernoulliSelector(rate=0.3, variables="temperature",
    ...                              rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> masks = selector(X, y=None)
    >>> list(masks) == ["temperature"]
    True
    >>> masks["temperature"].dtype
    dtype('bool')

    A list of variables yields one independent mask per variable:

    >>> selector = BernoulliSelector(rate=0.5, variables=["a", "b"],
    ...                              rng=np.random.default_rng(0))
    >>> sorted(selector(X, y=None))
    ['a', 'b']
    """

    def __init__(
        self,
        rate: float,
        variables: str | Sequence[str],
        rng: np.random.Generator | None = None,
    ):
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"rate must be in [0, 1], got {rate}")
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
        self.rate = rate
        self.variables = variables
        self.rng = rng

    def __call__(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        if self.rng is None:
            raise ValueError(
                f"{self!r} has no random generator bound: construct it with "
                "rng=numpy.random.Generator (the consuming generator binds "
                "its own) or set selector.rng before calling."
            )
        # One vectorized iid draw: shape (n_rows, k), True = deficit applied.
        draws = self.rng.random(size=(X.shape[0], len(self.variables))) < self.rate
        return {name: draws[:, i] for i, name in enumerate(self.variables)}

    def __repr__(self) -> str:
        return (
            f"BernoulliSelector(rate={self.rate!r}, "
            f"variables={self.variables!r})"
        )


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
    rate : float
        Fraction of rows affected per variable, in [0, 1]; the affected
        count is ``int(rate * n_rows)`` (floor, classic semantics).
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

    Examples
    --------
    >>> import numpy as np
    >>> selector = ExactCountSelector(rate=0.4, variables="temperature",
    ...                               rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> masks = selector(X, y=None)
    >>> masks["temperature"].sum()
    4
    """

    def __init__(
        self,
        rate: float,
        variables: str | Sequence[str],
        weights: np.ndarray
        | Callable[[np.ndarray, np.ndarray | None], np.ndarray]
        | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"rate must be in [0, 1], got {rate}")
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
        self.rate = rate
        self.variables = variables
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
        self, X: np.ndarray, y: np.ndarray | None = None
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
            f"ExactCountSelector(rate={self.rate!r}, "
            f"variables={self.variables!r}, "
            f"weights={'callable' if callable(self.weights) else self.weights!r})"
        )


class TopQuantileSelector:
    """Selector: fires rows in the top ``rate`` fraction of a column.

    Deterministic, value-driven rule: for each gated variable the threshold
    is ``quantile(col, 1 - rate)`` and a row fires when the compared value
    is ``>= threshold``. Like the other presets, the class name describes
    the RULE (top-quantile thresholding), never the deficit that follows.

    Two modes:

    - **self** (``drivers=None``): each variable fires rows where its OWN
      value is in its own top ``rate`` fraction.
    - **driver** (``drivers=...``): each variable fires rows where a DRIVER
      column's value is in the driver's top ``rate`` fraction. A single
      string is broadcast to all variables (one driver gates everything);
      a sequence gives one driver per variable (same length as
      ``variables``).

    Because the rule is deterministic, no ``rng`` is needed.

    Parameters
    ----------
    rate : float
        Fraction of rows that fire (``0 <= rate <= 1``). With ``rate=0``
        the threshold is the column maximum, so rows equal to the max
        still fire (``>=`` comparison).
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
    ``__call__(X, y=None, stats=None)`` returns
    ``{variable: bool mask of len(X)}``. ``stats`` (optional) is the row set
    on which the QUANTILE THRESHOLDS are computed; comparisons are always
    against ``X``. Defaults to ``X`` itself. This supports generators that
    gate expanded rows against the original columns' statistics (e.g. a
    row-duplication noise generator): pass ``stats=original_rows``.
    """

    def __init__(
        self,
        rate: float,
        variables: str | Sequence[str],
        column_mapping: dict[str, int],
        drivers: str | Sequence[str] | None = None,
    ):
        if not (0.0 <= rate <= 1.0):
            raise ValueError(
                f"rate must be within [0, 1], got {rate}"
            )
        variables = (variables,) if isinstance(variables, str) else tuple(variables)
        if len(variables) == 0:
            raise ValueError("variables must be a non-empty sequence")
        if len(set(variables)) != len(variables):
            raise ValueError(
                f"variable names must be unique, got {variables}"
            )
        for name in variables:
            if name not in column_mapping:
                raise ValueError(
                    f"variable {name!r} not in column_mapping"
                )
        if isinstance(drivers, str):
            drivers = (drivers,) * len(variables)
        elif drivers is not None:
            drivers = tuple(drivers)
            if len(drivers) != len(variables):
                raise ValueError(
                    f"drivers must be a single str or one driver per "
                    f"variable ({len(variables)}), got {len(drivers)}"
                )
        if drivers is not None:
            for name in drivers:
                if name not in column_mapping:
                    raise ValueError(
                        f"driver {name!r} not in column_mapping"
                    )
        self.rate = rate
        self.variables = variables
        self.column_mapping = column_mapping
        self.drivers = drivers

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        stats: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        stats = X if stats is None else stats
        masks = {}
        for name, driver in zip(self.variables, self.drivers or self.variables):
            dcol = self.column_mapping[driver]
            threshold = np.quantile(stats[:, dcol], 1.0 - self.rate)
            masks[name] = X[:, dcol] >= threshold
        return masks

    def __repr__(self) -> str:
        return (
            f"TopQuantileSelector(rate={self.rate!r}, "
            f"variables={self.variables!r}, "
            f"drivers={self.drivers!r})"
        )
