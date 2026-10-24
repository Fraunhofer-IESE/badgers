"""Value-free selector presets: rules that never read column values.

Because the rule is value-blind, no read column is needed at
construction; the presets take ``columns`` (the column positions they
cover). Two rng streams: the preset's own rng draws the mask (bind it
at construction with ``rng=``; a stochastic preset without one raises),
while the consuming generator realizes the returned masks with its own
rng (``rng.random(n) < spec``). Some consumers instead hand their own
rng to the preset at construction (e.g. the missingness generators
build ``ExactCountSelector(..., rng=...)`` from their stream) -- the
preset's draw is then the consumer's stream.
"""

from collections.abc import Callable, Sequence

import numpy as np

from .base import (
    _mask_from_indices,
    _require_rng,
    _resolve_targets,
    _validate_columns,
    _validate_rate,
)

__all__ = [
    "AlwaysSelector",
    "BernoulliSelector",
    "CountSelector",
    "ExactCountSelector",
]


class AlwaysSelector:
    """Selector: every row fires, for every target column.

    The trivial, value- and rng-free rule -- the degenerate rule "apply
    the deficit everywhere". Emits the uniform spec ``1.0`` per target.

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers (its scope; call-time
        ``targets`` must stay inside it).
    """

    def __init__(self, columns: int | Sequence[int]):
        self.columns = _validate_columns(columns)

    def __call__(self, X: np.ndarray, targets=None):
        targets = _resolve_targets(self, X, targets)
        return {col: 1.0 for col in targets}

    def __repr__(self) -> str:
        return f"AlwaysSelector(columns={self.columns!r})"


class BernoulliSelector:
    """Selector: iid per-row, per-column deficit indicator draws.

    For each target column, whether a row is affected is an independent
    Bernoulli draw with probability ``rate``, independent of the data
    values. In the missingness literature this pattern is called
    "completely at random". One vectorized ``(n_rows, k)`` draw: column
    order follows ``targets`` order (RNG parity with the legacy
    per-variable draw order).

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    rate : float, in [0, 1]
        Per-cell firing probability.
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws.
    """

    def __init__(self, columns: int | Sequence[int], rate: float,
                 rng: np.random.Generator | None = None):
        self.columns = _validate_columns(columns)
        self.rate = _validate_rate(rate, "rate")
        self.rng = rng

    def __call__(self, X: np.ndarray, targets=None):
        _require_rng(self)
        targets = _resolve_targets(self, X, targets)
        draws = self.rng.random(size=(X.shape[0], len(targets))) < self.rate
        return {col: draws[:, i] for i, col in enumerate(targets)}

    def __repr__(self) -> str:
        return (f"BernoulliSelector(columns={self.columns!r}, "
                f"rate={self.rate!r})")


class ExactCountSelector:
    """Selector: exactly ``int(rate * n_rows)`` rows per target column,
    drawn without replacement (uniform, or weighted by per-row
    probabilities).

    The exact-count sibling of :class:`BernoulliSelector`. Per column, a
    fixed number of affected rows, optionally with per-row selection
    probabilities ("soft" coupling).

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    rate : float, in [0, 1]
        Affected rows per column = ``int(rate * n_rows)`` (floor).
    weights : None, (n_rows, k) array, or callable ``(X) -> (n_rows, k)``
        Per-row selection probabilities, one column per target.
        ``None`` = uniform. The callable is evaluated per call (the
        caller binds any randomness inside it, BEFORE the row draws).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws.
    """

    def __init__(self, columns: int | Sequence[int], rate: float,
                 weights: np.ndarray
                 | Callable[[np.ndarray], np.ndarray] | None = None,
                 rng: np.random.Generator | None = None):
        self.columns = _validate_columns(columns)
        self.rate = _validate_rate(rate, "rate")
        self.weights = weights
        self.rng = rng

    def _resolve_weights(self, X: np.ndarray, targets) -> np.ndarray | None:
        k = len(targets)
        if self.weights is None:
            return None
        w = self.weights(X) if callable(self.weights) else self.weights
        w = np.asarray(w, dtype=float)
        if w.shape != (X.shape[0], k):
            raise ValueError(
                f"weights must have shape ({X.shape[0]}, {k}), got {w.shape}")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        return w

    def __call__(self, X: np.ndarray, targets=None):
        _require_rng(self)
        targets = _resolve_targets(self, X, targets)
        n_rows = X.shape[0]
        count = int(self.rate * n_rows)
        # Resolve the weights BEFORE the row draws: data-driven coupling
        # may consume rng draws itself, and those must precede the per-
        # column row draws (draw-order parity with the classic generators).
        weights = self._resolve_weights(X, targets)
        masks = {}
        for i, col in enumerate(targets):
            # Sampling without replacement: sequential by nature.
            p = None if weights is None else weights[:, i]
            idx = self.rng.choice(n_rows, size=count, replace=False, p=p)
            masks[col] = _mask_from_indices(n_rows, idx)
        return masks

    def __repr__(self) -> str:
        return (f"ExactCountSelector(columns={self.columns!r}, "
                f"rate={self.rate!r})")


class CountSelector:
    """Selector: exactly ``count`` distinct rows, one shared mask.

    The exact-count ROW preset: draws ``count`` row positions without
    replacement (uniform) and returns the same boolean mask under every
    target column — the requested count is exact at ROW level. This is
    the machinery behind the outliers generators' default count: an
    explicit count is a request, and a request that overflows the data
    raises.

    Parameters
    ----------
    columns : int or sequence of int
        Column positions this selector covers.
    count : int
        Number of rows to fire; must be an int >= 0 and <= the number
        of rows of the ``X`` passed at call time.
    rng : numpy.random.Generator or None, default=None
        Random generator used for the row draws.
    """

    def __init__(self, columns: int | Sequence[int], count: int,
                 rng: np.random.Generator | None = None):
        self.columns = _validate_columns(columns)
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool):
            raise ValueError(f"count must be an int, got {count!r}")
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        self.count = int(count)
        self.rng = rng

    def __call__(self, X: np.ndarray, targets=None):
        _require_rng(self)
        targets = _resolve_targets(self, X, targets)
        n_rows = X.shape[0]
        if self.count > n_rows:
            raise ValueError(
                f"count {self.count} exceeds the number of rows of X "
                f"({n_rows}): an exact count cannot select more distinct "
                "rows than exist")
        if self.count == 0:
            empty = np.zeros(n_rows, dtype=bool)
            return {col: empty for col in targets}
        # Sampling without replacement: sequential by nature.
        idx = self.rng.choice(n_rows, size=self.count, replace=False)
        mask = _mask_from_indices(n_rows, idx)
        return {col: mask for col in targets}

    def __repr__(self) -> str:
        return (f"CountSelector(columns={self.columns!r}, "
                f"count={self.count!r})")
