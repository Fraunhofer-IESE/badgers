"""Value-free selector presets: rules that never read column values.

Because the rule is value-blind, no ``column_mapping`` is needed at
construction; the stochastic presets bind a single rng stream (the consuming
generator binds its own generator here).
"""

from collections.abc import Callable, Sequence

import numpy as np

from .base import _mask_from_indices, _require_rng, _validate_rate, _validate_variables

__all__ = [
    "AlwaysSelector",
    "BernoulliSelector",
    "ExactCountSelector",
]


class AlwaysSelector:
    """Selector: every row fires, for every variable whose affected rows
    it decides.

    The trivial, value- and rng-free rule — the degenerate rule "apply the
    deficit everywhere". The default when a causal generator receives
    ``selector=None`` (noise, drift, edge perturbation).

    Parameters
    ----------
    variables : str or sequence of str
        Names of the variables whose affected rows the selector decides
        (the returned dict's keys).

    Calling
    -------
    ``__call__(X)``: returns an all-True mask per variable. No
    intensity kwarg — there is nothing to sweep (any intensity knob lives
    in the DEFICIT itself, e.g. noise_std).
    """

    def __init__(self, variables: str | Sequence[str]):
        self.variables = _validate_variables(variables)

    def __call__(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            name: np.ones(X.shape[0], dtype=bool)
            for name in self.variables
        }

    def __repr__(self) -> str:
        return f"AlwaysSelector(variables={self.variables!r})"


class BernoulliSelector:
    """Selector: iid per-row, per-variable deficit indicator draws.

    For each requested variable, whether a row is affected (``R = 1``) is an
    independent Bernoulli draw with probability ``rate``, independent of the
    data values. In the missingness literature this pattern is
    called "completely at random"; the class name here describes the RULE
    (iid Bernoulli firing), not the deficit, because the same rule applies to
    any observation-side deficit form (missing values, noise, out-of-distribution
    values, ...).

    Parameters
    ----------
    variables : str or sequence of str
        Name(s) of the variables whose affected rows this selector decides:
        one dict key and one INDEPENDENT mask per name (a single vectorized
        2D draw). Names are Graph-node names; the consuming generator
        resolves names to column positions via its column_mapping (the same
        binding as for ``rng``). This selector never reads column values
        (value-free rule).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws. The consuming generator binds
        its own generator here (single RNG stream). Calling without a bound
        generator raises ``ValueError``.

    Calling
    -------
    ``__call__(X)``: returns one mask per variable.

    Examples
    --------
    >>> import numpy as np
    >>> selector = BernoulliSelector("temperature", rate=0.3,
    ...                              rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> masks = selector(X)
    >>> list(masks) == ["temperature"]
    True
    >>> masks["temperature"].dtype
    dtype('bool')

    A list of variables yields one independent mask per variable:

    >>> selector = BernoulliSelector(["a", "b"], rate=0.5,
    ...                              rng=np.random.default_rng(0))
    >>> sorted(selector(X))
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
    ) -> dict[str, np.ndarray]:
        _require_rng(self)
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
        Variables whose affected rows this selector decides; one
        INDEPENDENT mask per name (rows are drawn separately per variable).
    weights : None, (n_rows, len(variables)) array, or callable
        Per-row selection probabilities. ``None`` = uniform. An array
        must have shape ``(X.shape[0], len(variables))`` with non-negative
        entries summing to (approximately) 1 per column -- pass them
        through :func:`badgers.core.utils.normalize_proba` first if
        needed. A callable ``(X) -> (n_rows, k)`` array is evaluated per
        call -- this is how data-driven coupling is
        expressed (the caller binds any randomness, e.g. rng, inside the
        callable).
    rng : numpy.random.Generator or None, default=None
        Random generator used for the draws. The consuming generator
        binds its own generator here (single RNG stream). Calling without
        a bound generator raises ``ValueError``.

    Calling
    -------
    ``__call__(X)``: the affected count is ``int(rate * n_rows)``
    (floor, classic semantics).

    Examples
    --------
    >>> import numpy as np
    >>> selector = ExactCountSelector("temperature", rate=0.4,
    ...                               rng=np.random.default_rng(0))
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> selector(X)["temperature"].sum()
    4
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        rate: float,
        weights: np.ndarray
        | Callable[[np.ndarray], np.ndarray]
        | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.variables = _validate_variables(variables)
        self.rate = _validate_rate(rate, "rate")
        self.weights = weights
        self.rng = rng

    def _resolve_weights(self, X: np.ndarray) -> np.ndarray | None:
        k = len(self.variables)
        if self.weights is None:
            return None
        w = self.weights(X) if callable(self.weights) else self.weights
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
    ) -> dict[str, np.ndarray]:
        _require_rng(self)
        n_rows = X.shape[0]
        count = int(self.rate * n_rows)
        # Resolve the weights BEFORE the row draws: data-driven coupling
        # may consume rng draws itself (e.g. choosing driver columns), and
        # those must precede the per-variable row draws (draw-order parity
        # with the classic generators).
        weights = self._resolve_weights(X)
        masks = {}
        for i, name in enumerate(self.variables):
            # Sampling without replacement: sequential by nature -- the
            # documented exception to the vectorization rule.
            p = None if weights is None else weights[:, i]
            idx = self.rng.choice(n_rows, size=count, replace=False, p=p)
            masks[name] = _mask_from_indices(n_rows, idx)
        return masks

    def __repr__(self) -> str:
        return (
            f"ExactCountSelector(variables={self.variables!r}, "
            f"rate={self.rate!r}, "
            f"weights={'callable' if callable(self.weights) else self.weights!r})"
        )
