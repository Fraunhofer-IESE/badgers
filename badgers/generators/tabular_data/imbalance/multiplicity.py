"""Per-row multiplicities for :class:`OversamplingGenerator`.

A multiplicity ``m[i]`` is the number of times row ``i`` of the input appears
in the output. ``m[i] >= 1`` for every row (``m[i] == 1`` = keep the row once;
``m[i] == 3`` = keep the original plus two extra copies). Oversampling never
drops a row, so ``m[i] == 0`` is invalid.

A :class:`MultiplicitiesSelector` decides the multiplicities from the data:
``__call__(X)`` returns an integer array of shape ``(n_rows,)``. Unlike the
deficit :class:`~badgers.core.selectors.Selector` (a ``[0, 1]`` dict keyed by
column position), a multiplicity is a plain per-row integer, so this is a
separate, simpler contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MultiplicitiesSelector(Protocol):
    """Structural protocol for per-row multiplicity selection.

    ``__call__(X)`` must return an integer array of shape ``(X.shape[0],)``
    whose entries are all ``>= 1``. A bare integer (broadcast to every row)
    is also accepted.
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class ConstantMultiplicity:
    """Give every row the same multiplicity.

    Parameters
    ----------
    multiplicity : int
        Number of copies of every row. Must be ``>= 1``.
    """

    def __init__(self, multiplicity: int):
        if int(multiplicity) < 1:
            raise ValueError(f"multiplicity must be an integer >= 1, got {multiplicity}")
        self._multiplicity = int(multiplicity)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._multiplicity, dtype=int)


class MaskedMultiplicity:
    """Rows selected by a mask get ``multiplicity`` copies; the rest keep one.

    Parameters
    ----------
    mask : array-like of bool/bool-index, or callable
        Either a boolean/integer index array of length ``n_rows``, or a
        callable ``mask(X) -> bool array (n_rows,)`` evaluated at generate
        time (preferred, so the mask can depend on the data — e.g. a class
        column).
    multiplicity : int
        Number of copies for selected rows. Must be ``>= 1``.
    """

    def __init__(self, mask, multiplicity: int):
        if int(multiplicity) < 1:
            raise ValueError(
                f"multiplicity must be an integer >= 1, got {multiplicity}"
            )
        self._mask = mask
        self._multiplicity = int(multiplicity)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        m = np.ones(n, dtype=int)
        if callable(self._mask):
            selected = np.asarray(self._mask(X), dtype=bool)
        else:
            selected = np.asarray(self._mask, dtype=bool)
        if selected.shape != (n,):
            raise ValueError(
                "mask must be a 1-D boolean array of length "
                f"{n}, got shape {selected.shape}"
            )
        m[selected] = self._multiplicity
        return m


def resolve_multiplicities(X: np.ndarray, selector=None) -> np.ndarray:
    """Resolve a selector (or ``None``) into a per-row multiplicity array.

    ``selector=None`` yields multiplicity ``1`` for every row (output ==
    input). A bare integer is broadcast to every row; a bare array is used
    as-is. The result is validated
    (integer, shape ``(n_rows,)``, all ``>= 1``) with a friendly mix-up guard
    for deficit-style values in ``(0, 1)`` or ``0``.
    """
    n = X.shape[0]
    if selector is None:
        return np.ones(n, dtype=int)
    m = selector(X) if callable(selector) else selector
    m = np.asarray(m)
    if m.ndim == 0:
        # Broadcast the scalar; keep its dtype so a fractional scalar (3.5)
        # is caught by _validate_multiplicities instead of silently truncating.
        m = np.full(n, m)
    if m.shape != (n,):
        raise ValueError(
            "multiplicities must be a 1-D array of length "
            f"{n} (one per row), got shape {m.shape}"
        )
    return _validate_multiplicities(m, n)


def _validate_multiplicities(m: np.ndarray, n: int) -> np.ndarray:
    """Validate ``m`` and return it as an integer array (all ``>= 1``)."""
    is_float = m.dtype == np.float64 or np.issubdtype(m.dtype, np.floating)

    # Mix-up guard: deficit selectors yield [0, 1] values (fractions or 0/1
    # one-hot). Oversampling multiplicities must be integers >= 1.
    if np.any((m > 0) & (m < 1)):
        raise ValueError(
            "multiplicities must be integers >= 1, but values in (0, 1) were "
            "found. This looks like a deficit Selector (e.g. "
            "ProportionalSelector), which belongs to the undersampling "
            "generator, not oversampling."
        )
    if np.any(m == 0):
        raise ValueError(
            "multiplicities must be integers >= 1 (a row with 0 copies would "
            "be dropped; oversampling never drops rows). If you meant a "
            "deficit/one-hot mask, use the undersampling generator instead."
        )
    if np.any(m < 0):
        raise ValueError(f"multiplicities must be >= 1, got {m[m < 0].min()!r}")
    if is_float and not np.all(m == np.round(m)):
        raise ValueError(
            "non-integer multiplicities are not supported in this version "
            "(e.g. a 3.5x row). Use integers for now."
        )
    return m.astype(int)
