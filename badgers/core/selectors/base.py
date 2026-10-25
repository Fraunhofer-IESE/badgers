"""Contract-level objects shared by the selector presets and their consumers.

This submodule holds the :class:`Selector` protocol, the shared validation
helpers used by all presets, and the consumer-side helpers that realize a
selector's output into selected fields. The presets themselves live in the
``value_free`` and ``value_driven`` submodules.

Selectors are index-only: variable names and ``column_mapping`` dicts do
not exist in the selector layer.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Selector",
    "get_selected_fields",
    "resolve_class_column",
]


@runtime_checkable
class Selector(Protocol):
    """Protocol for selectors: callables producing per-column selection specs.

    Implementations accept ``X`` and an optional ``targets`` sequence of
    column positions and return ``dict[int, spec]`` where the keys are the
    selected column positions and each spec is a ``[0, 1]`` scalar
    (uniform level for the whole column) or an array of shape
    ``(n_rows,)`` (per-row level). Values are DECISIONS when restricted to
    ``{0, 1}`` and PROBABILITIES otherwise; the consuming generator
    realizes both with a single rule (``rng.random(n) < spec``).
    """

    def __call__(self, X: np.ndarray,
                 targets: Sequence[int] | None = None) -> dict[int, float | np.ndarray]:
        ...  # pragma: no cover


def _validate_columns(columns) -> tuple[int, ...]:
    """Shared column-position validation for all presets."""
    if isinstance(columns, (int, np.integer)):
        columns = (int(columns),)
    try:
        cols = tuple(int(c) for c in columns)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"columns must be ints, got {columns!r}") from exc
    if len(cols) == 0:
        raise ValueError("columns must not be empty")
    if any(c < 0 for c in cols):
        raise ValueError(f"columns must be >= 0, got {cols}")
    if len(set(cols)) != len(cols):
        raise ValueError(f"columns must be unique, got {cols}")
    return cols


def _resolve_targets(selector, X: np.ndarray, targets) -> tuple[int, ...]:
    """Resolve call-time targets against the selector's own column scope.

    ``None`` defaults to the selector's constructor ``columns`` (or, for a
    user callable without that attribute, every column of ``X``). Validates
    range against ``X``, and ``targets ⊆ selector.columns`` when the
    selector declares a scope (self-contained selector invariant).
    """
    if targets is None:
        targets = getattr(selector, "columns", None)
        if targets is None:
            targets = range(X.shape[1])
    targets = _validate_columns(targets)
    n_cols = X.shape[1]
    if any(c >= n_cols for c in targets):
        raise ValueError(
            f"targets {targets} out of range for X with {n_cols} columns")
    own = getattr(selector, "columns", None)
    if own is not None:
        extra = [c for c in targets if c not in own]
        if extra:
            raise ValueError(
                f"targets {extra} outside the selector's columns {own}")
    return targets


def _validate_rate(rate: float, name: str) -> float:
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {rate}")
    return rate


def _check_spec(spec, n_rows: int) -> None:
    """Validate a selection spec: scalar or (n_rows,) array, values in [0, 1]."""
    arr = np.asarray(spec)
    if arr.ndim == 0 and arr.dtype == bool:
        raise ValueError(
            "spec must be a number or a 1-D array of length "
            f"{n_rows}, not a bool; bools are only valid as "
            "an (n,) array"
        )
    if arr.ndim > 1 or (arr.ndim == 1 and arr.shape != (n_rows,)):
        raise ValueError(
            f"spec must be a scalar or a 1-D array of length {n_rows}, "
            f"got shape {arr.shape}")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError("spec values must be in [0, 1]")


def _require_rng(selector) -> None:
    """Shared check that a stochastic preset has a random generator bound."""
    if selector.rng is None:
        raise ValueError(
            f"{selector!r} has no random generator bound: construct it with "
            "rng=numpy.random.Generator (the consuming generator binds "
            "its own) or set selector.rng before calling.")


def _mask_from_indices(n_rows: int, idx: np.ndarray) -> np.ndarray:
    """Boolean row mask that is True exactly at the given positions."""
    mask = np.zeros(n_rows, dtype=bool)
    mask[idx] = True
    return mask


def get_selected_fields(X, selector, rng):
    """Realized selection field: bool array ``(n_rows, len(targets))``.

    Column *i* corresponds to the selector's *i*-th target (its own
    ``columns`` when it declares a scope, else every column). Narrow on
    purpose: the caller decides how to write the field back into full
    coordinates.

    ``selector=None`` returns every column all-True WITHOUT consuming rng
    draws — the backward-compatible default of consuming generators.

    Single realization rule (shared with every consumer):
    ``rng.random(n) < spec``; the mask is decided against the UN-mutated
    input. Constant decisions (a scalar spec of exactly ``0.0`` or
    ``1.0``) are deterministic and realized WITHOUT consuming rng draws,
    so default selectors built on ``AlwaysSelector`` cost no draws.
    """
    n_rows, n_cols = X.shape
    if selector is None:
        return np.ones((n_rows, n_cols), dtype=bool)
    targets = _resolve_targets(selector, X, None)
    out = selector(X, targets=list(targets))
    unknown = [c for c in out if c not in targets]
    if unknown:
        raise ValueError(
            f"selector returned keys {unknown} not in targets {targets}")
    field = np.zeros((n_rows, len(targets)), dtype=bool)
    for i, col in enumerate(targets):
        if col in out:
            spec = out[col]
            _check_spec(spec, n_rows)
            spec = np.asarray(spec)
            if spec.ndim == 0 and float(spec) in (0.0, 1.0):
                # constant decision: deterministic, no rng draw consumed
                field[:, i] = float(spec)
            else:
                field[:, i] = rng.random(n_rows) < np.broadcast_to(spec, (n_rows,))
    return field


def resolve_class_column(selector, X: np.ndarray):
    """Recover the class column position from a value-reading selector.

    Value-driven presets expose a ``class_column`` (or ``column``)
    attribute holding the read column position. Returns ``None`` when the
    selector has no such attribute or the value is out of range.
    """
    col = getattr(selector, "class_column", None)
    if col is None:
        col = getattr(selector, "column", None)
    if col is not None and 0 <= col < X.shape[1]:
        return col
    return None
