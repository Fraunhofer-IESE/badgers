"""Contract-level objects shared by the selector presets and their consumers.

This submodule holds the :class:`Selector` protocol, the shared validation
helpers used by all presets, and the consumer-side helpers that interpret a
selector's output (``fired_mask``/``fired_rows``) or sniff its configuration
(``resolve_class_column``). The presets themselves live in the ``value_free``
and ``value_driven`` submodules.
"""

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Selector",
    "fired_mask",
    "fired_rows",
    "resolve_class_column",
]


@runtime_checkable
class Selector(Protocol):
    """Protocol for selectors: callables producing per-variable deficit masks.

    Implementations (preset objects or user callables) accept only ``X``
    and return a ``dict[str, numpy.ndarray]`` mapping each variable name to
    a boolean mask with one entry per input row, ``True`` meaning the
    deficit is applied to that row.

    Note: ``isinstance`` checks against this (runtime-checkable) protocol
    only verify that the object is callable, not the full signature. All
    intensity configuration lives in the constructor. Return values may be
    boolean masks (deterministic fire) or float SCORES in ``[0, 1]`` (the
    probability that the deficit is applied); the consuming generator
    realizes fractional scores appropriately (Bernoulli firing for modifying
    consumers, selection weights for resampling consumers).
    """

    def __call__(self, X: np.ndarray) -> dict[str, np.ndarray]:
        ...  # pragma: no cover


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


def _validate_column_mapping(variables, column_mapping: dict[str, int]) -> None:
    """Shared check that every variable whose affected rows the selector
    decides is a key of ``column_mapping``."""
    for name in variables:
        if name not in column_mapping:
            raise ValueError(
                f"variable {name!r} not in column_mapping"
            )


def _require_rng(selector) -> None:
    """Shared check that a stochastic preset has a random generator bound."""
    if selector.rng is None:
        raise ValueError(
            f"{selector!r} has no random generator bound: construct it with "
            "rng=numpy.random.Generator (the consuming generator binds "
            "its own) or set selector.rng before calling."
        )


def _mask_from_indices(n_rows: int, idx: np.ndarray) -> np.ndarray:
    """Boolean row mask that is True exactly at the given positions."""
    mask = np.zeros(n_rows, dtype=bool)
    mask[idx] = True
    return mask


def fired_mask(X: np.ndarray, selector) -> np.ndarray:
    """Per-row firing mask for a selector: OR of its masks.

    Masks are computed on the UN-mutated input (compute-all-masks-first
    invariant: consuming generators call this before any mutation).
    Entries may be boolean or float scores in ``[0, 1]``; a float score
    fires when non-zero (the degenerate Boolean reading of the score,
    mirroring the Classes variants). ``None`` = every row fires (backward
    compatible default of consuming generators).
    """
    if selector is None:
        return np.ones(X.shape[0], dtype=bool)
    masks = list(selector(X).values())
    fired = masks[0] if len(masks) == 1 else np.logical_or.reduce(masks)
    fired = np.asarray(fired)
    return fired if fired.dtype == bool else fired != 0


def fired_rows(X: np.ndarray, selector) -> np.ndarray:
    """Row positions affected by a selector: :func:`fired_mask`, flattened."""
    return np.flatnonzero(fired_mask(X, selector))


def resolve_class_column(selector, X: np.ndarray):
    """Recover the class column position from a value-reading selector.

    Value-reading selector presets (``TopQuantileSelector``,
    ``ClassSelector``, ...) expose a ``column_mapping`` attribute mapping
    variable names to column positions; the first variable whose position
    exists in ``X`` is used as the class column. Returns ``None`` when the
    selector has no such attribute (the consuming generator then treats
    all fired rows as one class, or requires a plain magnitude).
    """
    column_mapping = getattr(selector, "column_mapping", None)
    variables = getattr(selector, "variables", None)
    if column_mapping is None or not variables:
        return None
    for name in variables:
        pos = column_mapping.get(name)
        if pos is not None and 0 <= pos < X.shape[1]:
            return pos
    return None
