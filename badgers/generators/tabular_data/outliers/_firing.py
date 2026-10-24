"""Shared selector-firing plumbing for the outliers generators.

The consumers never inspect a selector's specs directly: they realize
the firing field through :func:`badgers.core.selectors.fired` and act
on seed rows and firing patterns. The direct (seed-copy) and the causal
(surgery) outlier generators share this plumbing.
"""

import numpy as np

from badgers.core.selectors import fired
from badgers.core.selectors.base import _resolve_targets


def fire_and_seed(X, selector, rng):
    """Realize the selector on ``X``: firing field, targets, seed rows.

    Returns ``(field, targets, seed_rows)``: the narrow bool field
    ``(n_rows, len(targets))`` (column ``i`` ↔ ``targets[i]``), the
    resolved target column positions, and the indices of the rows firing
    on at least one target (in increasing order).
    """
    field = fired(X, selector, rng)
    targets = list(_resolve_targets(selector, X, None))
    seed_rows = np.flatnonzero(field.any(axis=1))
    return field, targets, seed_rows


def iter_firing_groups(field, targets, seed_rows):
    """Yield ``(group_rows, fired_cols)`` per distinct firing pattern.

    Rows sharing one identical firing pattern are processed as one
    group; ``fired_cols`` are the target positions that pattern fired.
    Patterns firing nowhere are skipped (nothing to perturb).
    """
    patterns, group_of_row = np.unique(
        field[seed_rows], axis=0, return_inverse=True)
    for gi, pattern in enumerate(patterns):
        group_rows = np.flatnonzero(group_of_row == gi)
        fired_cols = [targets[j] for j in np.flatnonzero(pattern)]
        if not fired_cols:
            continue
        yield group_rows, fired_cols
