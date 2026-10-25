"""Shared selection plumbing for the outliers generators.

The consumers never inspect a selector's specs directly: they realize
the selected field through :func:`badgers.core.selectors.get_selected_fields` and act
on seed rows and selection patterns. The direct (seed-copy) and the causal
(surgery) outlier generators share this plumbing.
"""

import numpy as np

from badgers.core.selectors import get_selected_fields
from badgers.core.selectors.base import _resolve_targets


def select_and_seed(X, selector, rng):
    """Realize the selector on ``X``: selected field, targets, seed rows.

    Returns ``(field, targets, seed_rows)``: the narrow bool field
    ``(n_rows, len(targets))`` (column ``i`` ↔ ``targets[i]``), the
    resolved target column positions, and the indices of the rows selected
    on at least one target (in increasing order).
    """
    field = get_selected_fields(X, selector, rng)
    targets = list(_resolve_targets(selector, X, None))
    seed_rows = np.flatnonzero(field.any(axis=1))
    return field, targets, seed_rows


def iter_selection_groups(field, targets, seed_rows):
    """Yield ``(group_rows, selected_cols)`` per distinct selection pattern.

    Rows sharing one identical selection pattern are processed as one
    group; ``selected_cols`` are the target positions that pattern selected.
    Patterns selecting nothing are skipped (nothing to perturb).
    """
    patterns, group_of_row = np.unique(
        field[seed_rows], axis=0, return_inverse=True)
    for gi, pattern in enumerate(patterns):
        group_rows = np.flatnonzero(group_of_row == gi)
        selected_cols = [targets[j] for j in np.flatnonzero(pattern)]
        if not selected_cols:
            continue
        yield group_rows, selected_cols
