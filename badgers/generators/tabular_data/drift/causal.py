"""Causal drift generator.

Injects a deterministic distributional shift at the cells selected by the
selector (applied to all affected rows) and recomputes the descendants of the
affected rows through the structural equations (Pearl surgery: each descendant is
re-evaluated from its shifted parents with fresh noise).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import get_selected_fields
from badgers.core.selectors.base import _resolve_targets
from badgers.generators.tabular_data.causal._base import (
    CausalGenerator,
    _column_stds,
    _param_by_target,
)


class CausalDriftGenerator(CausalGenerator):
    """Generates causally-propagated systematic drift.

    The SELECTOR is required and owns the intervention targets: the
    selector's own ``columns`` are the nodes where the drift is injected
    (``selector=None`` raises).

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator used for the fresh noise drawn during the
        Pearl-surgery recompute of descendant nodes.
    dgp : DGP or None, default=None
        DGP instance (passed via the constructor or use_dgp(); fitted
        explicitly by the caller, or lazily inside generate() if
        unfitted).

    generate() parameters (passed via **params)
    ----------
    selector : Selector
        Row-selection rule from ``badgers.core.selectors`` (REQUIRED;
        ``None`` raises -- there is no relevant default: the selector
        owns the intervention targets).
        ``AlwaysSelector``, ``BernoulliSelector``, ``ExactCountSelector``,
        ``TopQuantileSelector``, or any callable with the same signature.
        Its ``columns`` are the intervention targets; the rows it selects
        receive the shift.
    shift_amount : float or dict, default=0.5
        Signed shift applied to the selected target cells, in units of the
        column's standard deviation. Positive shifts in the + direction,
        negative in the − direction. Deterministic: the exact shift value
        is applied to all affected rows (no random draw). Scalar
        (broadcast to every target) or a dict keyed by COLUMN POSITION
        that must cover every target.
    """

    deficit_name = "drift"

    def get_drift_labels(self) -> np.ndarray:
        """Return the drift labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the drift deficit, False for
        original rows.
        """
        return self._get_deficit_labels()

    @preprocess_inputs
    def generate(
        self,
        X,
        selector,
        shift_amount=0.5,
    ):
        """Inject a causally-propagated deterministic drift in place."""
        self._require_fitted()
        if selector is None:
            raise ValueError(
                "selector is required: it owns the intervention targets"
            )
        self._require_dgp(X)
        self._validate_structure(X)
        targets = list(_resolve_targets(selector, X, None))
        shift = _param_by_target(shift_amount, "shift_amount", targets)
        rng = self.random_generator

        # Apply the signed shift (deterministic: no random draw) to the
        # rows selected by the selector. The selected field is decided on the
        # UN-mutated data (thresholds/selection read the input as-is).
        field = get_selected_fields(X, selector, rng)
        rows = np.flatnonzero(field.any(axis=1))
        col_stds = _column_stds(X)
        affected = np.zeros(X.shape[0], dtype=bool)
        for i, col in enumerate(targets):
            mask = field[:, i]
            # Positive = shift in the + direction, negative = in the −
            # direction.
            X[mask, col] += shift[col] * col_stds[col]
            affected |= mask

        # Recompute the descendants of the AFFECTED rows through the
        # structural equations (Pearl surgery): descendant = f(new parents)
        # + fresh noise. The shift at the target nodes is systematic
        # (a single shift applied to all affected rows); unaffected rows
        # are left untouched. Intervene receives ALL target names, selected
        # or not.
        if rows.size:
            sub = X[rows]
            sub = self.dgp.intervene(sub, self._node_names(targets), rng=rng)
            X[rows] = sub

        self.deficit_labels_ = affected.copy()
        return X
