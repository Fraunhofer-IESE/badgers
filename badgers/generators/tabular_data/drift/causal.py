"""Causal drift generator.

Injects a systematic distributional shift at perturbation nodes (drawn once,
applied to all affected rows) and recomputes the descendants of the affected
rows through the structural equations (Pearl surgery: each descendant is
re-evaluated from its shifted parents with fresh noise).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import (
    CausalGenerator,
    _column_stds,
)


class CausalDriftGenerator(CausalGenerator):
    """Generates causally-propagated systematic drift.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (passed via the constructor or use_dgp(); fitted
        explicitly by the caller, or lazily inside generate() if
        unfitted).

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes : list of str
        Nodes where the drift is injected (required).
    selector : Selector or None, default=None
        Row-selection rule from ``badgers.core.selectors``
        (``AlwaysSelector``, ``BernoulliSelector``, ``ExactCountSelector``,
        ``TopQuantileSelector``, or any callable with the same signature).
        ``None`` = every row fires.
    drift_std : float or np.ndarray, default=0.5
        Std of the systematic shift, relative to column std. Scalar or
        per-perturbation-node.
    """

    deficit_name = "drift"

    def get_drift_labels(self) -> np.ndarray:
        """Return the drift labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the drift deficit, False for
        original rows.
        """
        if self.deficit_labels_ is None:
            raise RuntimeError(
                "Drift labels are only available after calling generate()."
            )
        return self.deficit_labels_

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        selector=None,
        drift_std=0.5,
    ):
        """Inject causally-propagated systematic drift in place."""
        self._require_fitted()
        self._require_dgp(X)
        self._validate_structure(X)
        self._validate_perturbation_nodes(perturbation_nodes)
        node_to_col = self.column_mapping
        n_samples = X.shape[0]
        col_stds = _column_stds(X)

        if np.isscalar(drift_std):
            drift_scale = np.full(len(perturbation_nodes), float(drift_std))
        else:
            drift_scale = np.asarray(drift_std, dtype=float)
            if drift_scale.shape != (len(perturbation_nodes),):
                raise ValueError(
                    "drift_std array must have one entry per perturbation node"
                )

        # Draw the systematic shift ONCE (applied to all rows).
        shifts = {}
        for j, node in enumerate(perturbation_nodes):
            col = node_to_col[node]
            shifts[col] = self.random_generator.normal(
                0, drift_scale[j] * col_stds[col]
            )

        # Apply the shift to the perturbation nodes (gated by the selector:
        # fire masks for ALL rows at once, thresholds on the un-mutated data).
        fires = self._fire_masks(
            X, perturbation_nodes, selector=selector,
        )
        affected = np.zeros(n_samples, dtype=bool)
        for node in perturbation_nodes:
            mask = fires[node]
            col = node_to_col[node]
            X[mask, col] += shifts[col]
            affected |= mask

        # Recompute the descendants of the AFFECTED rows through the
        # structural equations (Pearl surgery): descendant = f(new parents)
        # + fresh noise. The shift at the perturbation node is systematic
        # (a single shift applied to all affected rows); unaffected rows
        # are left untouched.
        affected_rows = np.flatnonzero(affected)
        if affected_rows.size:
            sub = X[affected_rows]
            self.dgp.intervene(
                sub, perturbation_nodes, rng=self.random_generator
            )
            X[affected_rows] = sub

        self.deficit_labels_ = affected.copy()
        return X, None
