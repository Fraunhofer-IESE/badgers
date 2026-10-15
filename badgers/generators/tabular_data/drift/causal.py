"""Causal drift generator.

Injects a systematic distributional shift at perturbation nodes (drawn once,
applied to all rows) and propagates it to descendants via the structural
equations. The shift at the perturbation node is systematic (not per-row
random), but each descendant still has its own exogenous residual noise,
which is re-added during the forward pass.
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
        DGP instance (attached via fit() or use_dgp()).

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes, mechanism, mechanism_cause, mechanism_rate :
        see CausalGenerator.
    drift_std : float or np.ndarray, default=0.5
        Std of the systematic shift, relative to column std. Scalar or
        per-perturbation-node.
    """

    deficit_name = "drift"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        mechanism="always",
        mechanism_cause=None,
        mechanism_rate=0.5,
        drift_std=0.5,
    ):
        """Inject causally-propagated systematic drift in place."""
        self._require_fitted()
        self._require_dgp(X)
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]
        rows_original = X.copy()
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

        # Apply the shift to the perturbation nodes (gated by the mechanism:
        # fire masks for ALL rows at once, thresholds on the un-mutated data).
        fires = self._fire_masks(
            X, perturbation_nodes, node_to_col, mechanism, mechanism_cause,
            mechanism_rate,
        )
        affected = np.zeros(n_samples, dtype=bool)
        for node in perturbation_nodes:
            mask = fires[node]
            col = node_to_col[node]
            X[mask, col] += shifts[col]
            affected |= mask

        # Propagate the delta to the descendants via the structural
        # equations: descendant_new = descendant_orig + beta @ delta_parents.
        # The descendant's own exogenous noise is preserved (it lives in
        # rows_original); only the drift at the perturbation node is
        # systematic (a single shift applied to all affected rows).
        self.propagate(X, perturbation_nodes, self.random_generator,
                       rows_original=rows_original)

        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt
