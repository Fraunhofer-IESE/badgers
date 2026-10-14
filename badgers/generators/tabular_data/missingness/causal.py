"""Causal missingness generator.

Injects missingness (NaN) at perturbation nodes and propagates it to
descendants via a topological mask pass. Two orthogonal axes:

- **Cause** (``mechanism``): why/when a value is missing — the m-graph $R$
  mechanism (``random``=MCAR, ``observed``=MAR, ``self``=MNAR).
- **Propagation** (``missingness_rule``): how missingness spreads —
  ``"any"`` (non-recoverable) or ``"all"`` (recoverable).
"""

import numpy as np

from badgers.core.causal_graph import (
    get_descendants,
    get_parents,
    topological_order,
)
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalMissingnessGenerator(CausalGenerator):
    """Generates causally-propagated missingness.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance. Used when ``missingness_rule="all"`` (recoverable
        children are computed from observed parents).

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes : list of str
        Nodes where missingness is injected.
    mechanism : str, default="random"
        Cause of missingness: ``"always"``, ``"random"`` (MCAR),
        ``"observed"`` (MAR), ``"self"`` (MNAR).
    mechanism_cause : str, default=None
        Fully-observed node driving missingness (required for ``"observed"``).
    mechanism_rate : float, default=0.3
        Base missingness probability / range fraction.
    missingness_rule : str, default="any"
        ``"any"`` (non-recoverable): child missing if any parent missing.
        ``"all"`` (recoverable): child missing only if all parents missing.
    """

    deficit_name = "missingness"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        mechanism="random",
        mechanism_cause=None,
        mechanism_rate=0.3,
        missingness_rule="any",
    ):
        """Inject causally-propagated missingness in place."""
        if missingness_rule not in ("any", "all"):
            raise ValueError(
                f"missingness_rule must be 'any' or 'all', got '{missingness_rule}'"
            )
        self._require_fitted()
        if missingness_rule == "all":
            self._require_dgp(X)
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]

        # Gate the defect by the mechanism: fire masks for ALL rows at once
        # (vectorized; quantile thresholds computed on the un-mutated data).
        fires = self._fire_masks(
            X, perturbation_nodes, node_to_col, mechanism, mechanism_cause,
            mechanism_rate,
        )
        affected = np.zeros(n_samples, dtype=bool)
        for node in perturbation_nodes:
            mask = fires[node]
            X[mask, node_to_col[node]] = np.nan
            affected |= mask

        # Propagate the mask to descendants (absorbing NaN mask pass).
        self.propagate(
            X, perturbation_nodes, self.random_generator,
            missingness_rule=missingness_rule,
        )

        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt

    def propagate(self, rows, perturbation_nodes, rng, **kwargs):
        """Absorbing NaN mask pass (overrides the default delta propagation).

        Missingness does not flow through the structural equation: a child is
        missing if any parent is missing (``rule="any"``, non-recoverable) or
        only if all parents are missing (``rule="all"``, recoverable).
        """
        missingness_rule = kwargs.get("missingness_rule", "any")
        order = topological_order(self.graph)
        descendants = set()
        for node in perturbation_nodes:
            descendants |= get_descendants(self.graph, node)
        descendant_cols = {self.column_mapping[n] for n in descendants}
        for node in order:
            col = self.column_mapping[node]
            if col not in descendant_cols:
                continue
            parents = get_parents(self.graph, node)
            if not parents:
                continue
            parent_cols = [self.column_mapping[p] for p in parents]
            parent_missing = np.isnan(rows[:, parent_cols])
            if missingness_rule == "any":
                child_missing = parent_missing.any(axis=1)
            else:  # "all"
                child_missing = parent_missing.all(axis=1)
            rows[child_missing, col] = np.nan
            # Recoverable children (rule="all") with >=1 observed parent are
            # computed from the observed parents via the structural equation.
            if missingness_rule == "all":
                recoverable_idx = np.where(~child_missing)[0]
                if len(recoverable_idx) > 0:
                    self._compute_recoverable(
                        rows, recoverable_idx, node, parents,
                        self.column_mapping,
                    )
        return rows

    def _compute_recoverable(self, rows, row_idx, node, parents, node_to_col):
        """Recompute recoverable child values from their observed parents.

        Uses the DGP's fitted mechanism directly: predicted =
        f(node, parent values), with missing parents zeroed (linear
        zero-intercept convention) plus the node's own redrawn noise for the
        unexplained variance.
        """
        col = node_to_col[node]
        scol = [node_to_col[p] for p in parents]
        # Only use observed (non-NaN) parents in the computation.
        parent_vals = rows[row_idx][:, scol].copy()
        parent_vals[np.isnan(parent_vals)] = 0.0
        predicted = self.dgp.response(node, parent_vals)
        noise = self.dgp.draw_noise(node, self.random_generator, len(row_idx))
        rows[row_idx, col] = predicted + noise
