"""Causal edge-perturbation generator (relationship-level defect).

Perturbs the fitted coefficient of a single edge (beta -> beta' =
weight_factor * beta) and recomputes the child and its descendants. The child's
residual variance is adjusted so the child's marginal variance is preserved
(marginal-preserving perturbation).
"""

import numpy as np

from badgers.core.causal_graph import (
    get_descendants,
    get_parents,
    topological_order,
)
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalEdgePerturbationGenerator(CausalGenerator):
    """Generates a relationship-level (edge weight) perturbation.

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
    edge : tuple of (str, str), required
        The (parent, child) edge to perturb.
    weight_factor : float, default=2.0
        Multiplier applied to the fitted coefficient (beta -> beta' =
        weight_factor * beta).
    """

    deficit_name = "relationship"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        mechanism="always",
        mechanism_cause=None,
        mechanism_rate=0.5,
        edge=None,
        weight_factor=2.0,
    ):
        """Inject a relationship-level (edge weight) perturbation in place."""
        if edge is None:
            raise ValueError("edge parameter is required")
        parent, child = edge
        self._require_fitted()
        self._require_dgp()
        if not self.graph.has_edge(parent, child):
            raise ValueError(f"edge ({parent}, {child}) not in graph")
        if perturbation_nodes != [child]:
            raise ValueError(
                f"perturbation_nodes must be [{child!r}] (the child of the "
                f"perturbed edge), got {perturbation_nodes!r}"
            )
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]

        # Fitted coefficient for the edge (parent -> child).
        child_parents = list(self.graph.predecessors(child))
        parent_idx = child_parents.index(parent)
        beta = self.dgp._beta_mean[child][parent_idx]
        beta_prime = weight_factor * beta

        # Marginal-preserving: adjust the child's residual variance so the
        # child's marginal variance is preserved.
        child_col = node_to_col[child]
        parent_col = node_to_col[parent]
        var_child = np.var(X[:, child_col])
        var_parent = np.var(X[:, parent_col])
        # New residual variance: var_child - beta'^2 * var_parent
        new_resid_var = var_child - (beta_prime ** 2) * var_parent
        if new_resid_var < 0:
            raise ValueError(
                f"Edge perturbation is infeasible: beta'^2 * var({parent}) "
                f"> var({child}). Reduce weight_factor."
            )
        new_resid_std = np.sqrt(new_resid_var)

        # Recompute the child with beta' and the adjusted residual, then
        # propagate to the child's descendants.
        self.propagate(
            X, perturbation_nodes, self.random_generator,
            edge=edge, beta_prime=beta_prime, new_resid_std=new_resid_std,
        )

        affected = np.ones(n_samples, dtype=bool)
        self.affected_mask_ = affected
        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt

    def propagate(self, rows, perturbation_nodes, rng, **kwargs):
        """Recompute the child with beta' and propagate to its descendants.

        Overrides the default forward pass because the edge weight itself
        changes (relationship-level defect), not just a value.
        """
        edge = kwargs["edge"]
        beta_prime = kwargs["beta_prime"]
        new_resid_std = kwargs["new_resid_std"]
        parent, child = edge
        order = topological_order(self.graph)
        child_descendants = get_descendants(self.graph, child)
        for node in order:
            col = self.column_mapping[node]
            if node != child and node not in child_descendants:
                continue
            parents = get_parents(self.graph, node)
            if not parents:
                continue
            parent_cols = [self.column_mapping[p] for p in parents]
            coefs = np.array([
                (beta_prime if p == parent else self._edge_beta(node, p))
                for p in parents
            ])
            predicted = rows[:, parent_cols] @ coefs
            if node == child:
                resid_std = new_resid_std
            else:
                resid_std = self.dgp._residual_std.get(node, 0.0)
            rows[:, col] = predicted + rng.normal(
                0, resid_std, size=rows.shape[0]
            )
        return rows

    def _edge_beta(self, node, parent):
        """Fitted coefficient for edge (parent -> node) from the DGP."""
        node_parents = list(self.graph.predecessors(node))
        idx = node_parents.index(parent)
        return self.dgp._beta_mean[node][idx]
