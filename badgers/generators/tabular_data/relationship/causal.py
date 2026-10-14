"""Causal edge-perturbation generator (relationship-level defect).

Perturbs the fitted coefficient of a single edge (beta -> beta' =
weight_factor * beta). The structural equation of the child itself changes,
so the child is RECOMPUTED through the scaled mechanism (variance
marginal-preserving); the child's descendants are then propagated as
deltas, preserving each descendant's own exogenous shock.
"""

import numpy as np

from badgers.core.causal_graph import get_parents
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.dgp import ScaledEdgeDGP, do
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
    perturbation_nodes : list of str
        Must be exactly ``[child]`` -- the child of the perturbed edge.
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
        edge=None,
        weight_factor=2.0,
    ):
        """Inject a relationship-level (edge weight) perturbation in place."""
        if edge is None:
            raise ValueError("edge parameter is required")
        parent, child = edge
        self._require_fitted()
        self._require_dgp(X)
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

        # Recompute the child through the SCALED edge (raises ValueError if
        # the variance budget is infeasible), then delta-cascade its change
        # to its descendants (each descendant keeps its own exogenous shock).
        rows_original = X.copy()
        self.propagate(
            X, perturbation_nodes, self.random_generator,
            edge=edge, weight_factor=weight_factor,
            rows_original=rows_original,
        )

        affected = np.ones(n_samples, dtype=bool)
        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt

    def propagate(self, rows, perturbation_nodes, rng, **kwargs):
        """Recompute the child through the scaled edge, then delta-cascade.

        Overrides the default delta propagation: the edge weight itself
        changes (relationship-level defect), so the child's value is
        recomputed via the scaled mechanism; its descendants follow with
        the standard delta pass (their own exogenous shocks preserved).
        """
        edge = kwargs["edge"]
        weight_factor = kwargs["weight_factor"]
        rows_original = kwargs.get("rows_original")
        if rows_original is None:
            raise ValueError(
                "edge perturbation requires rows_original (a copy of the "
                "pre-perturbation rows)"
            )
        parent, child = edge
        scaled = ScaledEdgeDGP(self.dgp, child, parent, weight_factor)
        # Raises ValueError('...infeasible...') when the variance budget fails.
        scaled.calibrate(rows)
        pcols = [self.column_mapping[p] for p in get_parents(self.graph, child)]
        child_new = (
            scaled.response(child, rows[:, pcols])
            + scaled.draw_noise(child, rng, rows.shape[0])
        )
        return do(self.dgp, rows, [child], values=[child_new], mode="delta",
                  rows_original=rows_original)
