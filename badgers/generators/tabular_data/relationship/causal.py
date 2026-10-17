"""Causal edge-perturbation generator (relationship-level defect).

Perturbs the fitted coefficient of a single edge (beta -> beta' =
weight_factor * beta). The structural equation of the child itself changes,
so the child is RECOMPUTED through the scaled mechanism (variance
marginal-preserving); the child's descendants are then recomputed through
the base mechanisms (Pearl surgery, fresh noise).
"""

import numpy as np

from badgers.core.causal_graph import get_parents
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.dgp.edits import scale_edge
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalEdgePerturbationGenerator(CausalGenerator):
    """Generates a relationship-level (edge weight) perturbation.

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
        # the variance budget is infeasible), then recompute its descendants
        # through the base mechanisms (Pearl surgery, fresh noise).
        scaled = scale_edge(self.dgp, child, parent, weight_factor)
        # Raises ValueError('...infeasible...') when the variance budget fails.
        scaled.calibrate(X)
        pcols = [self.column_mapping[p] for p in get_parents(self.graph, child)]
        child_new = (
            scaled.response(child, X[:, pcols])
            + scaled.draw_noise(child, self.random_generator, n_samples)
        )
        self.dgp.intervene(X, [child], values=[child_new],
                           rng=self.random_generator)

        affected = np.ones(n_samples, dtype=bool)
        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt
