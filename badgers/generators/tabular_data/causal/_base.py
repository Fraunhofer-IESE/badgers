"""Shared base class for the causal data quality defect generators.

Owns the shared plumbing: DAG/column_mapping/perturbation validation, DGP
fitting, mechanism gating (which rows get the defect), and label building.
Subclasses define the defect injection and propagation.
"""

import networkx as nx
import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.causal_graph import (
    get_ancestors,
    get_descendants,
    get_parents,
    topological_order,
    validate_dag,
)
from badgers.core.dgp import PyMCDGP


class CausalGenerator(GeneratorMixin):
    """Base class for causal defect generators.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        A fitted-or-fittable DGP. If None, a ``PyMCDGP`` is created on demand.
    """

    #: Subclasses set this to the deficit name used in labels.
    deficit_name = "causal"

    #: Whether the propagation recomputes the child's value from the
    #: structural equation (recompute semantics, used by edge perturbation
    #: where the equation itself changes) or adds the propagated delta on
    #: top of the original values (delta semantics, used by noise, drift
    #: and imbalance). Delta semantics preserves each node's exogenous
    #: shock exactly: v_new = v_orig + beta @ (parents_new - parents_old).
    delta_propagation = True

    def __init__(self, random_generator=default_rng(seed=0), dgp=None):
        self.random_generator = random_generator
        self.dgp = dgp
        self.graph = None
        self.column_mapping = None
        self.affected_mask_ = None

    # -- DGP plumbing -------------------------------------------------------

    def fit(self, X, graph, column_mapping, n_samples_mcmc=1000, fit_dgp=True):
        """Fit the DGP on the ORIGINAL data (explicit fit phase).

        Stores ``self.graph`` / ``self.column_mapping`` so ``generate()`` does
        not need them. Set ``fit_dgp=False`` for generators that do not need
        the structural equations (e.g. missingness with rule="any").
        """
        self.graph = graph
        self.column_mapping = column_mapping
        if fit_dgp:
            if self.dgp is None:
                self.dgp = PyMCDGP(
                    graph, column_mapping, n_samples_mcmc=n_samples_mcmc
                )
            self.dgp.fit(X)

    def use_dgp(self, dgp):
        """Attach an already-fitted DGP (shared across a pipeline)."""
        self.dgp = dgp
        self.graph = dgp.graph
        self.column_mapping = dgp.column_mapping

    def _require_fitted(self):
        if self.graph is None:
            raise RuntimeError(
                "Generator is not fitted: call fit(X, graph, column_mapping) "
                "or use_dgp(dgp) before generate()."
            )

    def _require_dgp(self):
        if self.dgp is None:
            raise RuntimeError(
                "This generator needs a fitted DGP: call fit(...) with "
                "fit_dgp=True, or use_dgp(dgp)."
            )

    def _ordered_descendants(self, perturbation_nodes):
        """Return the descendants of the perturbation nodes, topologically ordered."""
        descendants = set()
        for node in perturbation_nodes:
            descendants |= get_descendants(self.dgp.graph, node)
        order = topological_order(self.dgp.graph)
        return [node for node in order if node in descendants]

    def propagate(self, rows, perturbation_nodes, rng, **kwargs):
        """Propagate the perturbation to the descendants of the nodes.

        Default implementation: **delta propagation** (``delta_propagation``
        is True). For each descendant node ``v`` in topological order:

            delta_v = beta_v @ (parents_new - parents_orig)
            rows[v] = rows_orig[v] + delta_v

        This preserves each node's own exogenous shock exactly (it lives in
        ``rows_orig``) and leaves a node untouched whenever its parents did
        not change. ``rows`` holds the ORIGINAL values with only the
        perturbation applied to the perturbation-node columns; the
        pre-perturbation values must be passed as ``rows_original``.

        Generators with different propagation semantics override this
        method: missingness (absorbing NaN mask pass) and edge perturbation
        (recompute the child with the modified coefficient, delta for its
        descendants).
        """
        if not self.delta_propagation:
            return self._propagate_recompute(rows, perturbation_nodes)
        rows_original = kwargs.get("rows_original")
        if rows_original is None:
            raise ValueError(
                "delta propagation requires rows_original: the "
                "pre-perturbation values of rows (callers mutate rows in "
                "place, so pass a copy taken before the perturbation)"
            )
        return self._propagate_delta(rows, rows_original, perturbation_nodes)

    def _propagate_delta(self, rows, rows_original, perturbation_nodes):
        """Delta pass: add propagated deltas on top of the original values.

        For each descendant node v (topological order):
            rows[v] = rows_original[v] + beta_v @ (rows[parents] -
                                                   rows_original[parents])
        """
        graph = self.graph
        for node in self._ordered_descendants(perturbation_nodes):
            col = self.column_mapping[node]
            parents = get_parents(graph, node)
            parent_cols = [self.column_mapping[p] for p in parents]
            beta = self.dgp._beta_mean[node]
            parent_delta = rows[:, parent_cols] - rows_original[:, parent_cols]
            delta = parent_delta @ beta
            rows[:, col] = rows_original[:, col] + delta
        return rows

    def _propagate_recompute(self, rows, perturbation_nodes):
        """Value forward pass over the descendants (recompute semantics).

        Recomputes each descendant from its (possibly perturbed) parents:
            rows[v] = beta_v @ rows[parents] + eps_new
        Used when the structural equation itself changes (edge perturbation).
        """
        nodes = self._ordered_descendants(perturbation_nodes)
        return self.dgp.forward_pass(
            rows, nodes, self.random_generator, add_residual=True
        )

    def _sample(self, n_samples):
        """Delegate sampling to the DGP."""
        return self.dgp.sample(self.random_generator, n_samples)

    # -- Mechanism gating (Distinction 3) -----------------------------------

    def _should_defect(
        self,
        latent_row,
        mechanism,
        mechanism_cause,
        mechanism_rate,
        perturbation_nodes,
        column_mapping,
        X,
    ):
        """Return a boolean mask (one per perturbation node) for this row.

        ``True`` means the defect fires at that perturbation node.

        - ``always``: all True.
        - ``random``: each independently True with probability ``mechanism_rate``.
        - ``observed``: all True if ``mechanism_cause`` is in its top
          ``mechanism_rate`` fraction.
        - ``self``: each perturbation node independently True if its own value
          is in its top ``mechanism_rate`` fraction.
        """
        n_perturb = len(perturbation_nodes)
        if mechanism == "always":
            return np.ones(n_perturb, dtype=bool)
        if mechanism == "random":
            return self.random_generator.random(n_perturb) < mechanism_rate
        if mechanism == "observed":
            if mechanism_cause is None:
                raise ValueError(
                    "mechanism_cause is required for mechanism='observed'"
                )
            cause_col = column_mapping[mechanism_cause]
            threshold = np.quantile(X[:, cause_col], 1.0 - mechanism_rate)
            fires = latent_row[cause_col] >= threshold
            return np.full(n_perturb, bool(fires), dtype=bool)
        if mechanism == "self":
            mask = np.zeros(n_perturb, dtype=bool)
            for i, node in enumerate(perturbation_nodes):
                col = column_mapping[node]
                threshold = np.quantile(X[:, col], 1.0 - mechanism_rate)
                mask[i] = bool(latent_row[col] >= threshold)
            return mask
        raise ValueError(f"Unknown mechanism '{mechanism}'")

    # -- Validation ---------------------------------------------------------

    def _validate(self, X, perturbation_nodes):
        """Validate inputs and return the working sets.

        Uses ``self.graph`` / ``self.column_mapping`` (set by ``fit()``).

        Returns
        -------
        (order, node_to_col, descendants, exogenous, perturb_cols)
        """
        graph = self.graph
        column_mapping = self.column_mapping
        if graph is None:
            raise ValueError(
                "graph is not set: call fit(X, graph, column_mapping) first"
            )
        validate_dag(graph)
        if perturbation_nodes is None:
            raise ValueError("perturbation_nodes parameter is required")
        if not isinstance(perturbation_nodes, list):
            raise ValueError(
                "perturbation_nodes must be a list, "
                f"got {type(perturbation_nodes).__name__}"
            )
        if len(perturbation_nodes) == 0:
            raise ValueError("perturbation_nodes must not be empty")
        for node in perturbation_nodes:
            if node not in graph.nodes:
                raise ValueError(
                    f"perturbation node '{node}' not found in graph"
                )
        n_samples, n_features = X.shape
        if n_features != len(graph.nodes):
            raise ValueError(
                f"X has {n_features} columns but graph has "
                f"{len(graph.nodes)} nodes"
            )
        if column_mapping is None:
            raise ValueError("column_mapping parameter is required")
        if not isinstance(column_mapping, dict):
            raise ValueError(
                "column_mapping must be a dict, "
                f"got {type(column_mapping).__name__}"
            )
        for node in graph.nodes:
            if not isinstance(node, str):
                raise ValueError(
                    f"graph nodes must be strings, "
                    f"got {type(node).__name__} for node '{node}'"
                )
        missing = set(graph.nodes) - set(column_mapping.keys())
        if missing:
            raise ValueError(
                f"column_mapping missing node(s): {sorted(missing)}"
            )
        extra = set(column_mapping.keys()) - set(graph.nodes)
        if extra:
            raise ValueError(
                f"column_mapping has unknown node(s): {sorted(extra)}"
            )
        for node, idx in column_mapping.items():
            if not isinstance(idx, int):
                raise ValueError(
                    f"column_mapping values must be int, "
                    f"got {type(idx).__name__} for node '{node}'"
                )
            if idx < 0 or idx >= n_features:
                raise ValueError(
                    f"column index {idx} for node '{node}' "
                    f"is out of range [0, {n_features - 1}]"
                )
        seen = {}
        for node, idx in column_mapping.items():
            if idx in seen:
                raise ValueError(
                    f"duplicate column index {idx} "
                    f"for nodes {sorted([seen[idx], node])}"
                )
            seen[idx] = node

        order = topological_order(graph)
        node_to_col = column_mapping
        descendants = set()
        for pn in perturbation_nodes:
            descendants |= get_descendants(graph, pn)
        ancestors = set()
        for pn in perturbation_nodes:
            ancestors |= get_ancestors(graph, pn)
        perturb_set = set(perturbation_nodes)
        exogenous = ancestors - perturb_set - descendants
        perturb_cols = [node_to_col[n] for n in perturbation_nodes]
        return order, node_to_col, descendants, exogenous, perturb_cols

    # -- Sampler + labels ---------------------------------------------------

    def _labels_inplace(self, y, affected_mask, deficit_name):
        """Labels for an in-place defect (same length as the output rows).

        If ``y`` is None, returns a label array ("original"/deficit name).
        Otherwise returns ``y`` unchanged (the defect does not change targets);
        the affected-row mask is available as ``self.affected_mask_``.
        """
        if y is None:
            return np.where(affected_mask, deficit_name, "original")
        return y

    def _labels_expanded(self, y, affected_mask, deficit_name):
        """Labels for an expanded defect (one per output row).

        If ``y`` is None, returns a label array ("original"/deficit name).
        Otherwise returns ``y`` unchanged (already expanded by the caller).
        """
        if y is None:
            return np.where(affected_mask, deficit_name, "original")
        return y
