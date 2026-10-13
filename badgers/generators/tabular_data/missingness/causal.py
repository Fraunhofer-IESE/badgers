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
    indicator_nodes,
    topological_order,
    validate_observation_graph,
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
        DGP instance. Used only when ``missingness_rule="all"`` (recoverable
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
        if self.observation_graph is not None:
            if perturbation_nodes is not None:
                raise ValueError(
                    "perturbation_nodes must be None when an "
                    "observation_graph is provided (the graph declares the "
                    "monitoring)"
                )
            return self._generate_from_observation_graph(
                X, y, mechanism_rate
            )
        if missingness_rule == "all":
            self._require_dgp(X)
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]

        # Gate the defect by the mechanism. Compute ALL masks first, so the
        # quantile thresholds are computed on the un-mutated data.
        masks = [
            self._should_defect(
                X[i], mechanism, mechanism_cause, mechanism_rate,
                perturbation_nodes, node_to_col, X,
            )
            for i in range(n_samples)
        ]
        affected = np.zeros(n_samples, dtype=bool)
        for i, mask in enumerate(masks):
            for j, node in enumerate(perturbation_nodes):
                if mask[j]:
                    X[i, node_to_col[node]] = np.nan
                    affected[i] = True

        # Propagate the mask to descendants (absorbing NaN mask pass).
        self.propagate(
            X, perturbation_nodes, self.random_generator,
            missingness_rule=missingness_rule,
        )

        self.affected_mask_ = affected
        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt

    def _generate_from_observation_graph(self, rows, y, mechanism_rate):
        """Derive the missingness indicators from the observation graph.

        Walks the graph in topological order; each indicator ``R_*`` fires
        per row:

        - no parents (isolated indicator) -> MCAR: fires independently with
          probability ``mechanism_rate``;
        - parents present -> MAR/MNAR: fires on the top ``mechanism_rate``
          fraction of the first driver's column.

        The failure form here is NaN (dropout); other forms (stuck, spike,
        clip, bias) are the deficit's business (future work). Spread follows
        the graph's derivation edges (``_propagate_observation``).
        """
        validate_observation_graph(self.observation_graph, self.graph)
        n_samples = rows.shape[0]
        # Nodes carrying derivation_rule="all" recover values through the
        # fitted mechanism, so a DGP must exist BEFORE indicators fire
        # (recovering from post-NaN data would fit on garbage).
        # [controller pre-fix: without this, the "all" branch hits
        # self.dgp=None -> AttributeError, or would fit on NaN data]
        needs_dgp = any(
            self.observation_graph.nodes[node].get("derivation_rule", "any")
            == "all"
            for node in self.observation_graph.nodes
            if not str(node).startswith("R_")
        )
        if needs_dgp:
            self._require_dgp(rows)
        # Compute ALL firing masks first (against the un-mutated data), then
        # apply the NaN writes -- mirroring the structural path's invariant
        # ("compute ALL masks first") so a later indicator's quantile rule is
        # never evaluated on NaN-contaminated data, no matter the order in
        # which the indicators fire.
        fires_by_indicator = []
        for indicator, monitored in indicator_nodes(self.observation_graph):
            drivers = list(self.observation_graph.predecessors(indicator))
            if not drivers:  # MCAR
                fires = self.random_generator.random(n_samples) < mechanism_rate
            else:            # MAR / MNAR: quantile rule on the first driver
                col = self.column_mapping[drivers[0]]
                threshold = np.quantile(rows[:, col], 1.0 - mechanism_rate)
                fires = rows[:, col] >= threshold
            fires_by_indicator.append((indicator, monitored, fires))
        affected = np.zeros(n_samples, dtype=bool)
        for indicator, monitored, fires in fires_by_indicator:
            rows[fires, self.column_mapping[monitored]] = np.nan
            affected |= fires
        self._propagate_observation(rows)
        self.affected_mask_ = affected
        yt = self._labels_inplace(y, affected, self.deficit_name)
        return rows, yt

    def _propagate_observation(self, rows):
        """Absorbing NaN mask pass along the observation graph's derivation edges.

        Only measurement -> measurement edges participate (trigger edges
        into ``R_*`` nodes are skipped). The target's missingness follows
        its own ``derivation_rule`` attribute (default ``"any"``):

        - ``"any"``: missing if any derivation source is missing;
        - ``"all"``: missing only if ALL sources are missing — and rows
          with at least one observed source are recomputed from the still
          observed sources (``_compute_recoverable``), keeping the deficit
          recoverable.
        """
        for node in topological_order(self.observation_graph):
            if str(node).startswith("R_"):
                continue
            sources = [
                s for s in self.observation_graph.predecessors(node)
                if not str(s).startswith("R_")
            ]
            if not sources:
                continue
            rule = self.observation_graph.nodes[node].get(
                "derivation_rule", "any"
            )
            scol = [self.column_mapping[s] for s in sources]
            ccol = self.column_mapping[node]
            source_missing = np.isnan(rows[:, scol])
            if rule == "any":
                rows[source_missing.any(axis=1), ccol] = np.nan
            else:  # "all"
                all_missing = source_missing.all(axis=1)
                rows[all_missing, ccol] = np.nan
                recoverable = (~all_missing) & source_missing.any(axis=1)
                if recoverable.any():
                    # Structural order for the mechanism call: response()
                    # expects parent columns in graph.predecessors(node)
                    # order, while ``sources`` follow the observation
                    # graph's edge order (they can differ).
                    # [controller pre-fix: passing obs-source order can
                    # silently misalign the mechanism's coefficients]
                    source_set = set(sources)
                    parents = [
                        p for p in get_parents(self.graph, node)
                        if p in source_set
                    ]
                    self._compute_recoverable(
                        rows, np.where(recoverable)[0], node, parents,
                        self.column_mapping,
                    )
        return rows

    def propagate(self, rows, perturbation_nodes, rng, **kwargs):
        """Absorbing NaN mask pass (overrides the default forward pass).

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
