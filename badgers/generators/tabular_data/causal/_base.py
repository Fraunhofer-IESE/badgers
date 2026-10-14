"""Shared base class for the causal data quality defect generators.

Owns the shared plumbing: DAG/column_mapping/perturbation validation, DGP
fitting, mechanism gating (which rows get the defect), and label building.
Subclasses define the defect injection and propagation.
"""

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.causal_graph import (
    get_ancestors,
    get_descendants,
    topological_order,
    validate_dag,
)
from badgers.core.dgp import OlsDGP, do
from badgers.core.selectors import BernoulliSelector, TopQuantileSelector


class CausalGenerator(GeneratorMixin):
    """Base class for causal defect generators.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance. If None, the default OLS backend (``OlsDGP``) is
        created on demand and fitted lazily by ``_require_dgp``.
    """

    #: Subclasses set this to the deficit name used in labels.
    deficit_name = "causal"

    def __init__(self, random_generator=default_rng(seed=0), dgp=None):
        self.random_generator = random_generator
        self.dgp = dgp
        self.graph = None
        self.column_mapping = None

    # -- DGP plumbing -------------------------------------------------------

    def fit(self, X, graph, column_mapping, dgp=None):
        """Fit phase: store the structure; fit an injected DGP if given.

        Stores ``self.graph`` / ``self.column_mapping`` so ``generate()``
        does not need them. A DGP backend passed as ``dgp`` is fitted here
        (explicit fit phase); when ``dgp`` is None, the default ``OlsDGP``
        is created lazily by ``_require_dgp`` on first use.
        """
        self.graph = graph
        self.column_mapping = column_mapping
        if dgp is not None:
            self.dgp = dgp
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

    def _require_dgp(self, X=None):
        """Require a fitted DGP; create the default OLS backend on demand.

        Call ``_require_dgp(X)`` before any column mutation so the default
        backend is fitted on the ORIGINAL data. Generators that need no
        structural equation simply never call it (no DGP is created).
        """
        if self.dgp is None:
            if X is None:
                raise RuntimeError(
                    "This generator needs a fitted DGP: call fit(X, graph, "
                    "column_mapping, dgp=<backend>) or use_dgp(dgp) before "
                    "generate()."
                )
            self.dgp = OlsDGP(self.graph, self.column_mapping).fit(X)

    def propagate(self, rows, perturbation_nodes, rng, rows_original=None,
                  mode="delta"):
        """Cascade an applied perturbation to the descendants.

        Delegates to the library-level do-operation (backend-blind):
        ``mode="delta"`` preserves each node's exogenous shock exactly
        (requires ``rows_original``); ``mode="recompute"`` redraws
        descendant noise (requires ``rng``).
        """
        return do(self.dgp, rows, perturbation_nodes, mode=mode, rng=rng,
                  rows_original=rows_original)

    def _sample(self, n_samples):
        """Delegate sampling to the fitted DGP."""
        if self.dgp is None:
            raise RuntimeError(
                "_sample requires a fitted DGP: call fit(X, graph, "
                "column_mapping, dgp=<backend>) first."
            )
        return self.dgp.sample(self.random_generator, n_samples)

    # -- Mechanism gating (Distinction 3) -----------------------------------

    def _fire_masks(
        self,
        rows,
        perturbation_nodes,
        node_to_col,
        mechanism,
        mechanism_cause,
        mechanism_rate,
        stats_rows=None,
    ):
        """Fire masks for ALL rows at once: dict {node: bool row-mask}.

        The vectorized, chooser-based mechanism dispatcher: one dict of
        bool row-masks per node (quantile thresholds computed ONCE).
        Every random mechanism is backed by a preset from
        ``badgers.core.selectors``.

        ``mechanism="random"`` delegates to the ``BernoulliSelector``
        preset (one vectorized ``(n_rows, n_variables)`` draw, whose
        row-major fill reproduces the legacy per-row draws exactly).
        ``mechanism="observed"`` and ``mechanism="self"`` delegate to the
        ``TopQuantileSelector`` preset (driver mode / self mode; a
        deterministic threshold rule at the ``1 - mechanism_rate``
        quantile, so no draw-order concerns). Only
        ``mechanism="always"`` is computed inline (all-True masks).

        Parameters
        ----------
        rows : array to gate (the generator's working rows -- for the noise
            generator these are the n_copies-EXPANDED rows).
        perturbation_nodes : list of str, non-empty (pre-validated).
        node_to_col : column_mapping dict (node name -> column index).
        mechanism, mechanism_cause, mechanism_rate : the legacy knobs.
        stats_rows : array or None, default None
            Rows on which quantile thresholds are computed; defaults to
            ``rows``. The noise generator passes the UN-EXPANDED input so
            expanded rows are gated against the original columns'
            statistics (legacy behavior).

        Note: masks are computed against ``rows`` BEFORE any mutation
        (callers write NaNs/noise afterwards), which makes the gate
        order-independent -- the "compute ALL masks first" invariant the
        legacy missingness path already documented.
        """
        n_rows = rows.shape[0]
        stats = rows if stats_rows is None else stats_rows
        if mechanism == "random":
            selector = BernoulliSelector(
                tuple(perturbation_nodes), rng=self.random_generator,
            )
            return selector(rows, None, rate=mechanism_rate)
        if mechanism == "always":
            return {
                node: np.ones(n_rows, dtype=bool)
                for node in perturbation_nodes
            }
        if mechanism == "observed":
            if mechanism_cause is None:
                raise ValueError(
                    "mechanism_cause is required for mechanism='observed'"
                )
            # legacy semantics: mechanism_rate = TOP FRACTION that fires ->
            # quantile threshold = 1 - mechanism_rate
            selector = TopQuantileSelector(
                tuple(perturbation_nodes), node_to_col,
                drivers=mechanism_cause,
            )
            return selector(rows, None, quantile=1.0 - mechanism_rate,
                            stats=stats)
        if mechanism == "self":
            selector = TopQuantileSelector(
                tuple(perturbation_nodes), node_to_col,
            )
            return selector(rows, None, quantile=1.0 - mechanism_rate,
                            stats=stats)
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
        Otherwise returns ``y`` unchanged (the defect does not change targets).
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
