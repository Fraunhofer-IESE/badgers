"""Shared base class for the causal data quality defect generators.

Owns the shared plumbing: DAG/column_mapping/perturbation validation, DGP
configuration (no generator-side ``fit`` — the DGP is fitted by its own
``fit``, explicitly by the caller or lazily inside ``generate()``),
mechanism gating (which rows get the defect), and label building.
Subclasses define the defect injection and propagation.
"""

import inspect
import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.causal_graph import (
    get_ancestors,
    get_descendants,
    topological_order,
    validate_dag,
)
from badgers.core.selectors import AlwaysSelector


def _column_stds(X):
    """NaN-safe per-column stds (G9b): a finite scale for every column.

    NaNs are ignored (``np.nanstd``); a constant or all-NaN column gets a
    scale of 1.0 so a drawn offset stays finite -- a NaN-bearing column
    must never silently poison the whole output column (G9b).
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice
        std = np.nanstd(X, axis=0)
    return np.where(np.isfinite(std) & (std > 0), std, 1.0)


class CausalGenerator(GeneratorMixin):
    """Base class for causal defect generators.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (fitted or unfitted). This is the ONLY structure
        the generator accepts — a DGP carries its own graph and
        column_mapping. There is no graph-only configuration and no
        implicit default backend: if ``dgp`` is None, ``generate()``
        raises (build one with e.g.
        ``GcmSCMDGP(make_gcm_scm(graph, column_mapping))``).

    Notes
    -----
    No generator-side ``fit``, no graph-only configuration
        Causal generators have NO public ``fit()`` — the only fitted
        object is the DGP (``DGP.fit``), consistent with the rest of the
        library. The DGP is fitted either EXPLICITLY by the caller
        (preferred) or lazily INSIDE ``generate()`` when the configured
        DGP is still unfitted (``_require_dgp``).

    Pipelines
        In a pipeline, each causal stage receives the PREVIOUS stage's
        output as input. Fitting the DGP lazily on that contaminated
        input would refit the world per stage (and, with NaN-bearing
        columns, trigger complete-case filtering). Fit the DGP ONCE on
        the clean entry data and share the fitted instance::

            dgp = GcmSCMDGP(make_gcm_scm(graph, column_mapping)).fit(X)  # fit on clean X

            missingness = CausalMissingnessGenerator(rng=..., dgp=dgp)
            Xm, _ = missingness.generate(X, None, ...)

            drift = CausalDriftGenerator(rng=..., dgp=dgp)  # share
            Xt, _ = drift.generate(Xm, None, ...)

        All per-stage scale computations are NaN-safe, so even the lazy
        refit path behaves sensibly; the explicitly-fitted shared
        ``dgp=`` pattern simply guarantees ONE world for the whole
        pipeline. Fit the DGP once on clean data — the backends do not
        impute or filter NaN rows at fit time.
    """

    #: Subclasses set this to the deficit name used in labels.
    deficit_name = "causal"

    def __init__(self, random_generator=default_rng(seed=0), dgp=None):
        self.random_generator = random_generator
        self.dgp = dgp
        self.graph = None if dgp is None else dgp.graph
        self.column_mapping = None if dgp is None else dgp.column_mapping

    # -- DGP plumbing -------------------------------------------------------

    def use_dgp(self, dgp):
        """Attach a DGP (fitted or unfitted); the structure comes with it.

        An unfitted DGP is fitted lazily inside ``generate()`` on the
        generate-time data (``_require_dgp``). Prefer attaching a DGP you
        fitted explicitly on the clean entry data (shared across a
        pipeline).
        """
        self.dgp = dgp
        self.graph = dgp.graph
        self.column_mapping = dgp.column_mapping

    def _require_fitted(self):
        if self.graph is None:
            raise RuntimeError(
                "Generator has no structure: pass a dgp=... to the "
                "constructor, or use use_dgp(dgp), before generate()."
            )

    def _require_dgp(self, X=None):
        """Require a fitted DGP; fit lazily on demand.

        Call ``_require_dgp(X)`` before any column mutation so the DGP is
        fitted on the ORIGINAL data. Two routes:

        1. a fitted DGP was configured/attached — nothing to do;
        2. an UNFITTED DGP was configured/attached — it is fitted here
           on ``X`` (the generate-time data).

        There is NO implicit default backend: if no DGP was given,
        ``generate()`` fails. Build the world explicitly, e.g.
        ``dgp = GcmSCMDGP(make_gcm_scm(graph, column_mapping)).fit(X)``.
        """
        if self.dgp is None:
            raise RuntimeError(
                "This generator needs a DGP: pass a dgp=... to the "
                "constructor, or use use_dgp(dgp), before generate(). "
                "Build one with e.g. dgp = GcmSCMDGP(make_gcm_scm("
                "graph, column_mapping)).fit(X)."
            )
        elif not self.dgp.is_fitted():
            self.dgp.fit(X)

    def _sample(self, n_samples):
        """Delegate sampling to the fitted DGP."""
        if self.dgp is None or not self.dgp.is_fitted():
            raise RuntimeError(
                "_sample requires a fitted DGP: pass a fitted dgp=... to "
                "the constructor (or use_dgp(dgp)) first."
            )
        return self.dgp.sample(self.random_generator, n_samples)

    # -- Mechanism gating (Distinction 3) -----------------------------------

    def _fire_masks(
        self,
        rows,
        perturbation_nodes,
        node_to_col,
        selector=None,
    ):
        """Fire masks for ALL rows at once: dict {node: bool row-mask}.

        Delegates to a ``Selector`` from ``badgers.core.selectors``:
        ``selector(rows, y) -> {node: bool mask}``.
        ``None`` defaults to ``AlwaysSelector`` (every row fires).

        Parameters
        ----------
        rows : array to gate (the generator's working rows -- for the noise
            generator these are the n_copies-EXPANDED rows).
        perturbation_nodes : list of str, non-empty (pre-validated).
        node_to_col : column_mapping dict (node name -> column index).
        selector : Selector or None, default None
            The row-selection rule. ``None`` = ``AlwaysSelector`` (all
            rows fire). Presets: ``BernoulliSelector`` (iid rate),
            ``ExactCountSelector`` (exact count), ``TopQuantileSelector``
            (value threshold, self or driver mode), ``AlwaysSelector``;
            any user callable with the same signature works.

        Note: masks are computed against ``rows`` BEFORE any mutation
        (callers write NaNs/noise afterwards), which makes the gate
        order-independent -- the "compute ALL masks first" invariant the
        legacy missingness path already documented.
        """
        if selector is None:
            return {
                node: np.ones(rows.shape[0], dtype=bool)
                for node in perturbation_nodes
            }
        return selector(rows, None)

    # -- Validation ---------------------------------------------------------

    def _validate(self, X, perturbation_nodes):
        """Validate inputs and return the working sets.

        Uses ``self.graph`` / ``self.column_mapping`` (derived from the
        configured DGP).

        Returns
        -------
        (order, node_to_col, descendants, exogenous, perturb_cols)
        """
        graph = self.graph
        column_mapping = self.column_mapping
        if graph is None:
            raise ValueError(
                "no DGP configured: pass a dgp=... to the constructor, "
                "or use use_dgp(dgp) (the DGP carries graph and "
                "column_mapping)"
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
