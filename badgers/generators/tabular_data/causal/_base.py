"""Shared base class for the causal data quality defect generators.

Owns the shared plumbing: DGP configuration (no generator-side ``fit`` —
the DGP is fitted by its own ``fit``, explicitly by the caller or lazily
inside ``generate()``), input/structure validation, and mechanism gating
(which rows get the defect). Subclasses define the defect injection and
propagation, and manage their OWN labels: each generator stores the
per-row defect labels of its last generated output and exposes them
through a public getter (e.g. ``get_outliers_labels()``).
"""

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.causal_graph import validate_dag


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
        # Per-row deficit labels of the last generated output. Subclasses
        # fill this in ``generate()`` and expose it through their own
        # public getter (e.g. ``get_outliers_labels()``) which raises
        # RuntimeError when no output has been generated yet.
        self.deficit_labels_ = None

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

    def _fire_masks(self, rows, perturbation_nodes, selector=None):
        """Fire masks for ALL rows at once: dict {node: bool row-mask}.

        Delegates to a ``Selector`` from ``badgers.core.selectors``:
        ``selector(rows, y) -> {node: bool mask}``.
        ``None`` defaults to ``AlwaysSelector`` (every row fires).

        Parameters
        ----------
        rows : array to gate (the generator's working rows -- for the noise
            generator these are the n_copies-EXPANDED rows).
        perturbation_nodes : list of str, non-empty (pre-validated).
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

    def _validate_structure(self, X):
        """Validate the configured structure against the input data.

        Raises ``ValueError`` when:

        - no DGP is configured (no graph / column_mapping available);
        - the graph is not a valid DAG;
        - ``X`` has as many columns as the graph has nodes;
        - the column_mapping is a dict covering exactly the graph nodes
          with unique, in-range integer column indices.

        Returns nothing: all state needed downstream lives on ``self``
        (``self.graph``, ``self.column_mapping``).
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
        n_features = X.shape[1]
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
        seen = {}
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
            if idx in seen:
                raise ValueError(
                    f"duplicate column index {idx} "
                    f"for nodes {sorted([seen[idx], node])}"
                )
            seen[idx] = node

    def _validate_perturbation_nodes(self, perturbation_nodes):
        """Validate the perturbation nodes against the configured graph.

        Requires a non-empty list of nodes that exist in the graph.
        Raises ``ValueError`` otherwise; returns nothing.
        """
        graph = self.graph
        if graph is None:
            raise ValueError(
                "no DGP configured: pass a dgp=... to the constructor, "
                "or use use_dgp(dgp) (the DGP carries graph and "
                "column_mapping)"
            )
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
