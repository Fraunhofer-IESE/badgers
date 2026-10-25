"""Shared base class for the causal data quality defect generators.

Owns the shared plumbing: DGP configuration (no generator-side ``fit`` —
the DGP is fitted by its own ``fit``, explicitly by the caller or lazily
inside ``generate()``) and input/structure validation. Subclasses define
the defect injection and propagation, and manage their OWN labels: each
generator stores the per-row defect labels of its last generated output
and exposes them through a public getter (e.g. ``get_outliers_labels()``).

Firing (which rows get the defect) is realized by
``badgers.core.selectors.fired``; generators import and call it
themselves. The DGP's ``column_mapping`` is the ONLY name↔position
bridge: generators speak column positions, and ``_node_names``
translates them back at the intervene boundary.
"""

import warnings

import numpy as np

from badgers.core.base import GeneratorMixin
from badgers.core.causal_graph import validate_dag


def _column_stds(X):
    """NaN-safe per-column stds (G9b): a finite scale for every column.

    NaNs are ignored (``np.nanstd``); a constant or all-NaN column gets a
    scale of 1.0 so a drawn offset stays finite -- a NaN-bearing column
    must never silently poison the whole output column (G9b).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice
        std = np.nanstd(X, axis=0)
    return np.where(np.isfinite(std) & (std > 0), std, 1.0)


def _param_by_target(param, name, targets):
    """Per-target scalar parameter: scalar broadcast or dict by column position.

    ``{0: 0.3}`` keys are COLUMN POSITIONS (the selector's targets), not
    node names. The dict must cover every target column.
    """
    if np.isscalar(param):
        return {c: float(param) for c in targets}
    if not isinstance(param, dict):
        raise ValueError(
            f"{name} must be a scalar or a dict keyed by column "
            f"position, got {type(param).__name__}"
        )
    missing = [c for c in targets if c not in param]
    if missing:
        raise ValueError(
            f"{name} dict is missing entries for target column(s) {missing} "
            f"(keys are column positions; selector targets = {list(targets)})"
        )
    return {c: float(param[c]) for c in targets}


class CausalGenerator(GeneratorMixin):
    """Base class for causal defect generators.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
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

            missingness = CausalMissingnessGenerator(random_generator=..., dgp=dgp)
            Xm = missingness.generate(X, ...)

            drift = CausalDriftGenerator(random_generator=..., dgp=dgp)  # share
            Xt = drift.generate(Xm, ...)

        All per-stage scale computations are NaN-safe, so even the lazy
        refit path behaves sensibly; the explicitly-fitted shared
        ``dgp=`` pattern simply guarantees ONE world for the whole
        pipeline. Fit the DGP once on clean data — the backends do not
        impute or filter NaN rows at fit time.
    """

    #: Subclasses set this to the deficit name used in labels.
    deficit_name = "causal"

    def __init__(self, random_generator, dgp=None):
        self.random_generator = random_generator
        self.dgp = dgp
        self.graph = None if dgp is None else dgp.graph
        self.column_mapping = None if dgp is None else dgp.column_mapping
        # Per-row deficit labels of the last generated output. Subclasses
        # fill this in ``generate()`` and expose it through their own
        # public getter (e.g. ``get_outliers_labels()``) which raises
        # RuntimeError when no output has been generated yet.
        self.deficit_labels_ = None

    def _get_deficit_labels(self) -> np.ndarray:
        """Shared body of the per-family ``get_<deficit>_labels()`` getters.

        Returns ``self.deficit_labels_`` (the boolean per-row field filled
        by ``generate()``); raises ``RuntimeError`` when no output has
        been generated yet. The error message names the family through
        ``self.deficit_name``.
        """
        if self.deficit_labels_ is None:
            raise RuntimeError(
                f"{self.deficit_name.capitalize()} labels are only "
                "available after calling generate()."
            )
        return self.deficit_labels_

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

        Only the legacy ``quantile_range`` fallback of
        ``CausalImbalanceGenerator`` still takes node names.
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

    def _node_names(self, columns):
        """Node names for column positions (inverse of ``column_mapping``).

        Total after :meth:`_validate_structure`: the mapping is a bijection
        between graph nodes and in-range column indices, so every valid
        position has exactly one name.
        """
        names = {col: node for node, col in self.column_mapping.items()}
        return [names[c] for c in columns]
