"""Shared defaults for DGP backends (mixin base class).

Backends implement ``fit_node`` and expose ``response`` / ``draw_noise``
(plus a ``structural`` override when the composition is not additive).
Everything else is generic algebra, provided once here — identical for
every backend by construction:

- ``fit`` (whole-graph topological walk over ``fit_node``),
- ``sample`` (joint sampling),
- ``intervene`` (Pearl surgery on given rows).

Every cascade (``sample``/``intervene``) starts by asking the backend for
its parameter set via :meth:`pick_parameters` — a hook with the default
"THIS DGP" (deterministic backends have exactly one fitted parameter
set; uncertainty-carrying backends like ``PyMCSCMDGP`` override it to pin
one coherent draw from the caller's rng).

The DGP is immutable and rngless: all randomness enters through the
``rng`` argument the CALLER provides. Structure edits are NOT methods on
the DGP: modifications live in ``badgers.core.dgp.edits`` as free
functions returning a new DGP (exact view edits only).

Backends may override any of the evaluation verbs with a native
implementation; overrides must preserve the protocol semantics
(intervention semantics, column shapes) — conformance is tested per
backend against these generic reference implementations.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from badgers.core.causal_graph import (
    get_parents,
    ordered_descendants,
    topological_order,
)


def _as_node_list(nodes):
    """Normalize a node or iterable of nodes into a list."""
    if isinstance(nodes, (str, bytes)):
        return [nodes]
    try:
        return list(nodes)
    except TypeError:
        return [nodes]


class DGPMixin:
    """Mixin providing default ``fit``, ``sample``, ``intervene`` and the
    structure edits."""

    graph: nx.DiGraph
    column_mapping: dict

    def fit(self, X: np.ndarray) -> DGPMixin:
        """Fit every structural node's mechanism, in topological order."""
        for node in topological_order(self.graph):
            self.fit_node(node, get_parents(self.graph, node), X)
        return self


    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample ``n`` joint rows.

        Topological walk: every node contributes its :meth:`structural`
        values (default: the additive composition ``response(parents) +
        draw_noise``); for roots, ``parent_values`` has shape ``(n, 0)``.

        ONE parameter set drives the whole cascade (one call = one
        coherent model); see :meth:`pick_parameters`.
        """
        carrier = self.pick_parameters(rng)
        order = topological_order(carrier.graph)
        X_sampled = np.zeros((n, len(carrier.graph.nodes)))
        for node in order:
            col = carrier.column_mapping[node]
            pcols = [
                carrier.column_mapping[p]
                for p in get_parents(carrier.graph, node)
            ]
            X_sampled[:, col] = carrier.structural(
                node, X_sampled[:, pcols], rng, n
            )
        return X_sampled

    def structural(self, node, parent_values, rng, n):
        """One node's values for a cascade: compose the deterministic
        part and the noise.

        Default: the additive composition ``response(parents) +
        draw_noise``. Backends whose mechanisms are not additive (e.g.
        dowhy functional causal models, where ``evaluate`` owns the
        composition — rounded predictions, class CDFs, post-nonlinear
        links) override this and delegate the composition to the
        library's own ``evaluate`` while still drawing the noise with
        the caller's ``rng``.
        """
        return self.response(node, parent_values) + self.draw_noise(node, rng, n)

    # -- Value-level intervention --------------------------------------------

    def intervene(self, rows, nodes, values=None, *, rng=None):
        """Pearl surgery applied to given rows: fix the nodes at ``values``,
        recompute ALL their descendants with fresh noise.

        Semantics:

        1. If ``values`` is given, overwrite each node's column (the
           intervention itself; ``None`` keeps the current column values —
           callers that pre-write their perturbation use this).
        2. Walk the descendants of ``nodes`` in topological order (the
           intervened nodes themselves are excluded — pinned values stay
           pinned even if one intervened node is a descendant of another)
           and recompute each via :meth:`structural` (default
           ``response(parents_new) + fresh noise``).

        Upstream and non-descendant columns and rows are untouched.
        Pre-existing NaNs in a recomputed descendant column are
        re-evaluated (a NaN parent propagates to the child; a NaN child
        with finite parents is recomputed to a finite value).

        Args:
            rows: array of shape (n_rows, n_features); modified in place.
            nodes: one node or a list of nodes to intervene on.
            values: optional list of per-node column values (one array of
                shape (n_rows,) per node, same order as ``nodes``).
            rng: NumPy generator (fresh descendant noise); required.

        Returns:
            The same ``rows`` array, with the intervention applied.
        """
        nodes = _as_node_list(nodes)
        if rng is None:
            raise ValueError(
                "intervene requires rng (descendants are recomputed with "
                "fresh noise)"
            )
        # ONE parameter set drives the whole cascade (one intervention =
        # one coherent model); see pick_parameters.
        carrier = self.pick_parameters(rng)
        if values is not None:
            for node, vals in zip(nodes, values, strict=True):
                rows[:, carrier.column_mapping[node]] = vals
        pinned = set(nodes)
        n = rows.shape[0]
        for child in ordered_descendants(carrier.graph, nodes):
            if child in pinned:
                continue
            ccol = carrier.column_mapping[child]
            pcols = [
                carrier.column_mapping[p]
                for p in get_parents(carrier.graph, child)
            ]
            rows[:, ccol] = carrier.structural(child, rows[:, pcols], rng, n)
        return rows

    def pick_parameters(self, rng=None):
        """Hook called ONCE at the top of ``sample``/``intervene``: return
        the DGP instance whose mechanisms should drive this cascade.

        Generic default: ``self`` — a deterministic backend fits exactly
        ONE parameter set, so there is nothing to choose and no rng is
        consumed. Backends carrying parameter uncertainty override it:
        they use ``rng`` to select ONE coherent parameter draw and return
        a read-only view of themselves evaluating it. Separation of
        concerns: the DGP keeps every draw from ``fit``; the choice of
        which one drives a batch is made at the top of each cascade, from
        the caller's rng — never inside ``response`` (which takes no rng)
        and never by mutating the fitted DGP.
        """
        return self

