"""Shared defaults for DGP backends (mixin base class).

Backends implement ``fit_node`` and expose ``response`` / ``draw_noise``.
``fit`` (whole-graph loop) and ``sample`` (joint sampling) are generic
algebra, provided once here — identical for every backend by construction.
"""

import networkx as nx
import numpy as np

from badgers.core.causal_graph import get_parents, topological_order


class DGPMixin:
    """Mixin providing default ``fit`` and ``sample`` implementations."""

    graph: nx.DiGraph
    column_mapping: dict

    def fit(self, X: np.ndarray) -> DGPMixin:
        """Fit every structural node's mechanism, in topological order."""
        for node in topological_order(self.graph):
            self.fit_node(node, get_parents(self.graph, node), X)
        return self


    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample ``n`` joint rows.

        Topological walk: every node contributes
        ``response(parents) + draw_noise``; for roots, ``parent_values``
        has shape ``(n, 0)`` and ``response`` returns the fitted mean.
        """
        order = topological_order(self.graph)
        X_sampled = np.zeros((n, len(self.graph.nodes)))
        for node in order:
            col = self.column_mapping[node]
            pcols = [
                self.column_mapping[p]
                for p in get_parents(self.graph, node)
            ]
            X_sampled[:, col] = (
                self.response(node, X_sampled[:, pcols])
                + self.draw_noise(node, rng, n)
            )
        return X_sampled
