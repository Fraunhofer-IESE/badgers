"""Backend-agnostic protocol for a data generation process (DGP).

A DGP is a **bundle of per-node mechanisms**: each node's mechanism is
fitted independently from (data, its parents) — the modularity assumption,
the same one DoWhy's gcm fitting states explicitly. Everything above this
protocol (interventions, cascades, joint sampling) is *generic algebra*
written once in ``badgers.core.dgp.algebra`` / ``badgers.core.dgp.base``;
backends never implement cascades and never learn about deficits.
"""

from typing import Hashable, Iterable, Protocol

import networkx as nx
import numpy as np


class DGP(Protocol):
    """Protocol for a per-node data generation process."""

    #: The structural DAG (latent variables).
    graph: nx.DiGraph
    #: Node -> integer column index of its measurement.
    column_mapping: dict

    def fit_node(self, node: Hashable, parents: Iterable[Hashable],
                 data: np.ndarray) -> None:
        """Fit ONE node's mechanism from (data, parents).

        This is the only backend-specific operation. For a root node (no
        parents) the mechanism is a distribution (fitted mean + spread).
        """
        ...

    def response(self, node: Hashable,
                 parent_values: np.ndarray) -> np.ndarray:
        """Evaluate the fitted response of ``node`` for given parent values.

        ``parent_values`` has shape ``(n_rows, len(parents))``; for a root
        node it must have shape ``(n_rows, 0)``. Returns shape ``(n_rows,)``.
        """
        ...

    def draw_noise(self, node: Hashable, rng: np.random.Generator,
                   n: int) -> np.ndarray:
        """Draw ``n`` shocks from the node's own (fitted) noise.

        Representation-free by design: parametric backends draw from their
        fitted scale, empirical backends resample their residuals.
        """
        ...

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Tier 2: sample ``n`` joint rows from the fitted graph."""
        ...
