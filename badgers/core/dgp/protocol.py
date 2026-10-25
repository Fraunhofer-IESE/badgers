"""Backend-agnostic protocol for a data generation process (DGP).

A DGP is a **bundle of per-node mechanisms**: each node's mechanism is
fitted independently from (data, its parents) — the modularity assumption,
the same one DoWhy's gcm fitting states explicitly. Everything above this
protocol (interventions, cascades, joint sampling) is *generic algebra*
written once in ``badgers.core.dgp.base``;
backends never implement cascades and never learn about deficits.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Protocol

import networkx as nx
import numpy as np



class DGP(Protocol):
    """Protocol for a per-node data generation process."""

    #: The structural DAG (latent variables).
    graph: nx.DiGraph
    #: Node -> integer column index of its measurement.
    column_mapping: dict

    def fit(self, X: np.ndarray) -> DGP:
        """Fit every node's mechanism in topological order; return ``self``.

        ``DGPMixin`` provides the default topological walk over
        ``fit_node``; backends that do not inherit it must implement
        ``fit`` themselves. Generators call ``fit`` on any backend passed
        via ``fit(dgp=...)``.
        """
        ...

    def is_fitted(self) -> bool:
        """True iff every structural node has a fitted mechanism.

        Generator-side convenience: lets a generator fit an UNFITTED DGP
        lazily on the generate-time data while leaving a fitted one
        untouched. Backend-specific storage means backends implement
        this; read-only views delegate to the DGP they wrap.
        """
        ...

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

    def structural(self, node: Hashable, parent_values: np.ndarray,
                   rng: np.random.Generator, n: int) -> np.ndarray:
        """Compose ONE node's values for a cascade: deterministic part +
        noise.

        Default (``DGPMixin``): the additive composition
        ``response(parent_values) + draw_noise(node, rng, n)``. Backends
        whose mechanisms are not additive override it — e.g. the gcm
        backend delegates the composition to the mechanism's own
        ``evaluate(parents, noise)`` (rounded discrete predictions, class
        CDFs) while still drawing the noise with ``rng``.
        """
        ...

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Tier 2: sample ``n`` joint rows from the fitted graph."""
        ...

    def pick_parameters(self, rng=None) -> "DGP":
        """Hook called ONCE at the top of ``sample``/``intervene``: return
        the DGP instance whose mechanisms drive this cascade.

        Generic default (``DGPMixin``): ``self`` — deterministic backends
        fit exactly one parameter set. Uncertainty-carrying backends
        override it: they select ONE coherent parameter draw from ``rng``
        and return a read-only view of themselves evaluating it. The DGP
        keeps every draw from ``fit``; the choice is made at the top of
        each cascade from the caller's rng — never inside ``response``
        and never by mutating the fitted DGP.
        """
        ...

    # -- Value-level intervention -------------------------------------------

    def intervene(self, rows: np.ndarray, nodes: Hashable | Iterable,
                  values=None, *, rng=None) -> np.ndarray:
        """Pearl surgery applied to GIVEN rows: fix the nodes at ``values``,
        recompute all their descendants (topological order, fresh noise).

        ``DGPMixin`` provides the generic implementation; backends may
        override it natively, preserving the same semantics (upstream and
        non-descendant columns and rows untouched; intervened nodes keep
        their fixed values even if they are descendants of other
        intervened nodes).

        NOTE: structure edits are intentionally NOT part of this protocol.
        The DGP is an immutable, fitted world; structural modifications
        live in ``badgers.core.dgp.edits`` (free functions returning a new
        world: ``scale_edge``/``remove_edge``/``add_edge_contribution``/
        ``exogenize``/``set_form``).
        """
        ...
