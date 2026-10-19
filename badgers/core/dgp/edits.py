"""Structure edits as FREE FUNCTIONS on a fitted DGP.

The DGP is an immutable, fully-fitted SCM (graph + fitted per-node
mechanisms). Modifying the model is not something the DGP does to itself
— every edit is a transformation ``fn(dgp, ...) -> new DGP`` defined
here, generically, against the ``DGP`` protocol. The DGP itself exposes
only evaluation verbs (``fit``/``response``/``draw_noise``/
``sample``/``intervene``).

Edit semantics
--------------

- **Exact edits** (no estimation — the views here):
    - :func:`scale_edge` — rescale one edge's contribution by ``k``
      (variance-preserving; ``k=0`` is :func:`remove_edge`);
    - :func:`remove_edge` — zero one edge's contribution (SCM surgery);
    - :func:`exogenize` — make a node a root (drop incoming edges);
    - :func:`set_form` — replace a node's response with a user closure
      (optionally its noise law too);
    - :func:`add_edge_contribution` — add a NEW parent's additive term to
      a child (ANM-scoped; see below).

``add_edge_contribution`` is the only ANM-scoped edit: it augments an
additive-noise child's response with a user-authored term for the new
parent. For non-additive (future) backends, the general tool is
:func:`set_form` (rewrite the child's full response).

All edits compose (the returned object is itself a ``DGP``):
``set_form(scale_edge(dgp, ...), ...)`` etc. Views never mutate the
base DGP.

Re-estimation after a structural change is out of scope here: build
the modified SCM with the library's own API (dowhy mechanism
assignment or a new PyMC model) and wrap it again with ``GcmSCMDGP``
or ``PyMCSCMDGP``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from badgers.core.dgp.views import (
    _AddedEdgeContributionView,
    _ExogenizedView,
    _FormView,
    _ScaledEdgeView,
)

__all__ = [
    "scale_edge",
    "remove_edge",
    "add_edge_contribution",
    "exogenize",
    "set_form",
]


def scale_edge(dgp, child, parent, k):
    """Scale one edge's contribution by ``k`` (variance-preserving view).

    Returns the ``_ScaledEdgeView``; call ``calibrate(data)`` on it to
    evaluate the variance budget (raises ``ValueError`` if infeasible).
    """
    return _ScaledEdgeView(dgp, child, parent, k)


def remove_edge(dgp, child, parent):
    """Remove an edge s->t: zero the parent's contribution (SCM surgery
    view — the removed parent's influence is absorbed into the child's
    exogenous shock; no refit). Q3 option (a)."""
    return _ScaledEdgeView(dgp, child, parent, 0.0)


def add_edge_contribution(dgp, child, parent, contribution=None,
                          coefficient=None, draw_noise=None):
    """Add a NEW parent's additive term to a child's response (A5 cheap
    edit; ANM-scoped).

    The child's response becomes ``base_response(child, parents) +
    contribution(parent_col)``; the graph gains the edge ``(parent,
    child)``. Exactly one of ``contribution`` (a closure of the parent
    column) or ``coefficient`` (sugar for ``lambda x: coefficient * x``)
    must be given. The child's noise law is inherited from the base
    mechanism unless a ``draw_noise(rng, n)`` closure is supplied (Q2
    option (b) pattern).

    Raises ``ValueError`` if the edge would create a cycle (A6) or if the
    parent is already a parent of ``child``.
    """
    if contribution is None and coefficient is None:
        raise ValueError(
            "add_edge_contribution requires a contribution closure or a "
            "coefficient (an ESTIMATED new parent is not supported by "
            "this view edit; for re-estimation after a structural change, "
            "build the modified SCM with the library's own API and wrap "
            "it again with GcmSCMDGP or PyMCSCMDGP)."
        )
    if contribution is not None and coefficient is not None:
        raise ValueError(
            "pass either contribution or coefficient, not both"
        )
    if contribution is not None and not callable(contribution):
        raise ValueError("contribution must be callable")
    if (child, parent) in dgp.graph.edges:
        raise ValueError(
            f"edge ({parent}, {child}) already exists; use set_form or "
            "scale_edge to modify it"
        )
    if nx.has_path(dgp.graph, child, parent):
        raise ValueError(
            f"adding edge ({parent}, {child}) would create a cycle: there "
            "is a directed path from the child back to the parent"
        )
    if coefficient is not None:
        contribution = (lambda x: coefficient * x)
    return _AddedEdgeContributionView(
        dgp, child, parent, contribution, draw_noise
    )


def exogenize(dgp, node):
    """Make a node a root: remove its incoming edges (graph surgery as an
    edit; no refit). Returns an ``_ExogenizedView``.

    INTERVENTION-ONLY semantics: designed for the pin-then-intervene flow
    (the node's column is written by the generator; the cascade never
    recomputes a root). ``sample()`` on the returned view draws the node
    as ``0 + N(0, residual_std)`` — NOT the node's real marginal; see the
    ``_ExogenizedView`` docstring for the deferred empirical-resampling
    route.
    """
    return _ExogenizedView(dgp, node)


def set_form(dgp, node, response_fn, draw_noise=None):
    """Replace one node's functional form with a user-supplied closure
    (F ← F′; no refit). Noise law INHERITED from the base mechanism by
    default; pass a ``draw_noise(rng, n)`` closure to supply your own
    noise law (Q2 option (b)). Returns a ``_FormView``."""
    return _FormView(dgp, node, response_fn, draw_noise)
