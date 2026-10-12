"""Generic deficit algebra: do-operations (interventions) on a fitted DGP.

Written once, backend-blind: any object satisfying the ``DGP`` protocol
(OLS, PyMC, DoWhyGCM, hand-written fakes) gets identical behaviour here.
Backends never implement cascades; the cascade discipline lives in this
module (topological walk, prediction differences).
"""

import numpy as np

from badgers.core.causal_graph import (
    get_descendants,
    get_parents,
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


def ordered_descendants(graph, nodes):
    """Return the descendants of ``nodes``, in topological order.

    The nodes themselves are excluded (``nx.descendants`` semantics).
    """
    descendants = set()
    for node in _as_node_list(nodes):
        descendants |= get_descendants(graph, node)
    order = topological_order(graph)
    return [node for node in order if node in descendants]


def do(dgp, rows, nodes, values=None, mode="delta", rng=None,
       rows_original=None):
    """Intervention primitive: overwrite node columns, cascade downstream.

    Steps:

    1. Snapshot ``rows_original`` (defaults to a copy of ``rows`` when
       ``mode="recompute"``).
    2. If ``values`` is given, overwrite each node's column (the
       intervention itself).
    3. Walk the descendants of ``nodes`` in topological order. For each
       child:

       - ``mode="delta"`` — preserve the child's exogenous shock exactly::

           child = child_orig
                   + f(child, parents_new) - f(child, parents_orig)

       - ``mode="recompute"`` — redraw the child's shock::

           child = f(child, parents_new) + draw_noise(child, rng, n)

    Args:
        dgp: a fitted DGP (satisfies the ``DGP`` protocol).
        rows: array of shape (n_rows, n_features); modified in place.
        nodes: one node or a list of nodes to intervene on. Their parent
            columns must already be filled.
        values: optional list of per-node column values (one array of
            shape (n_rows,) per node, same order as ``nodes``).
        mode: ``"delta"`` (exogenous shocks preserved) or ``"recompute"``
            (fresh noise drawn).
        rng: NumPy generator; required for ``mode="recompute"``.
        rows_original: pre-perturbation values of ``rows``; required for
            ``mode="delta"`` (callers mutate ``rows`` in place, so pass a
            copy taken before the perturbation).

    Returns:
        The same ``rows`` array, with the cascade applied.
    """
    nodes = _as_node_list(nodes)
    if rows_original is None:
        if mode == "delta":
            raise ValueError(
                "delta propagation requires rows_original: the "
                "pre-perturbation values of rows (callers mutate rows in "
                "place, so pass a copy taken before the perturbation)"
            )
        rows_original = rows.copy()
    if values is not None:
        for node, vals in zip(nodes, values):
            rows[:, dgp.column_mapping[node]] = vals
    for child in ordered_descendants(dgp.graph, nodes):
        ccol = dgp.column_mapping[child]
        pcols = [
            dgp.column_mapping[p] for p in get_parents(dgp.graph, child)
        ]
        new_pred = dgp.response(child, rows[:, pcols])
        if mode == "delta":
            orig_pred = dgp.response(child, rows_original[:, pcols])
            rows[:, ccol] = rows_original[:, ccol] + (new_pred - orig_pred)
        elif mode == "recompute":
            if rng is None:
                raise ValueError(
                    "recompute propagation requires rng (fresh per-row noise)"
                )
            rows[:, ccol] = new_pred + dgp.draw_noise(
                child, rng, rows.shape[0]
            )
        else:
            raise ValueError(f"Unknown propagation mode '{mode}'")
    return rows
