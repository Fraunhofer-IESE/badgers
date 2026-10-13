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

    1. If ``values`` is given, overwrite each node's column (the
       intervention itself).
    2. Walk the descendants of ``nodes`` in topological order. For each
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
    if mode not in ("delta", "recompute"):
        raise ValueError(f"Unknown propagation mode '{mode}'")
    if rows_original is None:
        if mode == "delta":
            raise ValueError(
                "delta propagation requires rows_original: the "
                "pre-perturbation values of rows (callers mutate rows in "
                "place, so pass a copy taken before the perturbation)"
            )
    if values is not None:
        for node, vals in zip(nodes, values, strict=True):
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
    return rows


class ScaledEdgeDGP:
    """Read-only view of a base DGP with ONE edge's contribution scaled.

    For a child with parents ``P`` and a perturbed edge ``(p, child)``
    scaled by ``k`` (zero-intercept additive mechanisms), the modified
    response is derived without exposing coefficients::

        contribution(p) = f(P) - f(P with p zeroed)
        f'(P)           = f(P) + (k - 1) * contribution(p)

    The child's noise is rescaled to preserve the child's marginal
    variance (evaluated on the fit data via :meth:`calibrate`)::

        scaled_pred   = f'(P)                      (on the fit data)
        new_resid_var = var(child) - var(scaled_pred)
        (ValueError if new_resid_var < 0 -- infeasible)

        noise_ratio   = sqrt(new_resid_var) / std(child - f(child, P))

    For a single-parent child ``f'(P) = k * contribution(p)`` and the
    budget reduces to ``var(child) - k^2 * var(contribution)``; with
    several parents, subtracting the full scaled response is what keeps
    the marginal variance exact (the untouched parents' contribution is
    not double-counted).

    Everything else (graph, column_mapping, fit_node, sample) is
    delegated to the base DGP.
    """

    def __init__(self, base, child, parent, k):
        self._base = base
        self._child = child
        self._parent = parent
        self._k = float(k)
        child_parents = list(base.graph.predecessors(child))
        if parent not in child_parents:
            raise ValueError(f"edge ({parent}, {child}) not in graph")
        self._pidx = child_parents.index(parent)
        self._noise_ratio = None

    # -- variance budget ----------------------------------------------------

    def calibrate(self, data):
        """Evaluate the variance budget on the fit data; set the child's
        noise-rescale ratio. Raises ``ValueError`` if infeasible."""
        col = self._base.column_mapping[self._child]
        pcols = [
            self._base.column_mapping[p]
            for p in self._base.graph.predecessors(self._child)
        ]
        y = data[:, col]
        base_pred = self._base.response(self._child, data[:, pcols])
        zeroed = data[:, pcols].copy()
        zeroed[:, self._pidx] = 0.0
        contribution = base_pred - self._base.response(self._child, zeroed)
        scaled_pred = base_pred + (self._k - 1.0) * contribution
        new_resid_var = float(np.var(y)) - float(np.var(scaled_pred))
        if new_resid_var < 0:
            raise ValueError(
                "Edge perturbation is infeasible: var(f'(P)) exceeds "
                "var(child). Reduce k."
            )
        base_resid_std = float(np.std(y - base_pred))
        self._noise_ratio = (
            float(np.sqrt(new_resid_var) / base_resid_std)
            if base_resid_std > 0
            else 0.0
        )
        return self

    # -- protocol views -----------------------------------------------------

    @property
    def graph(self):
        return self._base.graph

    @property
    def column_mapping(self):
        return self._base.column_mapping

    def fit_node(self, node, parents, data):
        self._base.fit_node(node, parents, data)

    def fit(self, X):
        """Fit the base DGP, then calibrate the variance budget on ``X``."""
        self._base.fit(X)
        return self.calibrate(X)

    def response(self, node, parent_values):
        base_pred = self._base.response(node, parent_values)
        if node != self._child:
            return base_pred
        zeroed = parent_values.copy()
        zeroed[:, self._pidx] = 0.0
        contribution = base_pred - self._base.response(node, zeroed)
        return base_pred + (self._k - 1.0) * contribution

    def draw_noise(self, node, rng, n):
        shocks = self._base.draw_noise(node, rng, n)
        if node == self._child and self._noise_ratio is not None:
            return shocks * self._noise_ratio
        return shocks

    def sample(self, rng, n):
        return self._base.sample(rng, n)
