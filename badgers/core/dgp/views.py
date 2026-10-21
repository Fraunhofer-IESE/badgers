"""Read-only DGP views backing the edit free functions.

Each view wraps a fitted base DGP and overrides the minimal per-node
surface (``response`` / ``draw_noise`` / ``graph``); ``fit``/``sample``/
``intervene`` are inherited from ``DGPMixin`` and correctly go THROUGH
the view's overridden mechanisms — so views compose
(``set_form(scale_edge(base, ...), ...)`` etc.) and sampling an edited
DGP honors the edit. Views never mutate the base DGP.

Parameter-uncertainty hook: all views override ``pick_parameters`` to
first let the base pick (default: the base itself → the view works as-is
for deterministic backends); when the base returns a PINNED draw (e.g.
``PyMCSCMDGP``'s posterior), the view returns a shallow copy of itself with
the pinned base rebound, so the edit's overridden mechanisms evaluate on
that parameter draw.
"""

import copy

import numpy as np

from badgers.core.dgp.base import DGPMixin


class _ScaledEdgeView(DGPMixin):
    """Read-only view of a base DGP with ONE edge's contribution scaled.

    For a child with parents ``P`` and a perturbed edge ``(p, child)``
    scaled by ``k`` (zero-intercept additive mechanisms), the modified
    response is derived without exposing coefficients::

        contribution(p) = f(P) - f(P with p zeroed)
        f'(P)           = f(P) + (k - 1) * contribution(p)

    The child's noise is rescaled to preserve the child's marginal
    variance (evaluated on the calibration data via :meth:`calibrate`)::

        scaled_pred   = f'(P)                      (on the calibration data)
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
        """Evaluate the variance budget on the calibration data; set the
        child's noise-rescale ratio. Raises ``ValueError`` if infeasible."""
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

    def is_fitted(self):
        return self._base.is_fitted()

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

    def pick_parameters(self, rng=None):
        """Let the base pick; when it returns a PINNED draw (posterior
        backends), rebind it onto a shallow copy of this view so the
        scaled mechanisms evaluate on that draw."""
        pinned = self._base.pick_parameters(rng)
        if pinned is self._base:
            return self
        clone = copy.copy(self)
        clone._base = pinned
        return clone


class _ExogenizedView(DGPMixin):
    """Read-only view of a base DGP with a node exogenized (made a root).

    Pearl surgery as an edit: the node's incoming edges are REMOVED from
    the graph (a modified copy; the base DGP is untouched), and the node's
    response becomes zero (it contributes nothing back). At generate time
    the node's input column IS its value — the intervention cascade never
    recomputes a root, so the column survives untouched.

    INTERVENTION-ONLY semantics (decided 2026-10-15, YAGNI): this edit is
    designed for the pin-then-intervene flow. ``sample()`` on an
    exogenized DGP draws the node as ``0 + N(0, residual_std)`` — the
    base mechanism's residual scale around a ZERO mean — which is NOT the
    node's real marginal (wrong center, typically too narrow, Gaussian
    shape). The honest fix would be an empirical-resampling mechanism,
    but that requires training-data retention (violating A3 option (b)) —
    deferred until a re-randomization-baseline use case actually exists;
    a generator that needs it can bootstrap the column itself.
    """

    def __init__(self, base, node):
        self._base = base
        self._node = node
        if node not in base.graph.nodes:
            raise ValueError(f"node '{node}' not in graph")
        graph = base.graph.copy()
        graph.remove_edges_from(list(base.graph.in_edges(node)))
        self.graph = graph
        self.column_mapping = base.column_mapping

    def is_fitted(self):
        return self._base.is_fitted()

    def fit_node(self, node, parents, data):
        self._base.fit_node(node, parents, data)

    def response(self, node, parent_values):
        if node == self._node:
            return np.zeros(parent_values.shape[0])
        return self._base.response(node, parent_values)

    def draw_noise(self, node, rng, n):
        return self._base.draw_noise(node, rng, n)

    def pick_parameters(self, rng=None):
        pinned = self._base.pick_parameters(rng)
        if pinned is self._base:
            return self
        clone = copy.copy(self)
        clone._base = pinned
        return clone


class _FormView(DGPMixin):
    """Read-only view of a base DGP with ONE node's functional form replaced.

    The node's response becomes a user-supplied function of the parent
    values (F ← F′; the user-authored route — no refit, A5). The node's
    noise law is INHERITED from the base mechanism by default (the base
    residual scale is calibrated for F, not F′); a user-supplied
    ``draw_noise(rng, n)`` closure (Q2 option b, decided 2026-10-15)
    replaces it.

    Everything else is delegated to the base DGP.
    """

    def __init__(self, base, node, response_fn, draw_noise=None):
        self._base = base
        self._node = node
        if node not in base.graph.nodes:
            raise ValueError(f"node '{node}' not in graph")
        if not callable(response_fn):
            raise ValueError("response_fn must be callable")
        if draw_noise is not None and not callable(draw_noise):
            raise ValueError("draw_noise must be callable or None")
        self.graph = base.graph
        self.column_mapping = base.column_mapping
        self._response_fn = response_fn
        self._draw_noise_fn = draw_noise

    def is_fitted(self):
        return self._base.is_fitted()

    def fit_node(self, node, parents, data):
        self._base.fit_node(node, parents, data)

    def response(self, node, parent_values):
        if node == self._node:
            out = self._response_fn(parent_values)
            return np.broadcast_to(
                np.asarray(out, dtype=float), (parent_values.shape[0],)
            )
        return self._base.response(node, parent_values)

    def draw_noise(self, node, rng, n):
        if node == self._node and self._draw_noise_fn is not None:
            out = np.asarray(self._draw_noise_fn(rng, n), dtype=float)
            return np.broadcast_to(out, (n,))
        return self._base.draw_noise(node, rng, n)

    def pick_parameters(self, rng=None):
        pinned = self._base.pick_parameters(rng)
        if pinned is self._base:
            return self
        clone = copy.copy(self)
        clone._base = pinned
        return clone


class _AddedEdgeContributionView(DGPMixin):
    """View of a base DGP with a NEW parent's additive term added to a
    child's response (ANM-scoped cheap edit; see
    :func:`badgers.core.dgp.edits.add_edge_contribution`).

    The child's response becomes ``base_response(child, parents) +
    contribution(parent_col)`` where the parent column is read through
    THIS view (so edits compose: the added term sees the scaled/formed
    DGP). The child's noise law is inherited unless a user closure is
    supplied.
    """

    def __init__(self, base, child, parent, contribution, draw_noise=None):
        self._base = base
        self._child = child
        self._parent = parent
        if child not in base.graph.nodes or parent not in base.graph.nodes:
            raise ValueError("both child and parent must be graph nodes")
        graph = base.graph.copy()
        graph.add_edge(parent, child)
        self.graph = graph
        self.column_mapping = base.column_mapping
        self._contribution = contribution
        self._draw_noise_fn = draw_noise
        # OLD parent set (base graph order): base.response is evaluated on
        # exactly these columns, extracted from the incoming (VIEW graph
        # order) parent_values by name.
        self._base_parents = list(base.graph.predecessors(child))

    def is_fitted(self):
        return self._base.is_fitted()

    def fit_node(self, node, parents, data):
        self._base.fit_node(node, parents, data)

    def response(self, node, parent_values):
        if node != self._child:
            return self._base.response(node, parent_values)
        parents = list(self.graph.predecessors(node))
        pidx = parents.index(self._parent)
        base_cols = [parents.index(p) for p in self._base_parents]
        base_pred = self._base.response(node, parent_values[:, base_cols])
        return base_pred + self._contribution(parent_values[:, pidx])

    def draw_noise(self, node, rng, n):
        if node == self._child and self._draw_noise_fn is not None:
            out = np.asarray(self._draw_noise_fn(rng, n), dtype=float)
            return np.broadcast_to(out, (n,))
        return self._base.draw_noise(node, rng, n)

    def pick_parameters(self, rng=None):
        return self._base.pick_parameters(rng)
