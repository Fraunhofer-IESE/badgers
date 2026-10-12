"""Default backend: deterministic per-node least squares (pure NumPy).

Mechanisms are **zero-intercept linear**: ``child = beta @ parents``. The
zero intercept is a precondition of the scaled-edge algebra (``f' = k*f``
on a single edge's contribution); intercepts would need the wider
edge-scaling treatment (future work).
"""

import numpy as np

from badgers.core.dgp.base import DGPMixin


class OlsDGP(DGPMixin):
    """Per-node linear least-squares DGP (deterministic, dependency-free).

    - root node: fitted mean ``mu`` and spread ``sigma``;
    - non-root node: zero-intercept least squares ``beta`` (one weight per
      parent, in structural predecessor order) and residual std.
    """

    def __init__(self, graph, column_mapping):
        self.graph = graph
        self.column_mapping = column_mapping
        self._beta = {}
        self._residual_std = {}
        self._mu = {}
        self._sigma = {}

    def fit_node(self, node, parents, data):
        col = self.column_mapping[node]
        if not parents:
            self._mu[node] = float(np.mean(data[:, col]))
            self._sigma[node] = float(np.std(data[:, col]))
            return
        pcols = [self.column_mapping[p] for p in parents]
        X_parents = data[:, pcols]
        beta, _, _, _ = np.linalg.lstsq(
            X_parents, data[:, col], rcond=None
        )
        self._beta[node] = beta
        self._residual_std[node] = float(
            np.std(data[:, col] - X_parents @ beta)
        )

    def _is_root(self, node):
        return node not in self._beta

    def response(self, node, parent_values):
        if self._is_root(node):
            if node not in self._mu:
                raise RuntimeError(
                    "OlsDGP.response called before fit_node for node "
                    f"'{node}'"
                )
            return np.full(parent_values.shape[0], self._mu[node])
        return parent_values @ self._beta[node]

    def draw_noise(self, node, rng, n):
        if self._is_root(node):
            if node not in self._mu:
                raise RuntimeError(
                    "OlsDGP.draw_noise called before fit_node for node "
                    f"'{node}'"
                )
            return rng.normal(0, self._sigma[node], size=n)
        return rng.normal(0, self._residual_std[node], size=n)
