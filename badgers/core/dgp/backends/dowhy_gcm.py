"""DoWhy-GCM backend: per-node ``AdditiveNoiseModel`` mechanisms.

Route B preparation per the structural-modifications resolution (D1–D6):
the backend implements ONLY the per-node protocol core
(``fit_node``/``response``/``draw_noise``); ``fit``/``sample``/``intervene``
and all structure edits are the generic ``DGPMixin`` reference
implementations. Native Route-B overrides (e.g. ``scale_edge`` delegating
to ``set_causal_mechanism``) are a follow-up and must then be
conformance-tested against the generic path.

Mapping to GCM concepts (dowhy 0.13):

- non-root node = ``gcm.AdditiveNoiseModel`` with GCM's linear regressor
  factory (``gcm.ml.create_linear_regressor()``) — the deterministic part
  ``f(P)`` (estimated with an intercept; badgers' zeroing-trick edge edits
  remain exact for linear forms) plus an additive exogenous shock;
- root node = an empirical distribution over the observed column;
- noise = EMPIRICAL (resampled residuals) rather than a fitted parametric
  scale — GCM's idiomatic default. GCM's own sampling paths use global
  ``np.random`` with no seed parameter, so ``draw_noise`` resamples the
  stored residuals/shocks with the caller's :class:`numpy.random.Generator`
  instead (deterministic given the rng; the library-wide reproducibility
  contract).
"""

import numpy as np

from badgers.core.dgp.base import DGPMixin


class DowhyGcmDGP(DGPMixin):
    """Per-node additive-noise DGP fitted through ``dowhy.gcm`` objects.

    - root node: empirical distribution (fitted mean ``mu`` + resampled
      centered observations as shocks);
    - non-root node: ``gcm.AdditiveNoiseModel`` with a linear
      ``prediction_model``; ``response`` is the ANM prediction, and the
      node's own noise is the EMPIRICAL residual distribution.

    Requires the optional ``dowhy`` extra (``pip install badgers[dowhy]``).
    """

    def __init__(self, graph, column_mapping):
        self.graph = graph
        self.column_mapping = column_mapping
        self._anm = {}       # node -> AdditiveNoiseModel (non-root)
        self._values = {}    # node -> observed column (root)
        self._mu = {}        # node -> fitted mean (root)
        self._residuals = {}  # node -> empirical residuals (non-root)

    def is_fitted(self):
        """True iff every graph node has a fitted mechanism (G9)."""
        return all(
            node in self._mu or node in self._anm
            for node in self.graph.nodes
        )

    def fit_node(self, node, parents, data):
        from dowhy.gcm import AdditiveNoiseModel
        from dowhy.gcm.ml import create_linear_regressor

        col = self.column_mapping[node]
        # Complete-case fit (G9): NaN-contaminated
        # rows are dropped for the ESTIMATION only; too few rows cannot
        # identify the mechanism.
        mask = np.isfinite(data[:, col])
        for p in parents:
            mask &= np.isfinite(data[:, self.column_mapping[p]])
        d = data[mask]
        if d.shape[0] < len(parents) + 1:
            raise ValueError(
                f"Cannot fit node '{node}': only {d.shape[0]} complete-case "
                f"rows available (need at least {len(parents) + 1}); the "
                f"data contains too many NaNs."
            )
        if not parents:
            values = np.asarray(d[:, col], dtype=float)
            self._values[node] = values
            self._mu[node] = float(np.mean(values))
            return
        pcols = [self.column_mapping[p] for p in parents]
        anm = AdditiveNoiseModel(
            prediction_model=create_linear_regressor()
        )
        anm.fit(X=d[:, pcols], Y=d[:, col])
        self._anm[node] = anm
        # sklearn predict returns (n, 1): ravel BEFORE the residual
        # subtraction or numpy broadcasts to (n, n).
        residuals = np.asarray(
            d[:, col] - anm.prediction_model.predict(d[:, pcols]).ravel(),
            dtype=float,
        )
        self._residuals[node] = residuals

    def response(self, node, parent_values):
        if node in self._anm:
            return np.asarray(
                self._anm[node].prediction_model.predict(parent_values),
                dtype=float,
            ).ravel()
        if node not in self._mu:
            raise RuntimeError(
                f"DowhyGcmDGP.response called before fit_node for node "
                f"'{node}'"
            )
        return np.full(parent_values.shape[0], self._mu[node])

    def draw_noise(self, node, rng, n):
        if node in self._anm:
            # empirical noise: resample fitted residuals with OUR rng
            # (GCM's own sampling uses global np.random — no seed param).
            return rng.choice(self._residuals[node], size=n)
        if node not in self._values:
            raise RuntimeError(
                f"DowhyGcmDGP.draw_noise called before fit_node for node "
                f"'{node}'"
            )
        # root: resample the centered empirical distribution so that
        # response(node) + draw_noise(node) reconstructs the marginal.
        return rng.choice(self._values[node], size=n) - self._mu[node]
