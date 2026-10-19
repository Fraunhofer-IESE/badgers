"""Adapter for externally built ``dowhy.gcm.StructuralCausalModel`` objects.

Users familiar with dowhy-gcm build their own causal model — custom
``PredictionModel`` subclasses, ``EmpiricalDistribution`` roots, any
mechanism the library offers — and hand the finished object to badgers.
Badgers does NOT replace dowhy-gcm: this adapter translates the badgers
DGP protocol onto the mechanisms the user configured, so ``sample`` and
Pearl-surgery ``intervene`` run on the generic ``DGPMixin`` path.

Mapping (dowhy 0.13):

- ``fit`` delegates mechanism fitting wholesale to ``gcm.fit`` on a
  DataFrame in ``column_mapping`` order — whatever the user configured
  per node is fitted by gcm exactly as gcm would fit it;
- non-root ``AdditiveNoiseModel`` nodes: ``response`` is
  ``prediction_model.predict(parent_values)``; noise is the EMPIRICAL
  residual distribution (``mechanism.estimate_noise`` residuals cached at
  fit time and resampled with the caller's rng — gcm's own draw paths use
  global ``np.random`` with no seed, which would break the library-wide
  reproducibility contract);
- root ``EmpiricalDistribution`` nodes: ``response`` returns zeros and
  ``draw_noise`` resamples the stored observations ``mechanism.data``
  (response + noise reconstructs the marginal);
- any other mechanism type raises a clear ``NotImplementedError`` at fit
  time naming the node and mechanism class.
"""

import numpy as np
import pandas as pd

from badgers.core.causal_graph import get_parents
from badgers.core.dgp.base import DGPMixin


class GcmSCMDGP(DGPMixin):
    """Plug an externally built ``dowhy.gcm`` causal model into badgers.

    Args:
        causal_model: a user-built ``dowhy.gcm.StructuralCausalModel``
            (or any object with ``.graph`` and per-node
            ``'causal_mechanism'`` attributes on the graph).
        column_mapping: optional dict node name -> column index of the
            numpy data; defaults to graph node insertion order. When
            supplied explicitly it must cover every graph node exactly
            once (keys and column indices both form complete sets).

    Raises:
        ValueError: if an explicit ``column_mapping`` does not cover the
            graph nodes exactly (missing/extra node names, or missing/
            duplicate column indices).

    Requires the optional ``dowhy`` extra (``pip install badgers[dowhy]``).

    Classifier nodes couple ``response`` to ``draw_noise``: ``response``
    captures the pending parent values that ``draw_noise`` then consumes
    (it receives no parents itself). Discrete classes are numeric-encoded
    (``get_class_names`` coerced to float).
    """

    def __init__(self, causal_model, column_mapping=None):
        self._cm = causal_model
        self.graph = causal_model.graph
        if column_mapping is None:
            column_mapping = {node: i for i, node in enumerate(self.graph.nodes)}
        else:
            nodes = set(self.graph.nodes)
            keys = set(column_mapping)
            if keys != nodes:
                raise ValueError(
                    f"column_mapping must cover every graph node exactly "
                    f"once: missing nodes {sorted(nodes - keys)}, unknown "
                    f"nodes {sorted(keys - nodes)}."
                )
            if len(column_mapping) != len(nodes) or len(set(column_mapping.values())) != len(nodes):
                raise ValueError(
                    "column_mapping values must form a complete, "
                    "duplicate-free column index set matching the number "
                    "of graph nodes."
                )
        self.column_mapping = column_mapping
        self._mech = {}
        self._residuals = {}  # node -> empirical residuals (non-root ANM)
        self._data = {}       # node -> observed values (empirical root)
        self._clf = {}        # node -> ClassifierFCM mechanism
        self._pending = {}    # node -> parent_values captured by last response call

    def is_fitted(self):
        """True iff every graph node has a translated mechanism."""
        return all(node in self._mech for node in self.graph.nodes)

    def fit_node(self, node, parents, data):
        """Translate one node's mechanism (the bulk of the work happens in
        :meth:`fit`, which delegates to ``gcm.fit``; ``fit_node`` validates
        and caches the per-node translation)."""
        from dowhy.gcm import AdditiveNoiseModel, ClassifierFCM, EmpiricalDistribution

        mech = self.graph.nodes[node].get("causal_mechanism")
        if mech is None:
            raise ValueError(
                f"Node '{node}' has no causal mechanism: set one with "
                f"causal_model.set_causal_mechanism(...) before fitting."
            )
        if isinstance(mech, AdditiveNoiseModel):
            if not parents:
                raise ValueError(
                    f"Node '{node}' is a root but its mechanism is an "
                    f"AdditiveNoiseModel; use EmpiricalDistribution (or a "
                    f"ScipyDistribution) for root nodes."
                )
            self._mech[node] = mech
            return
        if isinstance(mech, EmpiricalDistribution):
            if parents:
                raise ValueError(
                    f"Node '{node}' has parents {list(parents)} but its "
                    f"mechanism is an EmpiricalDistribution, which is only "
                    f"valid for root nodes."
                )
            self._mech[node] = mech
            return
        if isinstance(mech, ClassifierFCM):
            if not parents:
                raise ValueError(
                    f"Node '{node}' is a root but its mechanism is a "
                    f"ClassifierFCM; use EmpiricalDistribution for roots."
                )
            self._mech[node] = mech
            self._clf[node] = mech
            return
        raise NotImplementedError(
            f"GcmSCMDGP does not support mechanism "
            f"{type(mech).__name__} for node '{node}' yet. Supported: "
            f"AdditiveNoiseModel (non-root), ClassifierFCM (non-root, "
            f"discrete), EmpiricalDistribution (root)."
        )

    def fit(self, X):
        """Validate and translate all mechanisms BEFORE delegating to
        ``gcm.fit`` on a DataFrame, then cache the empirical noise sources
        (residuals / root observations).

        A pandas DataFrame is accepted directly and its dtypes are
        preserved (gcm classifiers need string categorical targets while
        the remaining columns stay numeric); numpy input is converted to a
        DataFrame in ``column_mapping`` order.

        Pre-validating every node first avoids wasted compute and
        confusing dowhy-gcm errors when a mechanism is missing or of an
        unsupported type.

        NaN handling is delegated to gcm's per-mechanism estimators
        (unlike ``DowhyGcmDGP``'s complete-case policy).
        """
        import dowhy.gcm as gcm

        data = np.asarray(X)
        # validate all nodes up front (this also caches the mechanism
        # translation into self._mech)
        for node in self.graph.nodes:
            self.fit_node(node, get_parents(self.graph, node), data)

        columns = [None] * len(self.graph.nodes)
        for node, col in self.column_mapping.items():
            columns[col] = node
        if isinstance(X, pd.DataFrame):
            # keep the caller's dtypes (mixed continuous/categorical data is
            # required by gcm classifiers — an object-dtype numpy array
            # would coerce every column to strings)
            df = X.loc[:, columns]
        else:
            df = pd.DataFrame(np.asarray(X), columns=columns)
        gcm.fit(self._cm, df)
        # cache the empirical noise sources per node
        from dowhy.gcm import AdditiveNoiseModel, ClassifierFCM

        for node in self.graph.nodes:
            mech = self._mech[node]
            if isinstance(mech, AdditiveNoiseModel):
                parents = get_parents(self.graph, node)
                residuals = mech.estimate_noise(
                    df[[node]].to_numpy(), df[parents].to_numpy()
                )
                self._residuals[node] = np.asarray(residuals).ravel()
            elif isinstance(mech, ClassifierFCM):
                # gcm.fit already fitted the classifier; nothing extra to cache
                pass
            else:  # EmpiricalDistribution root
                self._data[node] = np.asarray(mech.data).ravel()
        return self

    def response(self, node, parent_values):
        from dowhy.gcm import ClassifierFCM

        mech = self._mech[node]
        if isinstance(mech, ClassifierFCM):
            # the deterministic part of a classifier FCM contributes nothing:
            # the CLASS comes from draw_noise (inverse-CDF sampled from the
            # classifier's class probabilities over the PENDING parent
            # values — draw_noise receives no parents, so they are captured
            # here at the last response call).
            self._pending[node] = parent_values
            return np.zeros(parent_values.shape[0])
        if hasattr(mech, "prediction_model"):
            # NaN-cascade semantics (G9): sklearn estimators REJECT NaN
            # input, so evaluate finite rows only and let NaN parents
            # propagate to a NaN response (the mixin's sample/intervene
            # then cascade NaNs to the descendants exactly like the pure
            # NumPy backends did).
            parent_values = np.asarray(parent_values, dtype=float)
            finite = np.isfinite(parent_values).all(axis=1)
            out = np.full(parent_values.shape[0], np.nan)
            if finite.any():
                out[finite] = np.asarray(
                    mech.prediction_model.predict(parent_values[finite]),
                    dtype=float,
                ).ravel()
            return out
        # EmpiricalDistribution root: the marginal comes entirely from
        # draw_noise; the deterministic part contributes zeros (the mixin's
        # sample/intervene always call response, even for roots, with
        # parent_values of shape (n, 0)).
        return np.zeros(parent_values.shape[0])

    def draw_noise(self, node, rng, n):
        if node in self._clf:
            # classifier FCM: the noise draw IS the class — inverse-CDF
            # sampling of the class probabilities over the pending parent
            # values (gcm's own evaluate is unseeded, never call it here).
            if node not in self._pending:
                raise RuntimeError(
                    f"GcmSCMDGP.draw_noise for classifier node '{node}' requires a "
                    f"preceding response(node, parent_values) call in the same cascade"
                )
            pending = np.asarray(self._pending.pop(node), dtype=float)
            # NaN-cascade semantics: NaN parent rows get NaN classes
            # (no probabilities can be estimated for them).
            finite = np.isfinite(pending).all(axis=1)
            idx = np.full(n, -1)
            if finite.any():
                P = np.asarray(
                    self._clf[node].estimate_probabilities(pending[finite]),
                    dtype=float,
                )
                u = rng.random(int(finite.sum()))
                cum = np.cumsum(P, axis=1)
                idx[finite] = (u[:, None] > cum).sum(axis=1)
                idx[finite] = np.clip(idx[finite], 0, P.shape[1] - 1)
            out = np.full(n, np.nan)
            finite_idx = idx >= 0
            out[finite_idx] = np.asarray(
                self._clf[node].get_class_names(idx[finite_idx]), dtype=float
            )
            return out
        if node in self._residuals:
            # empirical noise: resample fitted residuals with the caller's rng
            return rng.choice(self._residuals[node], size=n)
        # root: resample the observed marginal (response contributes zeros)
        return rng.choice(self._data[node], size=n)
