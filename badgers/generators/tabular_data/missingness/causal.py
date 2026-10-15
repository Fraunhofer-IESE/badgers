"""Causal missingness generator (observation layer).

Chooser-only: injects missingness (NaN) at the fired cells of the
perturbation nodes. The deficit does NOT cascade to descendants --
missingness is a property of the measurement process, not of the world.

- **Cause** (``mechanism``): why/when a value is missing -- the m-graph $R$
  mechanism (``random``=MCAR, ``observed``=MAR, ``self``=MNAR), backed by
  the selector presets (BernoulliSelector / TopQuantileSelector).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalMissingnessGenerator(CausalGenerator):
    """Generates observation-layer missingness (chooser-only).

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes : list of str
        Nodes where missingness is injected.
    mechanism : str, default="random"
        Cause of missingness: ``"always"``, ``"random"`` (MCAR),
        ``"observed"`` (MAR), ``"self"`` (MNAR).
    mechanism_cause : str, default=None
        Fully-observed node driving missingness (required for ``"observed"``).
    mechanism_rate : float, default=0.3
        Base missingness probability / range fraction.
    """

    deficit_name = "missingness"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        mechanism="random",
        mechanism_cause=None,
        mechanism_rate=0.3,
    ):
        """Inject missingness at the fired cells (observation-layer).

        Chooser-only: a Selector decides WHICH cells fire (value-blind
        Bernoulli, own-value top-quantile, or driver top-quantile); the
        mechanism writes NaN at the fired cells. NO propagation: the
        deficit does not cascade to descendants (observation semantics --
        missingness is a property of the measurement, not of the world).
        """
        self._require_fitted()
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]

        # Gate the defect by the mechanism: fire masks for ALL rows at once
        # (vectorized; quantile thresholds computed on the un-mutated data).
        fires = self._fire_masks(
            X, perturbation_nodes, node_to_col, mechanism, mechanism_cause,
            mechanism_rate,
        )
        affected = np.zeros(n_samples, dtype=bool)
        for node in perturbation_nodes:
            mask = fires[node]
            X[mask, node_to_col[node]] = np.nan
            affected |= mask

        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt

