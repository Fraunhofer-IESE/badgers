"""Causal missingness generator (observation layer).

Chooser-only: injects missingness (NaN) at the fired cells of the
perturbation nodes. The deficit does NOT cascade to descendants --
missingness is a property of the measurement process, not of the world.

- **Cause** (``selector``): why/when a value is missing -- the m-graph $R$
  mechanism (``BernoulliSelector``=MCAR, driver-mode ``TopQuantileSelector``
  =MAR, self-mode =MNAR).
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
    selector : Selector or None, default=None
        Row-selection rule from ``badgers.core.selectors``
        (``BernoulliSelector`` = MCAR, ``TopQuantileSelector`` with
        drivers = MAR, self mode = MNAR, ``ExactCountSelector``,
        ``AlwaysSelector``, or any callable with the same signature).
        ``None`` = every row fires.
    """

    deficit_name = "missingness"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        selector=None,
    ):
        """Inject missingness at the fired cells (observation-layer).

        Chooser-only: a Selector decides WHICH cells fire (e.g. value-blind
        Bernoulli, own-value top-quantile, or driver top-quantile); the
        generator writes NaN at the fired cells. NO propagation: the
        deficit does not cascade to descendants (observation semantics --
        missingness is a property of the measurement, not of the world).
        """
        self._require_fitted()
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]

        # Gate the defect by the selector: fire masks for ALL rows at once
        # (vectorized; quantile thresholds computed on the un-mutated data).
        fires = self._fire_masks(
            X, perturbation_nodes, node_to_col, selector=selector,
        )
        affected = np.zeros(n_samples, dtype=bool)
        for node in perturbation_nodes:
            mask = fires[node]
            X[mask, node_to_col[node]] = np.nan
            affected |= mask

        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt

