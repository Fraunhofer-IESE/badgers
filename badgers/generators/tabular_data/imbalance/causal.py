"""Causal imbalance generator.

Injects selection bias by removing rows whose perturbation-node values fall
outside a truncated quantile range (the intersection across multiple
perturbation nodes). The result is a pure subset of the input rows: the kept
rows' parents are unchanged, so no structural equation is re-evaluated and
no DGP is needed.
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalImbalanceGenerator(CausalGenerator):
    """Generates selection bias as a quantile-range truncation (imbalance).

    The row selection is observation-side only: rows whose perturbation-node
    values fall outside ``quantile_range`` are removed; kept rows are
    returned untouched (a pure subset). No DGP is required.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes : list of str
        Nodes whose quantile ranges define the kept set.
    quantile_range : tuple of (float, float), default=(0.5, 1.0)
        Quantile range of the perturbation node's values that is KEPT. With
        multiple perturbation nodes, the kept set is the intersection of all
        ranges. Rows outside are removed (in-place subset).
    """

    deficit_name = "imbalance"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        quantile_range=(0.5, 1.0),
    ):
        """Inject causally-propagated imbalance in place (subset of rows)."""
        lo, hi = quantile_range
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"quantile_range must satisfy 0 <= lo < hi <= 1, "
                f"got {quantile_range}"
            )
        self._require_fitted()
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_samples = X.shape[0]

        # Keep rows inside the intersection of all perturbation-node ranges.
        keep = np.ones(n_samples, dtype=bool)
        for node in perturbation_nodes:
            col = node_to_col[node]
            b_lo = np.quantile(X[:, col], lo)
            b_hi = np.quantile(X[:, col], hi)
            keep &= (X[:, col] >= b_lo) & (X[:, col] <= b_hi)

        X = X[keep]
        if y is not None:
            y = y[keep]
        affected = np.ones(X.shape[0], dtype=bool)

        self.affected_mask_ = affected
        yt = self._labels_inplace(y, affected, self.deficit_name)
        return X, yt
