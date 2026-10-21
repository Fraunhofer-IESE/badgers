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
    selector : Selector or None, default=None
        Row-selection rule from ``badgers.core.selectors``. The rows FIRED
        by the selector are dropped (subsampling polarity: ``True`` =
        deficit applied = row removed); with multiple variables the union
        of fired rows is dropped. ``None`` falls back to the
        ``perturbation_nodes`` + ``quantile_range`` rule below.
    perturbation_nodes : list of str
        Nodes whose quantile ranges define the kept set (only used when
        ``selector=None``).
    quantile_range : tuple of (float, float), default=(0.5, 1.0)
        Quantile range of the perturbation node's values that is KEPT. With
        multiple perturbation nodes, the kept set is the intersection of all
        ranges. Rows outside are removed (in-place subset). Only used when
        ``selector=None``.
    """

    deficit_name = "imbalance"

    def get_imbalance_labels(self) -> np.ndarray:
        """Return the imbalance labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the imbalance deficit (the kept
        subset), False for... none (all kept rows are affected).
        """
        if self.deficit_labels_ is None:
            raise RuntimeError(
                "Imbalance labels are only available after calling "
                "generate()."
            )
        return self.deficit_labels_

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        selector=None,
        perturbation_nodes=None,
        quantile_range=(0.5, 1.0),
    ):
        """Inject causally-propagated imbalance in place (subset of rows)."""
        self._require_fitted()
        n_samples = X.shape[0]

        if selector is not None:
            # Explicit selector: rows FIRED by the selector are dropped
            # (subsampling polarity). Union over the selector's variables.
            dropped = np.zeros(n_samples, dtype=bool)
            for mask in selector(X, y).values():
                dropped |= np.asarray(mask, dtype=bool)
            keep = ~dropped
        else:
            self._validate_structure(X)
            self._validate_perturbation_nodes(perturbation_nodes)
            node_to_col = self.column_mapping
            # Legacy quantile-range rule: keep rows inside the intersection
            # of all perturbation-node ranges.
            lo, hi = quantile_range
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"quantile_range must satisfy 0 <= lo < hi <= 1, "
                    f"got {quantile_range}"
                )
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

        self.deficit_labels_ = affected.copy()
        return X, None
