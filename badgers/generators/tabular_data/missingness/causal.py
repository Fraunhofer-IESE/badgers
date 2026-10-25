"""Causal missingness generator (observation layer).

Chooser-only: injects missingness (NaN) at the cells fired by the
selector. The deficit does NOT cascade to descendants -- missingness is a
property of the measurement process, not of the world (NO intervene: the
DGP world is never re-evaluated).

- **Cause** (``selector``): why/when a value is missing -- the m-graph $R$
  mechanism (``BernoulliSelector``=MCAR, driver-mode ``TopQuantileSelector``
  =MAR, self-mode =MNAR).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import fired
from badgers.core.selectors.base import _resolve_targets
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalMissingnessGenerator(CausalGenerator):
    """Generates observation-layer missingness (chooser-only).

    The SELECTOR is required and owns the intervention targets: the
    selector's own ``columns`` are the columns where missingness is
    injected (``selector=None`` raises).

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.

    generate() parameters (passed via **params)
    ----------
    selector : Selector
        Row-selection rule from ``badgers.core.selectors`` (REQUIRED;
        ``None`` raises -- there is no relevant default: the selector
        owns the missingness targets).
        ``BernoulliSelector`` = MCAR, ``TopQuantileSelector`` with
        ``driver`` = MAR, self mode = MNAR, ``ExactCountSelector``,
        ``AlwaysSelector``, or any callable with the same signature.
        Its ``columns`` are the missingness targets; the cells it fires
        are written NaN.
    """

    deficit_name = "missingness"

    def get_missing_labels(self) -> np.ndarray:
        """Return the missingness labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that contain at least one missing value
        (deficit fired), False for original rows.
        """
        return self._get_deficit_labels()

    @preprocess_inputs
    def generate(
        self,
        X,
        selector,
    ):
        """Inject missingness at the fired cells (observation-layer).

        Chooser-only: a Selector decides WHICH cells fire (e.g. value-blind
        Bernoulli, own-value top-quantile, or driver top-quantile); the
        generator writes NaN at the fired cells. NO propagation: the
        deficit does not cascade to descendants (observation semantics --
        missingness is a property of the measurement, not of the world).
        """
        self._require_fitted()
        if selector is None:
            raise ValueError(
                "selector is required: it owns the intervention targets"
            )
        self._validate_structure(X)
        targets = list(_resolve_targets(selector, X, None))
        rng = self.random_generator

        # The selector decides the affected cells: the firing field is
        # realized for ALL rows at once (vectorized; value-driven rules
        # read the un-mutated data).
        field = fired(X, selector, rng)
        affected = np.zeros(X.shape[0], dtype=bool)
        for i, col in enumerate(targets):
            mask = field[:, i]
            X[mask, col] = np.nan
            affected |= mask

        self.deficit_labels_ = affected.copy()
        return X
