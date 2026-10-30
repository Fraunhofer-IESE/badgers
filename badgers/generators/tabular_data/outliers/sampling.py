"""Generic non-causal outlier sampling generator.

Direct counterpart of
:class:`badgers.generators.tabular_data.outliers.causal.CausalOutlierPropagationGenerator`:
same contract (required ``selector`` + pluggable
``out_of_distribution_sampler``), no DGP surgery — the selected cells of
the seed rows are simply replaced, no other cell is recomputed.
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import (
    OutOfDistributionSampler,
    create_out_of_distribution_sampler,
)
from badgers.generators.tabular_data.outliers import OutliersGenerator
from badgers.generators.tabular_data.outliers._selection import (
    select_and_seed,
    iter_selection_groups,
)


class OutlierSamplingGenerator(OutliersGenerator):
    """
    Generates outlier rows by copying real rows of ``X`` and replacing the
    selector-selected cells with out-of-distribution values.

    The generator synthesizes rows (it appends to ``X`` instead of
    modifying it). The selector OWNS the targets of the generated rows
    (``selector=None`` raises): its ``columns`` are the perturbed cells and
    the rows it selects are the SEEDS — each appended outlier row starts as
    a copy of a selected row, and every cell selected by the selector is
    replaced by a value drawn from the out-of-distribution sampler. A cell
    the selector did not select keeps the seed row's own value. Each
    distinct selection pattern is sampled as one group.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.

    generate() parameters
    ---------------------
    selector : Selector
        Row/column selection rule from ``badgers.core.selectors``
        (REQUIRED; ``None`` raises — there is no relevant default: the
        selector owns the count and the perturbed targets).
    out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
        Sampling strategy for the selected cells.
        A string is resolved via :func:`badgers.core.sampling.create_out_of_distribution_sampler`.
    """

    deficit_name = "outliers"

    @preprocess_inputs
    def generate(self, X, selector, *, out_of_distribution_sampler=None):
        """
        Generate outlier rows with per-cell out-of-distribution values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (the world the outlier rows are seeded from).
        selector : Selector
            Row/column selection rule from ``badgers.core.selectors``
            (REQUIRED; ``None`` raises). Its ``columns`` are the perturbed
            targets; the selected rows of ``X`` are the seeds.
        out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
            Sampling strategy for the selected cells.
            A string is resolved via
            :func:`badgers.core.sampling.create_out_of_distribution_sampler`.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples + n_seeds, n_features)
            Original data with the outlier rows appended (no appended rows
            when the selector selects nothing).
        """
        if selector is None:
            raise ValueError(
                "selector is required: it owns the outlier count and the "
                "perturbed targets (e.g. CountSelector([0], count=10) "
                "selects 10 seed rows on column 0)")
        rng = self.random_generator

        # Resolve out-of-distribution sampler (the perturbed cells)
        if out_of_distribution_sampler is None:
            ood = create_out_of_distribution_sampler("zscore")
        elif isinstance(out_of_distribution_sampler, str):
            ood = create_out_of_distribution_sampler(out_of_distribution_sampler)
        elif isinstance(out_of_distribution_sampler, OutOfDistributionSampler):
            ood = out_of_distribution_sampler
        else:
            raise ValueError(
                f"out_of_distribution_sampler must be an "
                f"OutOfDistributionSampler instance or str, "
                f"got {type(out_of_distribution_sampler).__name__}"
            )

        field, targets, seed_rows = select_and_seed(X, selector, rng)
        self._set_outlier_labels(len(X), seed_rows.size)
        if seed_rows.size == 0:
            return X
        outliers = X[seed_rows].astype(np.float64, copy=True)
        for group_rows, selected_cols in iter_selection_groups(
                field, targets, seed_rows):
            outliers[np.ix_(group_rows, selected_cols)] = (
                ood.sample(rng, X, selected_cols, group_rows.size))
        return np.vstack([X, outliers])
