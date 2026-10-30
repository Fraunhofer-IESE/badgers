"""Causal outlier propagation generator."""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import (
    OutOfDistributionSampler,
    create_out_of_distribution_sampler,
)
from badgers.generators.tabular_data.causal._base import CausalGenerator
from badgers.generators.tabular_data.outliers._selection import (
    select_and_seed,
    iter_selection_groups,
)


class CausalOutlierPropagationGenerator(CausalGenerator):
    """
    Generates outlier rows by copying real rows of ``X`` and intervening
    per cell on the nodes the selector selects, propagating the effect to
    all descendants.

    The generator synthesizes rows (it appends to ``X`` instead of
    modifying it), so it keeps its own target input: the selector OWNS
    the targets of the generated rows (``selector=None`` raises). The
    selector realizes on ``X``: the rows it selects are the SEEDS, appended as
    copies of those rows (every appended row has at least one selected cell —
    labels all ``True``). Each selected target cell is pinned to an
    out-of-distribution
    value (per-cell pinning — a row is pinned only where its own selection
    pattern selects), while an unselected cell keeps the seed row's own
    value EXCEPT when it is a descendant of a selected target: descendants
    recompute through the structural equations. Each group of rows
    sharing one distinct selection pattern gets its own surgery through the
    fitted DGP (``dgp.intervene`` — Pearl surgery: selected targets stay
    pinned, their descendants get fresh noise).

    Value-driven selectors read ``X`` directly (no bootstrap
    indirection): e.g. ``TopQuantileSelector`` seeds the top-quantile
    rows, ``ClassCountSelector`` seeds exact per-class row counts.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (fitted or unfitted) carrying the graph and
        column_mapping. No other structure is accepted: if None,
        ``generate()`` raises (build one with e.g.
        ``GcmSCMDGP(scm).fit(X)``).

    generate() parameters (passed via **params)
    ----------
    selector : Selector
        Row/column selection rule from ``badgers.core.selectors``
        (REQUIRED; ``None`` raises -- there is no relevant default: the
        selector owns the intervention targets). Its ``columns`` are the
        intervention targets of the generated rows; the selected rows of
        ``X`` are the seeds.
    out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
        Sampling strategy for the pinned target cells.
        A string is resolved via :func:`create_out_of_distribution_sampler`.
    """

    deficit_name = "outliers"

    def __init__(self, random_generator, dgp=None):
        super().__init__(random_generator=random_generator, dgp=dgp)
        self.outlier_labels_ = None

    def get_outliers_labels(self) -> np.ndarray:
        """
        Return the outlier labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: ``True`` for the appended rows (every appended row is a
        selected seed), ``False`` for original rows.
        """
        if self.outlier_labels_ is None:
            raise RuntimeError(
                "Outlier labels are only available after calling generate()."
            )
        return self.outlier_labels_

    @preprocess_inputs
    def generate(self, X, selector, *, out_of_distribution_sampler=None):
        """
        Generate outlier rows with per-cell pinned intervention targets.

        The selector decides which rows of ``X`` seed the generated rows
        and which target cells of them are pinned; each distinct selection
        pattern gets its own surgery.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (the world the outlier rows are seeded from).
        selector : Selector
            Row/column selection rule from ``badgers.core.selectors``
            (REQUIRED; ``None`` raises -- there is no relevant default:
            the selector owns the intervention targets). Its ``columns``
            own the intervention targets; the selected rows of ``X`` are the
            seeds, so value-driven presets read the original data.
        out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
            Sampling strategy for the pinned target cells.
            A string is resolved via
            :func:`create_out_of_distribution_sampler`.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples + n_seeds, n_features)
            Original data with the outlier rows appended (no appended rows
            when the selector selects nothing).
        """
        rng = self.random_generator
        if selector is None:
            raise ValueError(
                "selector is required: it owns the intervention targets "
                "(e.g. CountSelector([0], count=10) selects 10 seed rows "
                "on column 0)")
        self._validate_structure(X)

        # Resolve out-of-distribution sampler (the pinned target cells)
        if out_of_distribution_sampler is None:
            ood = create_out_of_distribution_sampler("zscore")
        elif isinstance(out_of_distribution_sampler, str):
            ood = create_out_of_distribution_sampler(
                out_of_distribution_sampler
            )
        elif isinstance(out_of_distribution_sampler, OutOfDistributionSampler):
            ood = out_of_distribution_sampler
        else:
            raise ValueError(
                f"out_of_distribution_sampler must be an "
                f"OutOfDistributionSampler instance or str, "
                f"got {type(out_of_distribution_sampler).__name__}"
            )

        # Require a (lazily) fitted DGP on the ORIGINAL data before any
        # column mutation. Unfitted DGPs are fitted here; there is no
        # implicit default backend.
        self._require_dgp(X)

        # 1) the selector realizes on X: the selected rows seed the generated
        #    rows (copies of those rows — every appended row has at least one
        #    selected cell)
        field, targets, seed_rows = select_and_seed(X, selector, rng)
        self.outlier_labels_ = np.concatenate([
            np.zeros(X.shape[0], dtype=bool),
            np.ones(seed_rows.size, dtype=bool),
        ])
        if seed_rows.size == 0:
            return X
        outliers = X[seed_rows].copy()

        # 2) one surgery per distinct selection pattern: only the cells this
        #    pattern selected are pinned; the group's rows are then
        #    recomputed through the structural equations (Pearl surgery:
        #    pinned targets stay pinned, their descendants get fresh
        #    noise; unselected non-descendants keep the seed row's values)
        for group_rows, selected_cols in iter_selection_groups(field, targets, seed_rows):
            outliers[np.ix_(group_rows, selected_cols)] = (
                ood.sample(rng, X, selected_cols, group_rows.size))
            outliers[group_rows] = self.dgp.intervene(
                outliers[group_rows], self._node_names(selected_cols), rng=rng)
        return np.vstack([X, outliers])
