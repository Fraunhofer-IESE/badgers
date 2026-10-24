"""Causal outlier propagation generator."""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import fired
from badgers.core.selectors.base import _resolve_targets
from badgers.core.sampling import (
    WithinDistributionSampler,
    OutOfDistributionSampler,
    create_within_distribution_sampler,
    create_out_of_distribution_sampler,
)
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalOutlierPropagationGenerator(CausalGenerator):
    """
    Generates outlier rows by intervening per cell on the nodes the
    selector fires, and propagating the effect to all descendants.

    The generator synthesizes rows (it appends to ``X`` instead of
    modifying it), so it keeps its own target input: the selector OWNS
    the targets of the generated rows (``selector=None`` raises).
    ``n_outliers`` rows are first bootstrapped from ``X`` (the within
    sampler draws EVERY column — a full resample of the world, replacing
    the previous zero-fill of the untouched columns with plausible
    values). The selector is then applied to the BOOTSTRAPPED rows: each
    fired target cell is pinned to an out-of-distribution value
    (per-cell pinning — a row is pinned only where its own firing
    pattern says fire), while an unfired target cell keeps its bootstrap
    value. Each group of rows sharing one distinct firing pattern gets
    its own surgery through the fitted DGP (``dgp.intervene`` — Pearl
    surgery: fired targets stay pinned, their descendants recompute with
    fresh noise; a target that is a DESCENDANT of a fired target
    recomputes from the fired parents).

    Rows where the selector fires nowhere stay pure bootstrap rows and
    are labeled ``False`` by :meth:`get_outliers_labels`.

    Value-free presets are recommended (``AlwaysSelector``,
    ``BernoulliSelector``, ``ExactCountSelector``): a value-driven
    selector reads the BOOTSTRAPPED rows, not ``X``.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
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
        (REQUIRED: ``None`` is rejected). Its ``columns`` are the
        intervention targets of the generated rows; firing is decided
        on the BOOTSTRAPPED rows.
    n_outliers : int, default=10
        Number of outlier rows to generate.
    within_distribution_sampler : WithinDistributionSampler or str, default="normal"
        Sampling strategy for the bootstrap rows (all columns).
        A string is resolved via :func:`create_within_distribution_sampler`.
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
        output: ``True`` for rows where the selector fired at least one
        target cell (the outlier deficit), ``False`` for original rows
        and for pure bootstrap rows (nothing fired).
        """
        if self.outlier_labels_ is None:
            raise RuntimeError(
                "Outlier labels are only available after calling generate()."
            )
        return self.outlier_labels_

    @preprocess_inputs
    def generate(self, X, selector=None, n_outliers=10,
                 within_distribution_sampler=None,
                 out_of_distribution_sampler=None):
        """
        Generate outlier rows with per-cell pinned intervention targets.

        The selector decides which target cells of the generated rows
        are pinned; each distinct firing pattern gets its own surgery.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (the world the outlier rows are bootstrapped from).
        selector : Selector
            Row/column selection rule from ``badgers.core.selectors``
            (REQUIRED: ``None`` is rejected). Its ``columns`` own the
            intervention targets; firing is decided on the BOOTSTRAPPED
            rows, so value-free presets are recommended.
        n_outliers : int, default=10
            Number of outlier rows to generate.
        within_distribution_sampler : WithinDistributionSampler or str, default="normal"
            Sampling strategy for the bootstrap rows (all columns).
            A string is resolved via
            :func:`create_within_distribution_sampler`.
        out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
            Sampling strategy for the pinned target cells.
            A string is resolved via
            :func:`create_out_of_distribution_sampler`.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples + n_outliers, n_features)
            Original data with the outlier rows appended.
        """
        rng = self.random_generator
        if selector is None:
            raise ValueError(
                "selector is required: it owns the intervention targets "
                "(e.g. AlwaysSelector([0]) fires every row on column 0)")
        if n_outliers <= 0:
            raise ValueError(
                f"n_outliers must be positive, got {n_outliers}"
            )
        self._validate_structure(X)

        # Resolve within-distribution sampler (the bootstrap rows)
        if within_distribution_sampler is None:
            within = create_within_distribution_sampler("normal")
        elif isinstance(within_distribution_sampler, str):
            within = create_within_distribution_sampler(
                within_distribution_sampler
            )
        elif isinstance(within_distribution_sampler, WithinDistributionSampler):
            within = within_distribution_sampler
        else:
            raise ValueError(
                f"within_distribution_sampler must be a "
                f"WithinDistributionSampler instance or str, "
                f"got {type(within_distribution_sampler).__name__}"
            )

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

        # 1) bootstrap the generated rows (all columns; the row is a full
        #    resample from the world)
        n_features = X.shape[1]
        outliers = np.asarray(
            within.sample(rng, X, list(range(n_features)), n_outliers),
            dtype=X.dtype,
        )

        # 2) the selector decides which TARGET CELLS are pinned (firing
        #    is decided on the BOOTSTRAPPED rows)
        field = fired(outliers, selector, rng)
        targets = list(_resolve_targets(selector, outliers, None))
        self.outlier_labels_ = np.concatenate(
            [np.zeros(X.shape[0], dtype=bool), field.any(axis=1)])

        # 3) one surgery per distinct firing pattern: only the cells this
        #    pattern fired are pinned; the group's rows are then
        #    recomputed through the structural equations (Pearl surgery:
        #    pinned targets stay pinned, their descendants get fresh
        #    noise). Pure bootstrap rows are left as bootstrapped.
        patterns, group_of_row = np.unique(field, axis=0, return_inverse=True)
        for gi, pattern in enumerate(patterns):
            group_rows = np.flatnonzero(group_of_row == gi)
            fired_cols = [targets[j] for j in np.flatnonzero(pattern)]
            if not fired_cols:
                continue  # pure bootstrap rows: nothing fired, nothing pinned
            outliers[np.ix_(group_rows, fired_cols)] = (
                ood.sample(rng, X, fired_cols, group_rows.size))
            outliers[group_rows] = self.dgp.intervene(
                outliers[group_rows], self._node_names(fired_cols), rng=rng)
        return np.vstack([X, outliers])
