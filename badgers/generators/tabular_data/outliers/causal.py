"""Causal outlier propagation generator."""

import numpy as np

from badgers.core.causal_graph import get_ancestors, get_descendants
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import (
    WithinDistributionSampler,
    OutOfDistributionSampler,
    create_within_distribution_sampler,
    create_out_of_distribution_sampler,
)
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalOutlierPropagationGenerator(CausalGenerator):
    """
    Generates outliers by intervening on one or more nodes in a causal
    graph and propagating the effect to all descendants.

    Implements the do()-style intervention via the shared DGP machinery:
    incoming edges to perturbation nodes are severed (their values are
    pinned), ancestors are sampled from their distributions, perturbation
    nodes are set directly to outlier values, and only descendants are
    recomputed through the fitted DGP (``dgp.intervene`` — Pearl surgery
    with fresh noise, uniform across backends).

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
    perturbation_nodes : list of str
        Nodes where the intervention is applied. Each outlier row gets
        perturbations at ALL of these nodes simultaneously.
    n_outliers : int, default=10
        Number of outlier rows to generate.
    within_distribution_sampler : WithinDistributionSampler or str, default="normal"
        Sampling strategy for exogenous (ancestor) nodes.
        A string is resolved via :func:`create_within_distribution_sampler`.
    out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
        Sampling strategy for perturbation nodes.
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
        output: True for rows that are outliers, False for original rows.
        """
        if self.outlier_labels_ is None:
            raise RuntimeError(
                "Outlier labels are only available after calling generate()."
            )
        return self.outlier_labels_

    def _set_outlier_labels(self, n_original: int, n_outliers: int):
        self.outlier_labels_ = np.concatenate(
            [np.zeros(n_original, dtype=bool), np.ones(n_outliers, dtype=bool)]
        )

    @preprocess_inputs
    def generate(self, X, y=None, perturbation_nodes=None, n_outliers=10,
                 within_distribution_sampler=None,
                 out_of_distribution_sampler=None):
        """
        Generate outliers by intervening on perturbation_nodes and
        propagating the effect to all descendants.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : np.ndarray or None
            Ignored (deprecated). The target is not modified; use
            ``get_outliers_labels()`` for the per-row outlier labels of
            the generated output.
        perturbation_nodes : list of int or str
            Nodes where the intervention is applied.
        n_outliers : int, default=10
            Number of outlier rows to generate.
        within_distribution_sampler : WithinDistributionSampler or str, default="normal"
            Sampling strategy for exogenous (ancestor) nodes.
            A string is resolved via
            :func:`create_within_distribution_sampler`.
        out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
            Sampling strategy for perturbation nodes.
            A string is resolved via
            :func:`create_out_of_distribution_sampler`.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples + n_outliers, n_features)
            Original data with outlier rows appended.
        y : np.ndarray or None
            Ignored (deprecated). The target is not modified; use
            ``get_outliers_labels()`` for the per-row outlier labels of
            the generated output.
        """
        self._validate_structure(X)
        self._validate_perturbation_nodes(perturbation_nodes)
        node_to_col = self.column_mapping
        graph = self.graph

        # Graph algebra for the propagation: descendants of the perturbation
        # nodes, and their exogenous (ancestor) nodes -- sampled from their
        # distributions rather than intervened on.
        descendants = set()
        for pn in perturbation_nodes:
            descendants |= get_descendants(graph, pn)
        ancestors = set()
        for pn in perturbation_nodes:
            ancestors |= get_ancestors(graph, pn)
        exogenous = ancestors - set(perturbation_nodes) - descendants
        perturb_cols = [node_to_col[n] for n in perturbation_nodes]

        if n_outliers <= 0:
            raise ValueError(
                f"n_outliers must be positive, got {n_outliers}"
            )

        n_samples, n_features = X.shape

        # Resolve within-distribution sampler (for exogenous nodes)
        if within_distribution_sampler is None:
            within_distribution_sampler = create_within_distribution_sampler("normal")
        elif isinstance(within_distribution_sampler, str):
            within_distribution_sampler = create_within_distribution_sampler(
                within_distribution_sampler
            )
        elif not isinstance(within_distribution_sampler, WithinDistributionSampler):
            raise ValueError(
                f"within_distribution_sampler must be a "
                f"WithinDistributionSampler instance or str, "
                f"got {type(within_distribution_sampler).__name__}"
            )

        # Resolve out-of-distribution sampler (for perturbation nodes)
        if out_of_distribution_sampler is None:
            out_of_distribution_sampler = create_out_of_distribution_sampler("zscore")
        elif isinstance(out_of_distribution_sampler, str):
            out_of_distribution_sampler = create_out_of_distribution_sampler(
                out_of_distribution_sampler
            )
        elif not isinstance(out_of_distribution_sampler, OutOfDistributionSampler):
            raise ValueError(
                f"out_of_distribution_sampler must be an "
                f"OutOfDistributionSampler instance or str, "
                f"got {type(out_of_distribution_sampler).__name__}"
            )

        # Require a (lazily) fitted DGP on the ORIGINAL data before any
        # column mutation. Unfitted DGPs are fitted here; there is no
        # implicit default backend.
        self._require_dgp(X)

        exog_cols = [node_to_col[n] for n in exogenous]

        # Generate n_outliers outlier rows
        outliers = np.zeros((n_outliers, n_features))

        # Step 1: Sample exogenous nodes from their distributions
        if exog_cols:
            outliers[:, exog_cols] = within_distribution_sampler.sample(
                self.random_generator, X, exog_cols, n_outliers,
            )

        # Step 2: Set perturbation nodes directly (sever incoming edges).
        # The out_of_distribution_sampler already handles sign (±) and
        # magnitude (e.g., 3σ + exponential for zscore), so no additional
        # perturbation is needed here.
        if perturb_cols:
            outliers[:, perturb_cols] = out_of_distribution_sampler.sample(
                self.random_generator, X, perturb_cols, n_outliers,
            )

        # Step 3: Propagate the descendants through the DGP (Pearl surgery:
        # pinned perturbation values stay pinned; every descendant is
        # recomputed from its parents via the fitted mechanisms with fresh
        # noise — same generic cascade as every other causal generator).
        self.dgp.intervene(outliers, perturbation_nodes,
                           rng=self.random_generator)

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(n_samples, n_outliers)
        return Xt, None
