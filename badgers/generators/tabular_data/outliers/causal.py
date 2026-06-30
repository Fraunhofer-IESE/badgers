"""Causal outlier propagation generator."""

import networkx as nx
import numpy as np
from numpy.random import default_rng

from badgers.core.causal_graph import (
    get_ancestors,
    get_descendants,
    get_parents,
    topological_order,
    validate_dag,
)
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import (
    WithinDistributionSampler,
    OutOfDistributionSampler,
    create_within_distribution_sampler,
    create_out_of_distribution_sampler,
)
from badgers.core.utils import random_sign
from badgers.generators.tabular_data.outliers import OutliersGenerator


class CausalOutlierPropagationGenerator(OutliersGenerator):
    """
    Generates outliers by intervening on one or more nodes in a causal
    graph and propagating the effect to all descendants.

    Implements the do()-style intervention: incoming edges to perturbation
    nodes are severed, ancestors are sampled from their distributions,
    perturbation nodes are set directly to outlier values, and only
    descendants are computed via forward pass.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.

    generate() parameters (passed via **params)
    ----------
    graph : nx.DiGraph
        Causal DAG. All nodes must be strings.
    column_mapping : dict of str -> int
        Mapping from graph node names to column indices in X. Required.
    perturbation_nodes : list of str
        Nodes where the intervention is applied. Each outlier row gets
        perturbations at ALL of these nodes simultaneously.
    n_outliers : int, default=10
        Number of outlier rows to generate.
    outlier_magnitude : float, default=3.0
        Number of standard deviations to perturb by.
    within_distribution_sampler : WithinDistributionSampler or str, default="normal"
        Sampling strategy for exogenous (ancestor) nodes.
        A string is resolved via :func:`create_within_distribution_sampler`.
    out_of_distribution_sampler : OutOfDistributionSampler or str, default="zscore"
        Sampling strategy for perturbation nodes.
        A string is resolved via :func:`create_out_of_distribution_sampler`.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y, graph=None, perturbation_nodes=None,
                 column_mapping=None, n_outliers=10,
                 outlier_magnitude=3.0,
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
            Target values. If None, a label array is created with
            "original" for input rows and "outliers" for generated rows.
            If provided, "outliers" labels are appended.
        graph : nx.DiGraph
            Causal DAG.
        perturbation_nodes : list of int or str
            Nodes where the intervention is applied.
        column_mapping : dict of str -> int
            Mapping from graph node names to column indices in X.
            Required. Every graph node must have an entry, and all
            values must be unique integers in 0..n_features-1.
        n_outliers : int, default=10
            Number of outlier rows to generate.
        outlier_magnitude : float, default=3.0
            Number of standard deviations to perturb by.
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
        yt : np.ndarray
            Labels: "original" for input rows, "outliers" for generated rows.
        """
        if graph is None:
            raise ValueError("graph parameter is required")

        validate_dag(graph)

        if perturbation_nodes is None:
            raise ValueError("perturbation_nodes parameter is required")

        if not isinstance(perturbation_nodes, list):
            raise ValueError(
                "perturbation_nodes must be a list, "
                f"got {type(perturbation_nodes).__name__}"
            )

        if len(perturbation_nodes) == 0:
            raise ValueError("perturbation_nodes must not be empty")

        for node in perturbation_nodes:
            if node not in graph.nodes:
                raise ValueError(
                    f"perturbation node '{node}' not found in graph"
                )

        n_samples, n_features = X.shape
        if n_features != len(graph.nodes):
            raise ValueError(
                f"X has {n_features} columns but graph has "
                f"{len(graph.nodes)} nodes"
            )

        if n_outliers <= 0:
            raise ValueError(
                f"n_outliers must be positive, got {n_outliers}"
            )

        # Validate column_mapping
        if column_mapping is None:
            raise ValueError("column_mapping parameter is required")

        if not isinstance(column_mapping, dict):
            raise ValueError(
                "column_mapping must be a dict, "
                f"got {type(column_mapping).__name__}"
            )

        # Check all graph nodes are strings
        for node in graph.nodes:
            if not isinstance(node, str):
                raise ValueError(
                    f"graph nodes must be strings, "
                    f"got {type(node).__name__} for node '{node}'"
                )

        # Check every graph node has a mapping entry
        missing = set(graph.nodes) - set(column_mapping.keys())
        if missing:
            raise ValueError(
                f"column_mapping missing node(s): {sorted(missing)}"
            )

        # Check no extra entries
        extra = set(column_mapping.keys()) - set(graph.nodes)
        if extra:
            raise ValueError(
                f"column_mapping has unknown node(s): {sorted(extra)}"
            )

        # Check all values are int and in range
        for node, idx in column_mapping.items():
            if not isinstance(idx, int):
                raise ValueError(
                    f"column_mapping values must be int, "
                    f"got {type(idx).__name__} for node '{node}'"
                )
            if idx < 0 or idx >= n_features:
                raise ValueError(
                    f"column index {idx} for node '{node}' "
                    f"is out of range [0, {n_features - 1}]"
                )

        # Check no duplicate column indices
        seen = {}
        for node, idx in column_mapping.items():
            if idx in seen:
                raise ValueError(
                    f"duplicate column index {idx} "
                    f"for nodes {sorted([seen[idx], node])}"
                )
            seen[idx] = node

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

        # Compute topological order
        order = topological_order(graph)

        # Use the explicit column mapping directly
        node_to_col = column_mapping

        # Fit linear coefficients: for each node v, regress X[:,v] on X[:,parents(v)]
        coefficients = {}
        for node in order:
            parents = get_parents(graph, node)
            if not parents:
                continue
            parent_cols = [node_to_col[p] for p in parents]
            X_parents = X[:, parent_cols]
            X_target = X[:, node_to_col[node]]
            beta, _, _, _ = np.linalg.lstsq(X_parents, X_target, rcond=None)
            for i, parent in enumerate(parents):
                coefficients[(parent, node)] = beta[i]

        # Compute column stds for perturbation magnitude
        col_stds = np.std(X, axis=0)
        col_stds[col_stds == 0] = 1.0

        # ---- do()-style intervention algorithm ----

        perturb_set = set(perturbation_nodes)

        # Identify all descendants of perturbation nodes (these propagate)
        descendants = set()
        for pn in perturbation_nodes:
            descendants |= get_descendants(graph, pn)

        # Identify exogenous nodes: ancestors of perturbation_nodes that
        # are NOT themselves perturbed and NOT descendants of perturbed nodes.
        # These are sampled from their natural distributions.
        ancestors = set()
        for pn in perturbation_nodes:
            ancestors |= get_ancestors(graph, pn)
        exogenous = ancestors - perturb_set - descendants
        exog_cols = [node_to_col[n] for n in exogenous]

        # Nodes that need forward-pass computation: descendants only
        descendant_cols = {node_to_col[n] for n in descendants}

        # Perturbation column indices
        perturb_cols = [node_to_col[n] for n in perturbation_nodes]

        # Generate n_outliers outlier rows
        outliers = np.zeros((n_outliers, n_features))

        # Step 1: Sample exogenous nodes from their distributions
        if exog_cols:
            outliers[:, exog_cols] = within_distribution_sampler.sample(
                self.random_generator, X, exog_cols, n_outliers,
            )

        # Step 2: Set perturbation nodes directly (sever incoming edges).
        # Sample base values, then add ±outlier_magnitude * std perturbation.
        if perturb_cols:
            outliers[:, perturb_cols] = out_of_distribution_sampler.sample(
                self.random_generator, X, perturb_cols, n_outliers,
            )
            for pn in perturbation_nodes:
                col = node_to_col[pn]
                signs = random_sign(self.random_generator, size=(n_outliers,))
                outliers[:, col] += signs * outlier_magnitude * col_stds[col]

        # Step 3: Forward-pass only to descendants of perturbation nodes.
        for node in order:
            col = node_to_col[node]
            if col not in descendant_cols:
                continue
            parents = get_parents(graph, node)
            if not parents:
                continue
            parent_cols = [node_to_col[p] for p in parents]
            parent_vals = outliers[:, parent_cols]
            coefs = np.array([
                coefficients.get((p, node), 0.0) for p in parents
            ])
            predicted = parent_vals @ coefs
            residual_std = np.std(
                X[:, col] - X[:, parent_cols] @ coefs
            )
            if residual_std == 0:
                residual_std = col_stds[col] * 0.1
            outliers[:, col] = predicted + self.random_generator.normal(
                0, residual_std, size=n_outliers
            )

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        # Build yt labels
        if y is None:
            yt = np.array(
                ["original"] * n_samples + ["outliers"] * n_outliers
            )
        else:
            yt = np.append(y, ["outliers"] * n_outliers)

        return Xt, yt
