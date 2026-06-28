"""Causal outlier propagation generator."""

import networkx as nx
import numpy as np
from numpy.random import default_rng

from badgers.core.causal_graph import (
    get_descendants,
    get_parents,
    node_to_index,
    topological_order,
    validate_dag,
)
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.outliers import OutliersGenerator


class CausalOutlierPropagationGenerator(OutliersGenerator):
    """
    Injects an outlier into a target node and propagates the perturbation
    to all descendants along the causal graph using linear approximation.

    The generator fits per-node linear regressions from the observed data,
    injects a perturbation at the target node, and propagates the delta
    to all descendants in topological order.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.

    generate() parameters (passed via **params)
    ----------
    graph : nx.DiGraph
        Causal DAG. Nodes must be int (column indices) or str (mapped to
        sorted column order).
    target_node : int or str
        Node to inject the outlier into.
    outlier_magnitude : float, default=3.0
        Number of standard deviations to perturb by.
    sample_index : int, default=0
        Row index of the sample to turn into an outlier.
        Must be in [0, n_samples). Raises IndexError otherwise.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y, graph=None, target_node=None,
                 outlier_magnitude=3.0, sample_index=0):
        """
        Generate an outlier by perturbing target_node and propagating
        the perturbation to all descendants.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : np.ndarray or None
            Target values (returned unchanged).
        graph : nx.DiGraph
            Causal DAG.
        target_node : int or str
            Node to inject the outlier into.
        outlier_magnitude : float, default=3.0
            Number of standard deviations to perturb by.
        sample_index : int, default=0
            Row index to perturb.

        Returns
        -------
        Xt : np.ndarray
            Data with propagated outlier.
        yt : np.ndarray or None
            Unchanged target values.
        """
        if graph is None:
            raise ValueError("graph parameter is required")

        validate_dag(graph)

        if target_node is None:
            raise ValueError("target_node parameter is required")

        if target_node not in graph.nodes:
            raise ValueError(
                f"target_node '{target_node}' not found in graph"
            )

        n_samples, n_features = X.shape
        if n_features != len(graph.nodes):
            raise ValueError(
                f"X has {n_features} columns but graph has {len(graph.nodes)} nodes"
            )

        if sample_index < 0 or sample_index >= n_samples:
            raise IndexError(
                f"sample_index {sample_index} out of bounds for {n_samples} samples"
            )

        # Compute topological order
        order = topological_order(graph)

        # Build node -> column index mapping
        node_to_col = {n: node_to_index(graph, n) for n in graph.nodes}

        # Fit linear coefficients: for each node v, regress X[:,v] on X[:,parents(v)]
        # Store coefficients as dict: parent_node -> coefficient
        coefficients = {}
        for node in order:
            parents = get_parents(graph, node)
            if not parents:
                continue
            parent_cols = [node_to_col[p] for p in parents]
            X_parents = X[:, parent_cols]
            X_target = X[:, node_to_col[node]]
            # Solve least squares: X_target ~ X_parents @ beta
            beta, _, _, _ = np.linalg.lstsq(X_parents, X_target, rcond=None)
            for i, parent in enumerate(parents):
                coefficients[(parent, node)] = beta[i]

        # Copy X to avoid modifying input
        Xt = X.copy()

        # Compute perturbation at target node
        target_col = node_to_col[target_node]
        target_std = np.std(X[:, target_col])
        if target_std == 0:
            target_std = 1.0  # fallback for constant columns
        delta_target = outlier_magnitude * target_std

        # Inject outlier at target node
        Xt[sample_index, target_col] = X[sample_index, target_col] + delta_target

        # Track deltas for each node (for propagation)
        deltas = {target_node: delta_target}

        # Propagate to descendants in topological order
        descendants = get_descendants(graph, target_node)
        for node in order:
            if node not in descendants:
                continue
            parents = get_parents(graph, node)
            # Compute propagated delta: sum over parents of (coef * parent_delta)
            delta = 0.0
            for parent in parents:
                if parent in deltas:
                    coef = coefficients.get((parent, node), 0.0)
                    delta += coef * deltas[parent]
            if delta != 0.0:
                col = node_to_col[node]
                Xt[sample_index, col] = X[sample_index, col] + delta
                deltas[node] = delta

        return Xt, y
