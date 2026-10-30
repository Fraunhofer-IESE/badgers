"""Shared helpers for the tabular-data tutorials.

Imported by the notebooks in this directory via ``from _shared import ...``.
Kept as a plain module (leading underscore) so it stays out of the docs build.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_linear_data(graph, n_samples=100, rng=None, coefficients=None):
    """Generate data from a linear structural equation model on a DAG.

    Each edge (u -> v) carries a coefficient w. Every node value is
        v = sum_{u in parents(v)} w_{u,v} * u + noise
    with noise ~ N(0, 1).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if coefficients is None:
        coefficients = {edge: 1.0 for edge in graph.edges()}

    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    X = np.zeros((n_samples, len(nodes)))
    for node in nx.topological_sort(graph):
        idx = node_to_idx[node]
        parents = list(graph.predecessors(node))
        if not parents:
            X[:, idx] = rng.normal(size=n_samples)
        else:
            X[:, idx] = rng.normal(size=n_samples)
            for parent in parents:
                coef = coefficients.get((parent, node), 1.0)
                X[:, idx] += coef * X[:, node_to_idx[parent]]
    return X, {node: i for i, node in enumerate(nodes)}


def draw_graph(graph, title=None):
    """Draw a DAG with fixed left-to-right node positions.

    Nodes are placed in topological generations: roots at x = 0, their
    descendants at x = 1, and so on; siblings are stacked vertically.
    This keeps the visual layout deterministic and consistent across
    all notebooks (no force-directed layout).
    """
    generations = {}  # node -> column index
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        generations[node] = 0 if not parents else max(generations[p] for p in parents) + 1
    columns = {}
    pos = {}
    for node, col in generations.items():
        pos[node] = (col, columns.setdefault(col, 0))
        columns[col] += 1
    # center each column vertically
    height = max(columns.values())
    pos = {node: (x, y - (columns[x] - 1) / 2) for node, (x, y) in pos.items()}
    fig, ax = plt.subplots(1, figsize=(6, 2))
    nx.draw(graph, pos, ax=ax, node_color="lightblue", node_size=800, arrowsize=15)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig


def plot_2panel(X, Xt, x_idx=0, y_idx=1, x_name="X", y_name="Y"):
    """Original | Transformed scatter of two dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    axes[0].scatter(X[:, x_idx], X[:, y_idx], color="C0", s=15)
    axes[1].scatter(Xt[:, x_idx], Xt[:, y_idx], color="C1", s=15)
    axes[0].set_title("Original")
    axes[1].set_title("Transformed")
    axes[0].set_xlabel(x_name)
    axes[1].set_xlabel(x_name)
    axes[0].set_ylabel(y_name)
    axes[1].set_ylabel(y_name)
    plt.tight_layout()
    return fig


def plot_class_counts(y, yt):
    """Bar charts of class counts, original | transformed."""
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    classes, nb = np.unique(y, return_counts=True)
    axes[0].bar(classes, nb, color=[f"C{i}" for i in range(len(classes))])
    axes[0].set_xlabel("classes")
    axes[0].set_ylabel("number of instances")
    classes_t, nb_t = np.unique(yt, return_counts=True)
    axes[1].bar(classes_t, nb_t, color=[f"C{i}" for i in range(len(classes_t))])
    axes[1].set_xlabel("classes")
    axes[1].set_ylabel("number of instances")
    axes[0].set_title("Original")
    axes[1].set_title("Transformed")
    plt.tight_layout()
    return fig, axes


def plot_missing(X, cls, Xt):
    """Scatter plots of the original and the transformed data (missing rows marked)."""
    missing_mask = np.isnan(Xt).any(axis=1)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    for label in np.unique(cls):
        ix = np.where(cls == label)
        axes[0].scatter(X[ix, 0], X[ix, 1], c=f"C{label}", label=f"{label}")
        ix = np.where(cls[~missing_mask] == label)
        axes[1].scatter(Xt[~missing_mask][ix, 0], Xt[~missing_mask][ix, 1], c=f"C{label}", label=f"{label}")
    # plot missing values
    axes[1].scatter(X[missing_mask][:, 0], X[missing_mask][:, 1], marker="x", color="black", label="missing")
    axes[0].set_title("Original")
    axes[1].set_title("Transformed")
    axes[0].legend()
    axes[1].legend()
    return fig, axes


def plot_noise(X, cls, Xt):
    """Scatter plots of the original and the transformed data."""
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    axes[0].scatter(X[:, 0], X[:, 1], c=cls, cmap="tab10", s=15)
    axes[1].scatter(Xt[:, 0], Xt[:, 1], c=cls, cmap="tab10", s=15)
    axes[0].set_title("Original")
    axes[1].set_title("Transformed")
    return fig, axes
