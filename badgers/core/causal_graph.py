"""Utility functions for working with causal DAGs represented as networkx DiGraphs."""

import networkx as nx


def validate_dag(graph: nx.DiGraph) -> None:
    """
    Validate that the input is a directed acyclic graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to validate.

    Raises
    ------
    ValueError
        If graph is not a nx.DiGraph or contains cycles.
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(
            f"graph must be a nx.DiGraph, got {type(graph).__name__}"
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("graph is not a directed acyclic graph (contains cycles)")


def topological_order(graph: nx.DiGraph) -> list:
    """
    Return nodes in topological order.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed acyclic graph.

    Returns
    -------
    list
        Nodes in topological order.
    """
    return list(nx.topological_sort(graph))


def get_parents(graph: nx.DiGraph, node) -> list:
    """
    Return the immediate predecessors (parents) of a node.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph.
    node : hashable
        The node whose parents to return.

    Returns
    -------
    list
        Parent nodes (in arbitrary order).
    """
    return list(graph.predecessors(node))


def get_descendants(graph: nx.DiGraph, node) -> set:
    """
    Return all nodes reachable from the given node (its descendants).

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph.
    node : hashable
        The node whose descendants to return.

    Returns
    -------
    set
        All descendant nodes.
    """
    return nx.descendants(graph, node)


def get_ancestors(graph: nx.DiGraph, node) -> set:
    """
    Return all nodes that can reach the given node (its ancestors).

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph.
    node : hashable
        The node whose ancestors to return.

    Returns
    -------
    set
        All ancestor nodes.
    """
    return nx.ancestors(graph, node)


def get_root_nodes(graph: nx.DiGraph) -> list:
    """
    Return nodes with no incoming edges (in-degree 0).

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph.

    Returns
    -------
    list
        Root nodes.
    """
    return [n for n in graph.nodes if graph.in_degree(n) == 0]


def get_leaf_nodes(graph: nx.DiGraph) -> list:
    """
    Return nodes with no outgoing edges (out-degree 0).

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph.

    Returns
    -------
    list
        Leaf nodes.
    """
    return [n for n in graph.nodes if graph.out_degree(n) == 0]


def node_to_index(graph: nx.DiGraph, node) -> int:
    """
    Map a graph node to a column index.

    Integer nodes map directly to themselves.
    String nodes are sorted alphabetically and mapped to 0..d-1.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph.
    node : int or str
        The node to map.

    Returns
    -------
    int
        Column index for this node.

    Raises
    ------
    ValueError
        If the node is not in the graph.
    """
    if node not in graph.nodes:
        raise ValueError(f"node '{node}' not found in graph")

    if isinstance(node, int):
        return node

    # String nodes: sort alphabetically and map to indices
    sorted_nodes = sorted([n for n in graph.nodes if isinstance(n, str)])
    return sorted_nodes.index(node)
