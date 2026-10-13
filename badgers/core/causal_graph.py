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


def indicator_nodes(observation_graph: nx.DiGraph) -> list:
    """
    Return the indicator nodes of an observation graph with their monitored
    variables, in topological order.

    An indicator is a node named ``R_<variable>``. It monitors the variable
    given by its optional ``monitors`` attribute, or ``<variable>`` by the
    naming convention. The monitored variable must appear in the graph
    (measurement nodes share the structural node names).

    Parameters
    ----------
    observation_graph : nx.DiGraph
        The observation-process DAG (must be acyclic).

    Returns
    -------
    list of (indicator, monitored) tuples
        In topological order.

    Raises
    ------
    ValueError
        If an indicator name does not resolve to a monitored variable.
    networkx.NetworkXUnfeasible
        If the observation graph has a cycle (``topological_order`` raises;
        prefer ``validate_observation_graph`` first for a friendlier error).
    """
    result = []
    for node in topological_order(observation_graph):
        if not str(node).startswith("R_"):
            continue
        monitored = observation_graph.nodes[node].get("monitors")
        if monitored is None:
            monitored = str(node)[2:]
        if monitored not in observation_graph.nodes:
            raise ValueError(
                f"indicator '{node}' does not resolve to a monitored "
                f"variable: set a 'monitors' attribute or add node "
                f"'{monitored}'"
            )
        result.append((node, monitored))
    return result


def validate_observation_graph(observation_graph: nx.DiGraph,
                               structural_graph: nx.DiGraph) -> None:
    """
    Validate an observation graph against the structural graph.

    Checks:

    - the observation graph is a DAG;
    - every measurement node exists in the structural graph (measurement
      nodes share the structural node names; indicator nodes ``R_*`` are
      observation-process nodes and are exempt);
    - indicator nodes (``R_*``) are leaves (no outgoing edges);
    - every ``derivation_rule`` attribute (measurement nodes) is
      ``"any"`` or ``"all"``;
    - derivation edges (measurement -> measurement) mirror the structural
      graph, so the fitted mechanisms stay consistent (v1 contract; the
      artifact-column extension relaxes this in a later round).

    Raises
    ------
    ValueError
        On cycles, unknown measurement nodes, unresolvable indicators,
        non-leaf indicators, non-mirroring derivation edges, or an
        unknown derivation_rule value.
    """
    validate_dag(observation_graph)
    structural = set(structural_graph.nodes)
    unknown = {
        node for node in observation_graph.nodes
        if not str(node).startswith("R_") and node not in structural
    }
    if unknown:
        raise ValueError(
            f"observation graph has node(s) not in the structural graph: "
            f"{sorted(unknown)}"
        )
    for node, _ in indicator_nodes(observation_graph):
        if observation_graph.out_degree(node) > 0:
            raise ValueError(
                f"indicator '{node}' must be a leaf (no outgoing edges)"
            )
    for node in observation_graph.nodes:
        if str(node).startswith("R_"):
            continue
        rule = observation_graph.nodes[node].get("derivation_rule", "any")
        if rule not in ("any", "all"):
            raise ValueError(
                f"node '{node}' has an unknown derivation_rule '{rule}' "
                f"(expected 'any' or 'all')"
            )
    derivations = {
        (s, t) for s, t in observation_graph.edges
        if not str(s).startswith("R_") and not str(t).startswith("R_")
    }
    if derivations != set(structural_graph.edges):
        raise ValueError(
            "derivation edges must mirror the structural graph (v1 "
            "contract): got "
            f"{sorted(derivations)}, expected "
            f"{sorted(structural_graph.edges)}"
        )



