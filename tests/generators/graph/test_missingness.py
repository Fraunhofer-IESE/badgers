import networkx as nx
import numpy as np

from badgers.generators.graph.missingness import NodesMissingCompletelyAtRandom, EdgesMissingCompletelyAtRandom


def test_nodes_missing_completely_at_random__removes_nodes(graph_erdos_renyi):
    """NodesMissingCompletelyAtRandom removes correct number of nodes."""
    G, _ = graph_erdos_renyi
    rng = np.random.default_rng(0)
    percentage_missing = 0.1
    generator = NodesMissingCompletelyAtRandom(random_generator=rng)
    Xt, _ = generator.generate(X=G, y=None, percentage_missing=percentage_missing)
    assert len(Xt) == len(G) - 10


def test_nodes_missing_completely_at_random__removes_nodes_and_labels(graph_erdos_renyi):
    """NodesMissingCompletelyAtRandom removes nodes and corresponding labels."""
    G, _ = graph_erdos_renyi
    rng = np.random.default_rng(0)
    generator = NodesMissingCompletelyAtRandom(random_generator=rng)
    Xt, yt = generator.generate(G, [0] * len(G))
    assert len(Xt) == len(G) - 10
    assert len(yt) == len(G) - 10


def test_edges_missing_completely_at_random__removes_edges(graph_erdos_renyi):
    """EdgesMissingCompletelyAtRandom removes correct number of edges."""
    G, _ = graph_erdos_renyi
    rng = np.random.default_rng(0)
    percentage_missing = 0.1
    generator = EdgesMissingCompletelyAtRandom(random_generator=rng)
    Xt, _ = generator.generate(X=G, y=None, percentage_missing=percentage_missing)
    assert Xt.number_of_nodes() == G.number_of_nodes()
    assert Xt.number_of_edges() == len(G.edges()) - int(G.number_of_edges() * percentage_missing)