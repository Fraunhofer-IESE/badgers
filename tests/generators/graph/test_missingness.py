import networkx as nx
import numpy as np

from badgers.generators.graph.missingness import NodesMissingCompletelyAtRandom, EdgesMissingCompletelyAtRandom


def test_nodes_missing_completely_at_random__removes_nodes(graph_erdos_renyi):
    """NodesMissingCompletelyAtRandom removes correct number of nodes."""
    G = graph_erdos_renyi
    rng = np.random.default_rng(0)
    percentage_missing = 0.1
    generator = NodesMissingCompletelyAtRandom(random_generator=rng)
    Xt = generator.generate(X=G, percentage_missing=percentage_missing)
    assert len(Xt) == len(G) - 10


def test_nodes_missing_completely_at_random__removes_node_attributes(graph_erdos_renyi):
    """Node attributes (e.g. labels) vanish with their node; surviving nodes keep their values."""
    G = graph_erdos_renyi
    nx.set_node_attributes(G, {n: f'label_{n}' for n in G.nodes()}, name='label')
    rng = np.random.default_rng(0)
    generator = NodesMissingCompletelyAtRandom(random_generator=rng)
    Xt = generator.generate(G, percentage_missing=0.1)

    assert len(Xt) == len(G) - 10
    assert all('label' in data for _, data in Xt.nodes(data=True))
    for n, data in Xt.nodes(data=True):
        assert data['label'] == f'label_{n}'
    # every removed node's attribute is gone with it
    assert Xt.number_of_nodes() == sum(1 for _, data in Xt.nodes(data=True) if 'label' in data)


def test_edges_missing_completely_at_random__removes_edges(graph_erdos_renyi):
    """EdgesMissingCompletelyAtRandom removes correct number of edges."""
    G = graph_erdos_renyi
    rng = np.random.default_rng(0)
    percentage_missing = 0.1
    generator = EdgesMissingCompletelyAtRandom(random_generator=rng)
    Xt = generator.generate(X=G, percentage_missing=percentage_missing)
    assert Xt.number_of_nodes() == G.number_of_nodes()
    assert Xt.number_of_edges() == len(G.edges()) - int(G.number_of_edges() * percentage_missing)


def test_edges_missing_completely_at_random__removes_edge_attributes(graph_erdos_renyi):
    """Edge attributes vanish with their edge; surviving edges keep their values."""
    G = graph_erdos_renyi
    nx.set_edge_attributes(G, {(u, v): f'weight_{u}_{v}' for u, v in G.edges()}, name='weight')
    rng = np.random.default_rng(0)
    generator = EdgesMissingCompletelyAtRandom(random_generator=rng)
    Xt = generator.generate(G, percentage_missing=0.1)

    assert Xt.number_of_edges() == len(G.edges()) - int(G.number_of_edges() * 0.1)
    assert all('weight' in data for _, _, data in Xt.edges(data=True))
    for u, v, data in Xt.edges(data=True):
        assert data['weight'] == f'weight_{u}_{v}'
    assert Xt.number_of_edges() == sum(1 for _, _, data in Xt.edges(data=True) if 'weight' in data)


def test_generate_returns_single_graph_object(graph_erdos_renyi):
    """generate returns the modified graph only (single object contract, not a tuple)."""
    G = graph_erdos_renyi
    rng = np.random.default_rng(0)
    nodes_generator = NodesMissingCompletelyAtRandom(random_generator=rng)
    edges_generator = EdgesMissingCompletelyAtRandom(random_generator=rng)
    assert isinstance(nodes_generator.generate(G, percentage_missing=0.1), nx.Graph)
    assert isinstance(edges_generator.generate(G, percentage_missing=0.1), nx.Graph)
