import networkx as nx
import pytest


@pytest.fixture
def graph_erdos_renyi():
    """100-node Erdős-Rényi graph as (G, None)."""
    G = nx.erdos_renyi_graph(n=100, p=0.25, seed=0, directed=False)
    return G, None
