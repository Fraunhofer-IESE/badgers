import networkx as nx
import pytest

from badgers.core.causal_graph import (
    validate_dag,
    topological_order,
    get_parents,
    get_descendants,
    get_ancestors,
    get_root_nodes,
    get_leaf_nodes,
    node_to_index,
)


# --- Fixtures ---

@pytest.fixture
def chain_graph():
    """X -> Y -> Z"""
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "Z")])
    return g


@pytest.fixture
def fork_graph():
    """C -> X, C -> Y"""
    g = nx.DiGraph()
    g.add_edges_from([("C", "X"), ("C", "Y")])
    return g


@pytest.fixture
def collider_graph():
    """X -> Z <- Y"""
    g = nx.DiGraph()
    g.add_edges_from([("X", "Z"), ("Y", "Z")])
    return g


@pytest.fixture
def diamond_graph():
    """A -> B, A -> C, B -> D, C -> D"""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return g


@pytest.fixture
def cyclic_graph():
    """X -> Y -> Z -> X"""
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "Z"), ("Z", "X")])
    return g


@pytest.fixture
def int_node_graph():
    """0 -> 1 -> 2"""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2)])
    return g


# --- validate_dag ---

def test_validate_dag__valid_dag(chain_graph):
    """Should not raise for a valid DAG."""
    validate_dag(chain_graph)


def test_validate_dag__cycle_raises(cyclic_graph):
    """Should raise ValueError for a cyclic graph."""
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        validate_dag(cyclic_graph)


def test_validate_dag__undirected_raises():
    """Should raise ValueError for an undirected graph."""
    g = nx.Graph()
    g.add_edge("A", "B")
    with pytest.raises(ValueError, match="must be a nx.DiGraph"):
        validate_dag(g)


# --- topological_order ---

def test_topological_order__chain(chain_graph):
    """Should return nodes in topological order for a chain."""
    order = topological_order(chain_graph)
    assert order.index("X") < order.index("Y") < order.index("Z")


def test_topological_order__diamond(diamond_graph):
    """Should return a valid topological order for a diamond graph."""
    order = topological_order(diamond_graph)
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


# --- get_parents ---

def test_get_parents__chain(chain_graph):
    """Y should have parent X in X->Y->Z."""
    assert get_parents(chain_graph, "Y") == ["X"]


def test_get_parents__root_node(chain_graph):
    """X (root) should have no parents."""
    assert get_parents(chain_graph, "X") == []


def test_get_parents__collider(collider_graph):
    """Z should have parents X and Y in X->Z<-Y."""
    parents = get_parents(collider_graph, "Z")
    assert set(parents) == {"X", "Y"}


# --- get_descendants ---

def test_get_descendants__chain(chain_graph):
    """X should have descendants Y and Z in X->Y->Z."""
    assert get_descendants(chain_graph, "X") == {"Y", "Z"}


def test_get_descendants__leaf_node(chain_graph):
    """Z (leaf) should have no descendants."""
    assert get_descendants(chain_graph, "Z") == set()


def test_get_descendants__fork(fork_graph):
    """C should have descendants X and Y in C->X, C->Y."""
    assert get_descendants(fork_graph, "C") == {"X", "Y"}


# --- get_ancestors ---

def test_get_ancestors__chain(chain_graph):
    """Z should have ancestors X and Y in X->Y->Z."""
    assert get_ancestors(chain_graph, "Z") == {"X", "Y"}


def test_get_ancestors__root_node(chain_graph):
    """X (root) should have no ancestors."""
    assert get_ancestors(chain_graph, "X") == set()


# --- get_root_nodes ---

def test_get_root_nodes__chain(chain_graph):
    """Only X should be a root node in X->Y->Z."""
    assert get_root_nodes(chain_graph) == ["X"]


def test_get_root_nodes__collider(collider_graph):
    """X and Y should be root nodes in X->Z<-Y."""
    assert set(get_root_nodes(collider_graph)) == {"X", "Y"}


# --- get_leaf_nodes ---

def test_get_leaf_nodes__chain(chain_graph):
    """Only Z should be a leaf node in X->Y->Z."""
    assert get_leaf_nodes(chain_graph) == ["Z"]


def test_get_leaf_nodes__fork(fork_graph):
    """X and Y should be leaf nodes in C->X, C->Y."""
    assert set(get_leaf_nodes(fork_graph)) == {"X", "Y"}


# --- node_to_index ---

def test_node_to_index__int_nodes(int_node_graph):
    """Integer nodes should map directly to themselves."""
    assert node_to_index(int_node_graph, 0) == 0
    assert node_to_index(int_node_graph, 1) == 1
    assert node_to_index(int_node_graph, 2) == 2


def test_node_to_index__str_nodes(chain_graph):
    """String nodes should map to sorted order indices."""
    assert node_to_index(chain_graph, "X") == 0
    assert node_to_index(chain_graph, "Y") == 1
    assert node_to_index(chain_graph, "Z") == 2
