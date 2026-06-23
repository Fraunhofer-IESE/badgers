"""Predefined scenarios for graph generators."""
import networkx as nx
from benchmarks.models import Scenario


def _erdos_renyi_factory(rng):
    G = nx.erdos_renyi_graph(n=100, p=0.1, seed=42)
    return G, None


SCENARIO_ERDOS_RENYI = Scenario(
    name="erdos_renyi_100",
    data_type="graph",
    factory=_erdos_renyi_factory,
    tags=["small", "random"],
)