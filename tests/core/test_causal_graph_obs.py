"""Tests for the observation-graph structure helpers (pure networkx)."""

import networkx as nx
import pytest

from badgers.core.causal_graph import (
    indicator_nodes,
    validate_observation_graph,
)


def _chain_structural():
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "Z")])
    return g


def _obs_graph(edges, nodes):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def test_indicator_nodes__mcar_isolated_indicator():
    obs = _obs_graph(
        [("X", "Y"), ("Y", "Z")], nodes=["X", "Y", "Z", "R_X"]
    )
    assert indicator_nodes(obs) == [("R_X", "X")]


def test_indicator_nodes__mar_driver_is_parent():
    obs = _obs_graph(
        [("Y", "R_X"), ("X", "Y"), ("Y", "Z")],
        nodes=["X", "Y", "Z", "R_X"],
    )
    # topological order: R_X comes after its driver Y
    assert indicator_nodes(obs) == [("R_X", "X")]


def test_indicator_nodes__monitors_attribute_overrides_convention():
    obs = _obs_graph([("Y", "R_odd")], nodes=["X", "Y", "R_odd"])
    obs.nodes["R_odd"]["monitors"] = "X"
    assert indicator_nodes(obs) == [("R_odd", "X")]


def test_indicator_nodes__unresolvable_indicator_raises():
    obs = _obs_graph([("Y", "R_nope")], nodes=["X", "Y", "R_nope"])
    with pytest.raises(ValueError, match="does not resolve"):
        indicator_nodes(obs)


def test_validate_observation_graph__accepts_valid_mirroring():
    obs = _obs_graph(
        [("X", "Y"), ("Y", "Z"), ("X", "R_X")],
        nodes=["X", "Y", "Z", "R_X"],
    )
    assert validate_observation_graph(obs, _chain_structural()) is None


def test_validate_observation_graph__rejects_cycle():
    obs = _obs_graph([("X", "Y"), ("Y", "X")], nodes=["X", "Y"])
    with pytest.raises(ValueError, match="directed acyclic"):
        validate_observation_graph(obs, _chain_structural())


def test_validate_observation_graph__rejects_unknown_node():
    obs = _obs_graph(
        [("X", "Y"), ("Y", "Z"), ("X", "R_X")],
        nodes=["X", "Y", "Z", "R_X", "PHANTOM"],
    )
    with pytest.raises(ValueError, match="not in the structural graph"):
        validate_observation_graph(obs, _chain_structural())


def test_validate_observation_graph__rejects_non_leaf_indicator():
    obs = _obs_graph(
        [("X", "Y"), ("Y", "Z"), ("X", "R_X"), ("R_X", "Z")],
        nodes=["X", "Y", "Z", "R_X"],
    )
    with pytest.raises(ValueError, match="must be a leaf"):
        validate_observation_graph(obs, _chain_structural())


def test_validate_observation_graph__derivation_edges_must_mirror_structure():
    # v1: derivation wiring (measurement -> measurement) must equal the
    # structural edge set, so fitted mechanisms stay consistent.
    obs = _obs_graph(
        [("X", "Z"), ("X", "R_X")], nodes=["X", "Y", "Z", "R_X"]
    )
    with pytest.raises(ValueError, match="must mirror the structural graph"):
        validate_observation_graph(obs, _chain_structural())
