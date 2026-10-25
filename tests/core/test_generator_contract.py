"""Cross-modality contract tests for the X-only generator API.

One representative built-in generator per implemented modality is checked
against the two core invariants of the migrated contract:

1. ``generate(X, **params)`` returns a SINGLE generated object (no tuple),
   of the modality-native type.
2. The removed positional ``y`` argument is rejected (TypeError).
"""

import networkx as nx
import numpy as np
import pytest

from badgers.generators.graph.missingness import NodesMissingCompletelyAtRandom
from badgers.generators.tabular_data.drift.drift import RandomShiftGenerator
from badgers.generators.text.typos import SwapLettersGenerator
from badgers.generators.time_series.trends import (
    GlobalAdditiveLinearTrendGenerator as TsSlopeGenerator,
)


def _small_graph(rng=None):
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return g


# (modality, generator, X factory, minimal params, expected Xt type)
REPRESENTATIVES = [
    pytest.param(
        "tabular",
        lambda: RandomShiftGenerator(random_generator=np.random.default_rng(0)),
        lambda: np.random.default_rng(0).normal(size=(100, 3)),
        {"shift_std": 0.1},
        np.ndarray,
        id="tabular",
    ),
    pytest.param(
        "time_series",
        lambda: TsSlopeGenerator(random_generator=np.random.default_rng(0)),
        lambda: np.random.default_rng(0).normal(size=(100, 1)),
        {"slope": 0.1},
        np.ndarray,
        id="time_series",
    ),
    pytest.param(
        "text",
        lambda: SwapLettersGenerator(random_generator=np.random.default_rng(0)),
        lambda: ["kilogram", "fluent", "python"],
        {"swap_proba": 1.0},
        list,
        id="text",
    ),
    pytest.param(
        "graph",
        lambda: NodesMissingCompletelyAtRandom(
            random_generator=np.random.default_rng(0)
        ),
        _small_graph,
        {"percentage_missing": 0.1},
        nx.Graph,
        id="graph",
    ),
]


@pytest.mark.parametrize(
    "modality,make_generator,make_X,params,expected_type", REPRESENTATIVES
)
def test_generate__single_value_return_of_native_type(
    modality, make_generator, make_X, params, expected_type
):
    X = make_X()
    Xt = make_generator().generate(X, **params)

    assert not isinstance(Xt, tuple)
    assert isinstance(Xt, expected_type)


@pytest.mark.parametrize(
    "modality,make_generator,make_X,params,expected_type", REPRESENTATIVES
)
def test_generate__rejects_removed_positional_y(
    modality, make_generator, make_X, params, expected_type
):
    X = make_X()
    y_placeholder = np.zeros(len(X) if hasattr(X, "__len__") else X.number_of_nodes())

    with pytest.raises(TypeError):
        make_generator().generate(X, y_placeholder, **params)
