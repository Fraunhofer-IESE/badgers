import unittest
from unittest import TestCase

import networkx as nx
import numpy as np

from badgers.generators.graph.missingness import NodesMissingCompletelyAtRandom, EdgesMissingCompletelyAtRandom


class TestNodesMissingCompletelyAtRandom(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.graph = nx.erdos_renyi_graph(n=100, p=0.25, seed=0, directed=False)

    def test_generate(self):
        percentage_missing = 0.1
        generator = NodesMissingCompletelyAtRandom(random_generator=self.rng)

        Xt, _ = generator.generate(X=self.graph, y=None, percentage_missing=percentage_missing)
        self.assertEqual(len(Xt), len(self.graph) - 10)

        Xt, yt = generator.generate(self.graph, [0] * len(self.graph))
        self.assertEqual(len(Xt), len(self.graph) - 10)
        self.assertEqual(len(yt), len(self.graph) - 10)


class TestEdgesMissingCompletelyAtRandom(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.graph = nx.erdos_renyi_graph(n=100, p=0.25, seed=0, directed=False)

    def test_generate(self):
        percentage_missing = 0.1
        generator = EdgesMissingCompletelyAtRandom(random_generator=self.rng)

        Xt, _ = generator.generate(X=self.graph, y=None, percentage_missing=percentage_missing)
        self.assertEqual(Xt.number_of_nodes(), self.graph.number_of_nodes())
        self.assertEqual(Xt.number_of_edges(),
                         len(self.graph.edges()) - int(self.graph.number_of_edges() * percentage_missing))

if __name__ == '__main__':
    unittest.main()