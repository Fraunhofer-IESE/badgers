import abc

import networkx as nx
import numpy
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin


class MissingGenerator(GeneratorMixin):
    """
    Base class for missing nodes transformer
    """

    def __init__(self, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """

        :param random_generator: A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        This method should be overridden by subclasses.
        """
        pass


class NodesMissingCompletelyAtRandom(MissingGenerator):
    """
    Removes nodes from the graph uniformly at random.
    """

    def __init__(self, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """
        Initialize the missingness generator.

        :param random_generator: A NumPy random number generator.
                               Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    def generate(self, X, percentage_missing: float = 0.1):
        """
        Generate a graph with a specified percentage of missing nodes.

        Any values associated with the nodes (e.g., node attributes such as labels)
        are stored as attributes on the graph nodes themselves. When a node is removed,
        its associated values are removed with it.

        :param X: The input graph from which nodes will be removed.
        :type X: nx.Graph
        :param percentage_missing: The percentage of nodes to be removed (float value between 0 and 1).
        :type percentage_missing: float
        :return: The modified graph with missing nodes.
        :rtype: nx.Graph
        """
        assert 0 < percentage_missing < 1
        if not isinstance(X, nx.Graph):
            raise NotImplementedError('badgers does only support networkx.Graph objects for graphs')

        nodes_to_be_removed = self.random_generator.choice(
            X.nodes(),
            int(X.number_of_nodes() * percentage_missing),
            replace=False
        )

        Xt = X.copy()
        Xt.remove_nodes_from(nodes_to_be_removed)

        return Xt


class EdgesMissingCompletelyAtRandom(MissingGenerator):
    """
    Removes edges from the graph uniformly at random.
    """

    def __init__(self, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """
        Initialize the missingness generator.

        :param random_generator: A NumPy random number generator.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    def generate(self, X, percentage_missing: float = 0.1):
        """
        Generate a graph with a specified percentage of missing edges.

        Any values associated with the edges (e.g., edge attributes such as weights)
        are stored as attributes on the graph edges themselves. When an edge is removed,
        its associated values are removed with it.

        :param X: The input graph from which edges will be removed.
        :type X: nx.Graph
        :param percentage_missing: The percentage of edges to be removed (float value between 0 and 1).
        :type percentage_missing: float
        :return: The modified graph with missing edges.
        :rtype: nx.Graph
        """
        assert 0 < percentage_missing < 1
        if not isinstance(X, nx.Graph):
            raise NotImplementedError('badgers does only support networkx.Graph objects for graphs')

        edges_to_be_removed = self.random_generator.choice(
            X.edges(),
            int(X.number_of_edges() * percentage_missing),
            replace=False
        )

        Xt = X.copy()
        Xt.remove_edges_from(edges_to_be_removed)

        return Xt
