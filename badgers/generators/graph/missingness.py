import abc
from copy import copy
from typing import Tuple

import networkx as nx
import numpy
import numpy as np
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
    def generate(self, X, y=None, **params) -> Tuple:
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

    def generate(self, X, y=None, percentage_missing: float = 0.1) -> Tuple:
        """
        Generate a graph with a specified percentage of missing nodes.

        :param X: The input graph from which nodes will be removed.
        :type X: nx.Graph
        :param y: Optional target array associated with the nodes in the graph.
                  If provided, the corresponding elements will also be removed.
        :type y: np.ndarray, optional
        :param percentage_missing: The percentage of nodes to be removed (float value between 0 and 1).
        :type percentage_missing: float
        :return: A tuple containing the modified graph with missing nodes and the modified target array (if provided).
        :rtype: Tuple[nx.Graph, Optional[np.ndarray]]
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

        if y is not None:
            yt = np.delete(y, nodes_to_be_removed)
        else:
            yt = None

        return Xt, yt


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

    def generate(self, X, y=None, percentage_missing: float = 0.1) -> Tuple:
        """
        Generate a graph with a specified percentage of missing edges.

        :param X: The input graph from which edges will be removed.
        :type X: nx.Graph
        :param y: Optional target data associated with the edges in the graph.
                  If provided, the corresponding elements will also be removed.
                  Can be a dictionary where keys are edge tuples and values are target values.
        :type y: dict, optional
        :param percentage_missing: The percentage of edges to be removed (float value between 0 and 1).
        :type percentage_missing: float
        :return: A tuple containing the modified graph with missing edges and the modified target data (if provided).
        :rtype: Tuple[nx.Graph, Optional[dict]]
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

        if y is None:
            yt = None
        elif isinstance(y, dict):
            yt = copy(y)
            for e in edges_to_be_removed:
                del yt[e]
        else:
            raise ValueError(f'This type of y is not supported {type(y)}, {y}')

        return Xt, yt
