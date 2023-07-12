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

    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """

        :param percentage_missing: The percentage of missing nodes (int value between 0 and 100 included)
        :param random_generator: A random generator
        """
        assert 0 <= percentage_missing <= 100
        self.percentage_missing = percentage_missing
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y=None, **params) -> Tuple:
        pass


class NodesMissingCompletelyAtRandom(MissingGenerator):
    """
    Removes nodes from the graph uniformly at random.
    """

    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    def generate(self, X, y=None, **params) -> Tuple:
        if not isinstance(X, nx.Graph):
            raise NotImplementedError('badgers does only support networkx.Graph objects for graphs')

        nodes_to_be_removed = self.random_generator.choice(
            X.nodes(),
            int(X.number_of_nodes() * self.percentage_missing / 100),
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

    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    def generate(self, X, y=None, **params) -> Tuple:
        if not isinstance(X, nx.Graph):
            raise NotImplementedError('badgers does only support networkx.Graph objects for graphs')

        edges_to_be_removed = self.random_generator.choice(
            X.edges(),
            int(X.number_of_edges() * self.percentage_missing / 100),
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
