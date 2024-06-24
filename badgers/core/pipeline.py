from typing import Dict


class Pipeline:
    """
    Class for chaining different generators together.

    Not compatible with scikit-learn Pipelines
    Only words for python 3.7 as it relies upon dictionaries, and they need to be ordered
    """

    def __init__(self, generators: Dict):
        """
        Creates a Pipeline consisting of different generators
        :param generators: a dictionary containing as keys the generator names and as values the generators instances
        """
        self.generators = generators

    def generate(self, X, y, params: Dict = None):
        """
        Calls all generators generate function in the order

        :param X: the input features
        :param y: the class labels, regression targets, or None
        :param params: a dictionary containing as key: the generator names and
        as values: the parameters for each corresponding generate function
        :return: Xt, yt
        """
        if params is None:
            params = dict()
        for name, generator in self.generators.items():
            X, y = generator.generate(X, y, **params.get(name, dict()))
        return X, y
