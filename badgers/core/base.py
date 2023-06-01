import abc
from typing import Tuple


class GeneratorMixin:

    @abc.abstractmethod
    def generate(self, X, y=None, **params) -> Tuple:
        """

        :param X: the input
        :param y: the target
        :param params: optional parameters
        :return: Xt, yt
        """
        pass