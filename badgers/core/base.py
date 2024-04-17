import abc
from typing import Tuple


class GeneratorMixin:

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        """

        :param X: the input
        :param y: the target
        :param params: optional parameters
        :return: Xt, yt
        """
        pass

    @abc.abstractmethod
    def check_X(self, X):
        """
        Input validation on X
        :param X: the input
        :return:
        """

    @abc.abstractmethod
    def check_y(self, y):
        """
        Input validation on y
        :param X: the input
        :return:
        """


class GeoLocatedGeneratorMixin(GeneratorMixin):

    def check_X(self, X):
        raise NotImplementedError()

    def check_y(self, y):
        raise NotImplementedError()


class GraphGeneratorMixin(GeneratorMixin):

    def check_X(self, X):
        raise NotImplementedError()

    def check_y(self, y):
        raise NotImplementedError()


class ImageGeneratorMixin(GeneratorMixin):

    def check_X(self, X):
        raise NotImplementedError()

    def check_y(self, y):
        raise NotImplementedError()


class TabularDataGeneratorMixin(GeneratorMixin):

    def check_X(self, X):
        raise NotImplementedError()

    def check_y(self, y):
        raise NotImplementedError()


class TimeSeriesDataGeneratorMixin(GeneratorMixin):

    def check_X(self, X):
        raise NotImplementedError()

    def check_y(self, y):
        raise NotImplementedError()


class TextDataGeneratorMixin(GeneratorMixin):

    def check_X(self, X):
        raise NotImplementedError()

    def check_y(self, y):
        raise NotImplementedError()
