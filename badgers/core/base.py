import abc


class GeneratorMixin:

    @abc.abstractmethod
    def generate(self, X, **params):
        """

        :param X: the input
        :param params: optional parameters
        :return: Xt
        """
        pass
