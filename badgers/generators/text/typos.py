import abc
from typing import Tuple

from numpy.random import default_rng

from badgers.core.base import GeneratorMixin


class TyposGenerator(GeneratorMixin):
    """
    Base class for transformers creating typos in a list of words
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class SwapLettersGenerator(TyposGenerator):
    """
    Swap adjacent letters in words randomly except for the first and the last letters.
    Example: 'kilogram' --> 'kilogarm'
    """

    def __init__(self, random_generator=default_rng(seed=0), swap_proba=0.1):
        """

        :param random_generator: A random generator
        :param swap_proba: Each word with a length greater than 3 will have this probability to contain a switch (max one per word)

        """
        super().__init__(random_generator)
        self.switching_proba = swap_proba

    def generate(self, X, y, **params) -> Tuple:
        """
        For each word with a length greater than 3, apply a single swap with probability `self.swap_proba`
        Where the swap happens is determined randomly

        :param X: A list of words where we apply typos
        :return: the transformed list of words
        """
        for i in range(len(X)):
            if len(X[i]) > 3 and self.random_generator.random() <= self.switching_proba:
                # get the ith word in the list and make it a list of letters
                word = list(X[i])
                # randomly chose letters to switch
                idx = self.random_generator.integers(1, len(word) - 2)
                word[idx], word[idx + 1] = word[idx + 1], word[idx]
                # save the word with switched letters as string
                X[i] = ''.join(word)

        return X, None
