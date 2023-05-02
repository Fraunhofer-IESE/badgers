from typing import List

from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator


class TyposTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers creating typos in a list of words
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator


class SwitchLettersTransformer(TyposTransformer):
    """
    Switch adjacent letters in words randomly except for the first and the last letters.
    Example: 'kilogram' --> 'kilogarm'
    """

    def __init__(self, random_generator=default_rng(seed=0), switching_proba=0.1):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param switching_proba: float (between 0 and 1), default = 0.1
            Each word with a length greater than 3 will have this probability to contain a switch (max one per word)

        """
        super().__init__(random_generator)
        self.switching_proba = switching_proba

    def transform(self, X: List[str]) -> List[str]:
        """
        For each word with a length greater than 3, apply a single switch with probability `self.switching_proba`
        Where the switch happens is determined randomly

        :param X: list of words (str)
            A list of words where we apply typos
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

        return X
