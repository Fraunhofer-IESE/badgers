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

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: A random generator


        """
        super().__init__(random_generator)

    def generate(self, X, y, swap_proba:float=0.1) -> Tuple:
        """
        For each word with a length greater than 3, apply a single swap with probability `self.swap_proba`
        Where the swap happens is determined randomly


        :param X: A list of words where we apply typos
        :param y: not used
        :param swap_proba: Each word with a length greater than 3 will have this probability to contain a switch (max one per word)
        :return: the transformed list of words
        """
        for i in range(len(X)):
            if len(X[i]) > 3 and self.random_generator.random() <= swap_proba:
                # get the ith word in the list and make it a list of letters
                word = list(X[i])
                # randomly chose letters to switch
                idx = self.random_generator.integers(1, len(word) - 2)
                word[idx], word[idx + 1] = word[idx + 1], word[idx]
                # save the word with switched letters as string
                X[i] = ''.join(word)

        return X, y


class LeetSpeakGenerator(TyposGenerator):

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: a random number generator
        """
        super().__init__(random_generator=random_generator)
        self.leet_speak_mapping = {
            "A": ["4", "/\\", "@", "/-\\", "^", "(L", "\u0414"],
            "B": ["I3", "8", "13", "|3", "\u00df", "!3", "(3", "/3", ")3", "|-]", "j3"],
            "C": ["[", "\u00a2", "<", "(", "\u00a9"],
            "D": [")", "|)", "(|", "[)", "I>", "|>", "?", "T)", "I7", "cl", "|}", "|]"],
            "E": ["3", "&", "\u00a3", "\u20ac", "[-", "|=-"],
            "F": ["|=", "\u0192", "|#", "ph", "/=", "v"],
            "G": ["6", "&", "(_+", "9", "C-", "gee", "(?,", "[,", "{,", "<-", "(."],
            "H": ["#", "/-/", "\\-\\", "[-]", "]-[", ")-(", "(-)", ":-:", "|~|", "|-|", "]~[", "}{", "!-!", "1-1",
                  "\\-/", "I+I", "?"],
            "I": ["1", "|", "][", "!", "eye", "3y3"],
            "J": [",_|", "_|", "._|", "._]", "_]", ",_]", "]"],
            "K": [">|", "|<", "1<", "|c", "|(7<"],
            "L": ["1", "2", "\u00a3", "7", "|_", "|"],
            "M": ["/\\/\\", "/V\\", "[V]", "|\\/|", "^^", "<\\/>", "{V}", "(v)", "(V)", "|\\|\\", "]\\/[", "nn", "11"],
            "N": ["^/", "|\\|", "/\\/", "[\\]", "<\\>", "{\\}", "/V", "^", "\u0e17", "\u0418"],
            "O": ["0", "()", "oh", "[]", "p", "<>", "\u00d8"],
            "P": ["|*", "|o", "|\u00ba", "?", "|^", "|>", "|\"", "9", "[]D", "|\u00b0", "|7"],
            "Q": ["(_,)", "()_", "2", "0_", "<|", "&", "9", "\u00b6", "\u204b", "\u2117"],
            "R": ["I2", "9", "|`", "|~", "|?", "/2", "|^", "lz", "7", "2", "12", "\u00ae", "[z", "\u042f", ".-", "|2",
                  "|-", "3"],
            "S": ["5", "$", "z", "\u00a7", "ehs", "es", "2"],
            "T": ["7", "+", "-|-", "']['", "\u2020", "\u00ab|\u00bb", "~|~"],
            "U": ["(_)", "|_|", "v", "L|", "\u0e1a"],
            "V": ["\\/", "|/", "\\|"],
            "W": ["\\/\\/", "vv", "\\N", "'//", "\\\\'", "\\^/", "\\/\\/", "(n)", "\\V/", "\\X/", "\\|/", "\\_|_/",
                  "\\_:_/", "uu", "2u", "\\\\//\\\\//", "\u0e1e", "\u20a9"],
            "X": ["><", "}{", "ecks", "\u00d7", "?", "}{", ")(", "]["],
            "Y": ["j", "`/", "\\|/", "\u00a5", "\\//"],
            "Z": ["2", "7_", "-/_", "%", ">_", "s", "~/_", "-\\_", "-|_"]
        }

    def randomly_replace_letter(self, letter, replacement_proba):
        """
        Randomly replace a letter with its leet counterpart
        :param letter:
        :param replacement_proba: the probability of replacing a letter with its leet counterpart
        :return:
        """
        if self.random_generator.random() < replacement_proba:
            return self.random_generator.choice(self.leet_speak_mapping[letter.upper()])
        else:
            return letter

    def generate(self, X, y, replacement_proba: float = 0.1) -> Tuple:
        """

        :param X: A list of words where we apply leet replacement
        :param y:
        :param replacement_proba: the probability of replacing a letter with its leet counterpart
        :return:
        """
        assert 0 <= replacement_proba <= 1
        Xt = [
            ''.join([self.randomly_replace_letter(l, replacement_proba=replacement_proba) for l in word])
            for word in X
        ]

        return Xt, y
