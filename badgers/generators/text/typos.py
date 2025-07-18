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
        Initialize the TyposGenerator with a given random number generator.

        :param random_generator: A random number generator used to introduce randomness in typo generation.
        :type random_generator: numpy.random.Generator, default=default_rng(seed=0)
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
        Initialize the SwapLettersGenerator with a given random number generator.

        :param random_generator: A random number generator used to introduce randomness in letter swapping.
        :type random_generator: numpy.random.Generator, default=default_rng(seed=0)
        """
        super().__init__(random_generator)

    def generate(self, X, y, swap_proba:float=0.1) -> Tuple:
        """
        For each word with a length greater than 3, apply a single swap with probability `swap_proba`.
        The position of the swap is chosen randomly among possible adjacent pairs of letters,
        excluding the first and last letters of the word.
        :param X: A list of words where typos are introduced.
        :param y: Not used in this method.
        :param swap_proba: Probability that a word with more than 3 characters will have one adjacent pair of letters swapped.
                           This probability applies to each eligible word independently.
        :return: A tuple containing the transformed list of words and the original labels `y` (unchanged).
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
        Initialize the LeetSpeakGenerator with a given random number generator.

        :param random_generator: A random number generator used to introduce randomness in leetspeak transformation.
        :type random_generator: numpy.random.Generator, default=default_rng(seed=0)
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
        Randomly replace a letter with its leet counterpart based on the provided probability.

        :param letter: The letter to potentially replace.
        :type letter: str
        :param replacement_proba: The probability of replacing the letter with its leet counterpart.
        :type replacement_proba: float
        :return: The replaced letter if a random draw is less than or equal to the replacement_proba, otherwise the original letter.
        :rtype: str
        """
        if letter.upper() in self.leet_speak_mapping:
            if self.random_generator.random() < replacement_proba:
                letter = self.random_generator.choice(self.leet_speak_mapping[letter.upper()])

        return letter

    def generate(self, X, y, replacement_proba: float = 0.1) -> Tuple:
        """
        Apply leet speak transformation to a list of words.

        :param X: A list of words where leet speak transformation is applied.
        :param y: The labels associated with the words, which remain unchanged.
        :param replacement_proba: The probability of replacing a letter with its leet counterpart.
                                  This probability applies to each letter in each word independently.
        :return: A tuple containing the transformed list of words and the original labels `y` (unchanged).
        """
        transformed_X = []
        for word in X:
            transformed_word = ''.join(
                self.randomly_replace_letter(letter, replacement_proba) for letter in word
            )
            transformed_X.append(transformed_word)

        return transformed_X, y
        assert 0 <= replacement_proba <= 1
        Xt = [
            ''.join([self.randomly_replace_letter(l, replacement_proba=replacement_proba) for l in word])
            for word in X
        ]

        return Xt, y

class SwapCaseGenerator(TyposGenerator):

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the SwapCaseGenerator with a given random number generator.

        :param random_generator: A random number generator used to introduce randomness in case swapping.
        :type random_generator: numpy.random.Generator, default=default_rng(seed=0)
        """
        super().__init__(random_generator)

    def randomly_swapcase_letter(self, letter, swapcase_proba):
        """
        Randomly swap the case of a letter based on the provided probability.

        :param letter: The letter whose case may be swapped.
        :type letter: str
        :param swapcase_proba: The probability of swapping the case of the letter.
        :type swapcase_proba: float
        :return: The letter with swapped case if a random draw is less than or equal to the swapcase_proba, otherwise the original letter.
        :rtype: str
        """
        if self.random_generator.random() < swapcase_proba:
            letter = letter.swapcase()

        return letter

    def generate(self, X, y, swapcase_proba: float = 0.1) -> Tuple:
        """
        Apply random case swapping to each letter in a list of words.

        :param X: A list of words where random case swapping is applied.
        :param y: The labels associated with the words, which remain unchanged.
        :param swapcase_proba: The probability of swapping the case of each letter.
                               This probability applies to each letter in each word independently.
        :return: A tuple containing the transformed list of words and the original labels `y` (unchanged).
        """
        assert 0 <= swapcase_proba <= 1
        Xt = [
            ''.join([self.randomly_swapcase_letter(l, swapcase_proba=swapcase_proba) for l in word])
            for word in X
        ]

        return Xt, y