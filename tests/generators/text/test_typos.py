import unittest
from copy import deepcopy

from numpy.random import default_rng

from badgers.generators.text.typos import SwapLettersGenerator, LeetSpeakGenerator


class TestSwapLettersGenerator(unittest.TestCase):
    def test_transform(self):
        """
        For all words with a length strictly greater than 3:
        - The first and the last letter must be the same (i.e., these are not switched)
        - Both original and transformed words must have the same length
        - Both original and transformed words must have the same letters (i.e., letters are switched but not replaced)

        For smaller words (length lower than or equal to 3):
        - no change occurs
        """
        trf = SwapLettersGenerator(random_generator=default_rng(0))
        X = [
            'abcdef',
            'abcde',
            'abcd',
            'abc',
            'ab',
            'a'
        ]

        Xt, _ = trf.generate(X=deepcopy(X), y=None, swap_proba=1)

        for i in range(len(X)):
            if len(X[i]) > 3:
                self.assertNotEqual(Xt[i], X[i])
                self.assertEqual(Xt[i][0], X[i][0])
                self.assertEqual(Xt[i][-1], X[i][-1])
                self.assertEqual(len(Xt[i]), len(X[i]))
                self.assertEqual(set(Xt[i]), set(X[i]))
            else:
                self.assertEqual(Xt[i], X[i])


class TestLeetSpeakGenerator(unittest.TestCase):

    def test_generate(self):
        X = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'fox']
        trf = LeetSpeakGenerator()
        Xt, _ = trf.generate(X, None)
        self.assertEqual(len(X), len(Xt))


if __name__ == '__main__':
    unittest.main()
