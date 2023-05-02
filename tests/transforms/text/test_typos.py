import unittest
from copy import deepcopy

from numpy.random import default_rng

from badgers.transforms.text.typos import SwitchLettersTransformer


class TestSwitchLettersTransformer(unittest.TestCase):
    def test_transform(self):
        """
        For all words with a length strictly greater than 3:
        - The first and the last letter must be the same (i.e., these are not switched)
        - Both original and transformed words must have the same length
        - Both original and transformed words must have the same letters (i.e., letters are switched but not replaced)

        For smaller words (length lower than or equal to 3):
        - no change occurs
        """
        trf = SwitchLettersTransformer(random_generator=default_rng(0), switching_proba=1)
        X = [
            'abcdef',
            'abcde',
            'abcd',
            'abc',
            'ab',
            'a'
        ]

        X_transformed = trf.transform(deepcopy(X))

        for i in range(len(X)):
            if len(X[i]) > 3:
                self.assertNotEqual(X_transformed[i], X[i])
                self.assertEqual(X_transformed[i][0], X[i][0])
                self.assertEqual(X_transformed[i][-1], X[i][-1])
                self.assertEqual(len(X_transformed[i]), len(X[i]))
                self.assertEqual(set(X_transformed[i]), set(X[i]))
            else:
                self.assertEqual(X_transformed[i], X[i])


if __name__ == '__main__':
    unittest.main()
