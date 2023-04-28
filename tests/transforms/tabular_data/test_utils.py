import unittest
from unittest import TestCase

from numpy.random import default_rng

from badgers.transforms.tabular_data.utils import normalize_proba


class TestUtils(TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)

    def test_normalize_proba(self):
        p = self.rng.uniform(0, 1, size=(3, 5))
        p = normalize_proba(p)
        self.assertEqual(p.shape, (3, 5))
        for s in p.sum(axis=0).tolist():
            self.assertAlmostEqual(s, 1.0)


if __name__ == '__main__':
    unittest.main()
