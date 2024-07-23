import unittest
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from badgers.core.utils import normalize_proba, random_sign, random_spherical_coordinate


class TestUtils(TestCase):

    def setUp(self) -> None:
        self.rng = default_rng(0)

    def test_normalize_proba(self):
        p = self.rng.uniform(0, 1, size=(3, 5))
        p = normalize_proba(p)
        self.assertEqual(p.shape, (3, 5))
        for s in p.sum(axis=0).tolist():
            self.assertAlmostEqual(s, 1.0)

    def test_random_signs(self):
        for X in [np.ones(1), np.ones(10), np.ones((10, 10))]:
            Xt = random_sign(random_generator=self.rng, size=X.shape)
            with self.subTest(X=X):
                self.assertEqual(
                    X.shape, Xt.shape
                )
                self.assertTrue(
                    np.all(X == np.abs(Xt))
                )
                if X.shape != (1,):
                    self.assertNotEqual(
                        np.sum(X == 1), np.sum(Xt == 1)
                    )

    def test_random_spherical_coordinate(self):
        size = 2
        r = 1
        x = random_spherical_coordinate(random_generator=self.rng, size=size, radius=r)
        self.assertAlmostEqual(np.sum(x ** 2), r ** 2)


if __name__ == '__main__':
    unittest.main()
