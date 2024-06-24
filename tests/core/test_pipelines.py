import unittest

from numpy.random import default_rng
from sklearn.datasets import make_blobs

from badgers.core.pipeline import Pipeline
from badgers.generators.tabular_data.imbalance import RandomSamplingClassesGenerator
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator


class MyTestCase(unittest.TestCase):
    def test_generate(self):
        random_generator = default_rng(0)
        X, y = make_blobs(centers=3, random_state=0)
        generators = {
            'imbalance': RandomSamplingClassesGenerator(random_generator=random_generator),
            'noise': GaussianNoiseGenerator(random_generator=random_generator)
        }
        pipeline = Pipeline(generators=generators)
        params = {
            'imbalance': {'proportion_classes': {0: 0.5, 1: 0.25, 2: 0.25}},
            'noise': {'noise_std':0.5}
        }
        Xt, yt = pipeline.generate(X=X, y=y, params=params)


if __name__ == '__main__':
    unittest.main()
