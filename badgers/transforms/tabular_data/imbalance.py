from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


class ImbalanceTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that makes tabular data imbalanced
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator

