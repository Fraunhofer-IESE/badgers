from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


class NoiseTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator


class GaussianNoiseTransformer(NoiseTransformer):
    def __init__(self, random_generator=default_rng(seed=0), signal_to_noise_ratio: float = 0.1):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param signal_to_noise_ratio: float, default 0.1
            The standard deviation of the noise to be added
        """
        super().__init__(random_generator=random_generator)
        self.signal_to_noise_ratio = signal_to_noise_ratio

    def transform(self, X):
        """
        Add Gaussian white noise to the data.
        The data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param X:
        :return:
        """
        X = check_array(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        #
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise
        Xt = Xt + self.random_generator.normal(loc=0, scale=self.signal_to_noise_ratio, size=Xt.shape)
        # inverse pca
        return scaler.inverse_transform(Xt)
