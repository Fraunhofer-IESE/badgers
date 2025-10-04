import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.utils import random_sign, random_spherical_coordinate
from badgers.generators.tabular_data.outliers import OutliersGenerator


class HyperCubeSampling(OutliersGenerator):
    """
    Sampling uniformly at random within a hypercube encapsulating all the instances


    See section 6.1.1 in [1]

    [1] Georg Steinbuss and Klemens Böhm. 2021.
        Generating Artificial Outliers in the Absence of Genuine Ones — A Survey.
        ACM Trans. Knowl. Discov. Data 15, 2, Article 30 (April 2021), 37 pages.
        https://doi.org/10.1145/3447822
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the HyperCubeSampling with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_outliers: int = 10, expansion: float = 0.0):
        """

        How to set the values for expansion.
        Per default expansion = 0, this means the hypercube will cover all the instances using min and max as boundaries
        It is possible to make the hypercube bigger, as proposed in [1] section 6.1.1

            Instances from Data usually determine the bounds a, b ∈ IRd . For this reason, this approach
            needs them as input. Tax and Duin [51] and Fan et al. [21] state only that these bounds should be
            chosen so that the hyper-rectangle encapsulates all genuine instances. [ 48] uses the minimum and
            maximum for each attribute obtained from Data. Theiler and Michael Cai [52] mention that the
            boundary does not need to be far beyond these boundaries. Abe et al. [1] propose the rule that the
            boundary should expand the minimum and maximum by 10%. Désir et al. [17] propose to expand
            the boundary by 20%.

        For expanding the hypercube by 10% use expansion = 0.1, for 20% use 0.2, etc.

        :param X: the input features (pandas DataFrame or numpy array).
        :param y: the class labels, target values, or None (if not provided).
        :param n_outliers: The number of outliers to generate.
        :param expansion: how much the hypercube shall be expanded beyond (min,max) range, in percent (0.1 == 10%)
        :return: A tuple containing the augmented feature matrix with added outliers and the corresponding target values.
                 If `y` is None, the returned target values will also be None.
        """
        assert expansion >= 0
        low = 0 - expansion
        high = 1 + expansion

        scaler = MinMaxScaler()
        scaler.fit(X)

        outliers = self.random_generator.uniform(low=low, high=high, size=(n_outliers, X.shape[1]))

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        return scaler.inverse_transform(outliers), yt


class ZScoreSamplingGenerator(OutliersGenerator):
    """
    Randomly generates outliers as data points with a z-score > 3.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the ZScoreSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_outliers: int = 10):
        """
        Randomly generates outliers as data points with a z-score > 3.

        The process involves the following steps:
        1. Standardize the input data so that it has a mean of 0 and a variance of 1.
        2. Generate outliers by:
           - choosing a random sign for each outlier.
           - for each dimension of the data, set the value to be 3 plus a random number drawn from an exponential distribution
             with default parameters (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html).
        3. Apply the inverse of the standardization transformation to convert the generated outliers back to the original scale.

        :param X: the input features (pandas DataFrame or numpy array).
        :param y: the class labels, target values, or None (if not provided).
        :param n_outliers: The number of outliers to generate.
        :return: A tuple containing the augmented feature matrix with added outliers and the corresponding target values.
                 If `y` is None, the returned target values will also be None.
        """

        # standardize X
        scaler = StandardScaler()

        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        # generate outliers
        outliers = np.array([
            random_sign(self.random_generator, size=Xt.shape[1]) * (
                3. + self.random_generator.exponential(size=Xt.shape[1]))
            for _ in range(n_outliers)
        ])

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        return scaler.inverse_transform(outliers), yt


class HypersphereSamplingGenerator(OutliersGenerator):
    """
    Generates outliers by sampling points from a hypersphere with radius at least 3 sigma
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the HypersphereSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, n_outliers: int = 10):
        """
        Randomly generates outliers by sampling points from a hypersphere.

        The process involves the following steps:
        1. Standardize the input data so that it has a mean of 0 and a variance of 1.
        2. Generate outliers by:
           - choosing angles uniformly at random for each dimension of the data.
           - setting the radius to be 3 plus a random number drawn from an exponential distribution with default parameters
             (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html).
        3. Convert the spherical coordinates to Cartesian coordinates.
        4. Apply the inverse of the standardization transformation to convert the generated outliers back to the original scale.

        :param X: the input features (pandas DataFrame or numpy array).
        :param y: the class labels, target values, or None (if not provided).
        :param n_outliers: The number of outliers to generate.
        :return: A tuple containing the augmented feature matrix with added outliers and the corresponding target values.
                 If `y` is None, the returned target values will also be None.
        """

        # standardize X
        scaler = StandardScaler()

        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        # computing outliers
        outliers = np.array([
            random_spherical_coordinate(
                random_generator=self.random_generator,
                size=Xt.shape[1],
                radius=3. + self.random_generator.exponential()
            )
            for _ in range(n_outliers)
        ])

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        return scaler.inverse_transform(outliers), yt
