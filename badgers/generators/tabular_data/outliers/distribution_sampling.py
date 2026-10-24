import numpy as np
from numpy.random import default_rng

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import (
    HypersphereSampler,
    UniformOutOfDistributionSampler,
    ZScoreSampler,
)
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
    def generate(self, X, n_outliers: int = 10, expansion: float = 0.0):
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
        :param n_outliers: The number of outliers to generate.
        :param expansion: how much the hypercube shall be expanded beyond (min,max) range, in percent (0.1 == 10%)
        :return: The augmented feature matrix with added outliers.
        """
        assert n_outliers > 0
        assert expansion >= 0

        outliers = UniformOutOfDistributionSampler(expansion=expansion).sample(
            self.random_generator, X, columns=list(range(X.shape[1])), n_samples=n_outliers
        )

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), n_outliers)
        return Xt


class ZScoreSamplingGenerator(OutliersGenerator):
    """
    Randomly generates outliers as data points with a z-score > 3.

    Very similar to "GaussTail" in section 6.1.5 in [1]

    [1] Georg Steinbuss and Klemens Böhm. 2021.
        Generating Artificial Outliers in the Absence of Genuine Ones — A Survey.
        ACM Trans. Knowl. Discov. Data 15, 2, Article 30 (April 2021), 37 pages.
        https://doi.org/10.1145/3447822
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the ZScoreSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, n_outliers: int = 10, scale: float = 1.0):
        r"""
        Randomly generates outliers as data points with a z-score > 3.

        Outlier values are produced by
        :class:`badgers.core.sampling.ZScoreSampler` directly on the
        original scale: each column independently gets a random sign and a
        distance of ``3 + exponential(scale)`` standard deviations from the
        column mean.

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The number of outliers to generate.
        :param scale: float (the scale parameter from
                    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
                    The scale parameter, :math:`\beta = 1/\lambda`. Must be
                    positive.
        :return: The augmented feature matrix with added outliers.
        """
        assert n_outliers > 0
        assert scale > 0

        outliers = ZScoreSampler(scale=scale).sample(
            self.random_generator, X, columns=list(range(X.shape[1])), n_samples=n_outliers
        )

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), n_outliers)
        return Xt


class HypersphereSamplingGenerator(OutliersGenerator):
    """
    Generates outliers by sampling points from a hypersphere with radius at least 3 sigma

    Very similar to "GaussTail" in section 6.1.5 in [1]

    [1] Georg Steinbuss and Klemens Böhm. 2021.
        Generating Artificial Outliers in the Absence of Genuine Ones — A Survey.
        ACM Trans. Knowl. Discov. Data 15, 2, Article 30 (April 2021), 37 pages.
        https://doi.org/10.1145/3447822
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the HypersphereSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, n_outliers: int = 10, scale: float = 1.0):
        r"""
        Generates outliers by sampling points from a hypersphere with radius at least 3 sigma.

        Outlier values are produced by
        :class:`badgers.core.sampling.HypersphereSampler` directly on the
        original scale: radii of ``3 + exponential(scale)`` in standardized
        space are converted to Cartesian coordinates and mapped back per
        column (``mean + z * std``).

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The number of outliers to generate.
        :param scale: float (the scale parameter from
                    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
                    The scale parameter, :math:`\beta = 1/\lambda`. Must be
                    positive.
        :return: The augmented feature matrix with added outliers.
        """
        assert n_outliers > 0
        assert scale > 0

        outliers = HypersphereSampler(scale=scale).sample(
            self.random_generator, X, columns=list(range(X.shape[1])), n_samples=n_outliers
        )

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), n_outliers)
        return Xt
