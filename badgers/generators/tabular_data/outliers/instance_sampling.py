import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.outliers import OutliersGenerator


class UniformInstanceAttributeSampling(OutliersGenerator):
    """
    Randomly generates outliers by sampling from existing instances attributes uniformly at random
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the UniformInstanceAttributeSampling with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_outliers: int = 10):
        """


        :param X: the input features (pandas DataFrame or numpy array).
        :param y: the class labels, target values, or None (if not provided).
        :param n_outliers: The number of outliers to generate.
        :return: A tuple containing the augmented feature matrix with added outliers and the corresponding target values.
                 If `y` is None, the returned target values will also be None.
        """

        outliers = pd.DataFrame(
            data=np.stack([self.random_generator.choice(X.iloc[:,i], size=n_outliers) for i in range(X.shape[1])]).T,
            columns = X.columns
        )

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        return outliers, yt