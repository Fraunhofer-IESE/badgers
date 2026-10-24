import warnings

import numpy as np
from numpy.random import default_rng

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import InstanceSampler
from badgers.generators.tabular_data.outliers import OutliersGenerator


class UniformInstanceAttributeSampling(OutliersGenerator):
    """
    Randomly generates outliers by sampling from existing instances attributes uniformly at random.

    Each appended outlier row is a per-cell bootstrap: every cell copies
    the value of a random real row (via
    :class:`badgers.core.sampling.InstanceSampler`), so values are
    individually plausible while the row as a whole never existed.

    The generator has no seed rows, so a selector has nothing to fire on:
    ``selector`` is accepted for API uniformity but IGNORED (a warning is
    emitted; ``n_outliers`` remains the only count knob). Requested counts
    are capped at the dataset size.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the UniformInstanceAttributeSampling with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, n_outliers: int = 10, *, selector=None):
        """
        Randomly generates outliers by sampling from existing instances attributes uniformly at random.

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The number of outliers to generate (must not exceed the number of rows of X).
        :param selector: accepted for API uniformity, ignored (this generator has no seed rows; the copy pool stays all of X).
        :return: The augmented feature matrix with added outliers.
        """
        if selector is not None:
            warnings.warn(
                "UniformInstanceAttributeSampling has no seed rows: the "
                "selector is ignored (the copy pool stays all of X) and "
                "n_outliers keeps counting the appended rows.",
                UserWarning,
                stacklevel=2,
            )
        if n_outliers > len(X):
            raise ValueError(
                f"n_outliers must not exceed the number of rows of X "
                f"({len(X)}), got {n_outliers}")
        assert n_outliers > 0

        outliers = InstanceSampler().sample(
            self.random_generator, X, columns=list(range(X.shape[1])),
            n_samples=n_outliers,
        )

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), n_outliers)
        return Xt