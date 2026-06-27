import warnings

import numpy as np
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.outliers import OutliersGenerator


class IndependentHistogramsGenerator(OutliersGenerator):
    """
    Randomly generates outliers from low density regions.
    Low density regions are estimated through several independent histograms (one for each feature).

    For each feature (column), a histogram is computed (it approximates the marginal distribution).
    Values are generated from bins with a low number of data points.

    All values generated for each feature are simply concatenated (independence hypothesis!).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, n_outliers: int = 10, bins: int = 10):
        """
        Randomly generates outliers from low density regions.
        Low density regions are estimated through several independent histograms (one for each feature).

        For each feature (column), a histogram is computed (it approximates the marginal distribution).
        Values are generated from bins with a low number of data points.

        All values generated for each feature are simply concatenated (independence hypothesis!).

        :param X: the input features (pandas DataFrame or numpy array).
        :param y: the class labels, target values, or None (not used).
        :param n_outliers: The number of outliers to generate.
        :param bins: The number of bins to use when creating histograms for each feature.
        :return: A tuple containing the augmented feature matrix with added outliers and the corresponding target values.
                  If `y` is None, the returned target values will also be None.
        """
        assert n_outliers > 0
        assert bins > 0

        # Pre-allocate output array for memory efficiency
        n_features = X.shape[1]
        outliers = np.empty((n_outliers, n_features), dtype=X.dtype)

        # Compute independent histograms for each feature
        for col in range(n_features):
            hist, bin_edges = np.histogram(X[:, col], bins=bins)

            # Compute inverse density and sampling probabilities
            inv_density = 1 - hist / np.max(hist)
            p = inv_density / np.sum(inv_density)

            # Generate bin indices for this feature
            indices = self.random_generator.choice(
                bins, p=p, size=n_outliers, replace=True
            )

            # Uniform sampling from selected bins
            outliers[:, col] = self.random_generator.uniform(
                low=bin_edges[indices],
                high=bin_edges[indices + 1]
            )

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        return outliers, yt


class HistogramSamplingGenerator(OutliersGenerator):
    """
    Randomly generates outliers from low density regions.
    Low density regions are estimated through a histogram.

    -----------------------------------------
    WARNING:
    This computes a full histogram in d-dimensions (d = nb features / columns), which is O(d²).
    Should only be used with low dimensionality data!
    It will raise an error if the number of dimensions is greater than 5.
    -----------------------------------------

    TODO: this works but is very inefficient, better strategies are welcome!
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the HistogramSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """

        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, n_outliers: int = 10,
                 threshold_low_density: float = 0.1, bins: int = 10):
        """
        Randomly generates outliers from low density regions. Low density regions are estimated through histograms

        1. Standardize the input data (mean = 0, variance = 1)
        2. Compute and normalize histogram for the data
        3. Sample datapoint uniformly at random within bins of low density
        4. Inverse the standardization transformation

        :param X: the input features
        :param y: not used
        :param n_outliers: The number of outliers to generate
        :param threshold_low_density: the threshold that defines a low density region (must be between 0 and 1)
        :param bins: number of bins for the histogram
        :return:
        """
        assert n_outliers > 0
        assert bins > 0
        assert 0 < threshold_low_density < 1
        n_features = X.shape[1]
        if n_features > 5:
            raise NotImplementedError('So far this generator only supports tabular data with at most 5 columns')
        # standardize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)

        # compute the histogram of the data
        hist, edges = np.histogramdd(Xt, density=False, bins=bins)
        # normalize - use max only (min is always 0 for histogram counts)
        norm_hist = hist / np.max(hist)
        # get coordinates of the histogram where the density is low (below a certain threshold)
        hist_coords_low_density = np.where(norm_hist <= threshold_low_density)
        low_density_coordinates = list(zip(*hist_coords_low_density))
        if len(low_density_coordinates) == 0:
            raise ValueError(f"Could not find regions of low density. Maybe threshold {threshold_low_density} is too high!")
        # randomly pick some coordinates in the histogram where the density is low
        hist_coords_random = self.random_generator.choice(low_density_coordinates, n_outliers, replace=True)

        # Vectorized outlier generation using meshgrid for efficient sampling
        n_out = hist_coords_random.shape[0]
        outliers = np.empty((n_out, n_features), dtype=Xt.dtype)

        for col in range(n_features):
            # Get min and max for this dimension
            dim_min = edges[col][0]
            dim_max = edges[col][-1]
            # Get bin indices for this dimension from the random coordinates
            bin_indices = np.array([coord[col] for coord in hist_coords_random])
            # Sample uniformly within each selected bin
            bin_width = edges[col][bin_indices + 1] - edges[col][bin_indices]
            random_offsets = self.random_generator.uniform(0, 1, size=n_out)
            outliers[:, col] = edges[col][bin_indices] + random_offsets * bin_width

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        return scaler.inverse_transform(outliers), yt


class LowDensitySamplingGenerator(OutliersGenerator):
    """
    Randomly generates outliers from low density regions.
    Low density regions are estimated using a KernelDensity estimator.
    Points are sampled uniformly at random and filtered out if they do not belong to a low density region

    TODO: this works but might not be efficient, a better sampling strategy is welcome
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the LowDensitySamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        super().__init__(random_generator=random_generator)
        self.density_estimator: KernelDensity = KernelDensity(bandwidth="scott")

    @preprocess_inputs
    def generate(self, X, y=None, n_outliers: int = 10, threshold_low_density: float = 0.1, max_samples: int = 100):
        """
        Generate data points belonging to low density regions.

        Pseudo code:
        - Standardize the data X
        - Estimate the density based upon the original data X
        - Computes a threshold for determining low density (so far 10th percentile)
        - Sample uniformly at random within the hypercube [min, max]
        - Estimate the density of the new points and filter out the ones with a density that is above the threshold

        :param X: the input features
        :param y: not used
        :param n_outliers: The number of outliers to generate
        :param threshold_low_density: the threshold that defines a low density region (must be between 0 and 1)
        :param max_samples:
        :return:
        """
        assert n_outliers > 0
        assert 0 < threshold_low_density < 1
        assert max_samples > 0
        # standardize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # fit density estimator
        self.density_estimator = self.density_estimator.fit(Xt)
        low_density_threshold = np.percentile(self.density_estimator.score_samples(Xt), threshold_low_density)

        if max_samples is None:
            max_samples = n_outliers * 100

        # Cache min/max for vectorized sampling
        dim_min = np.min(Xt, axis=0)
        dim_max = np.max(Xt, axis=0)

        # Generate all candidate samples at once (vectorized)
        candidates = self.random_generator.uniform(
            low=dim_min,
            high=dim_max,
            size=(max_samples, Xt.shape[1])
        )

        # Vectorized density estimation for all candidates (batch operation)
        densities = self.density_estimator.score_samples(candidates)

        # Vectorized filtering - keep only low density samples
        mask = densities <= low_density_threshold
        outliers = candidates[mask]

        if outliers.shape[0] < n_outliers:
            warnings.warn(
                f'LowDensitySamplingGenerator could not generate all {n_outliers} outliers. It only generated {len(outliers)}.')
        else:
            outliers = outliers[:n_outliers]

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        # add "outliers" as labels for outliers
        yt = np.array(["outliers"] * len(outliers))

        # in the case no outliers could be generated
        if outliers.shape[0] == 0:
            return outliers, yt

        return scaler.inverse_transform(outliers), yt
