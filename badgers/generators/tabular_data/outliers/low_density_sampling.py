import warnings

import numpy as np
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import InstanceSampler
from badgers.core.selectors.base import _resolve_targets
from badgers.generators.tabular_data.outliers import OutliersGenerator


def _resolve_count_and_targets(X, n_outliers, selector):
    """Shared trio plumbing: count validation + synthesized-column targets.

    The selector's KEYS choose the synthesized columns (its mask is
    discarded and the selector is never invoked); ``selector=None`` means
    all columns. The count must not exceed the dataset size (a larger
    count raises).
    """
    if n_outliers < 0:
        raise ValueError(f"n_outliers must be a positive count, got {n_outliers}")
    if n_outliers > len(X):
        raise ValueError(
            f"n_outliers must not exceed the number of rows of X "
            f"({len(X)}), got {n_outliers}")
    n_features = X.shape[1]
    if selector is None:
        return list(range(n_features))
    return sorted(_resolve_targets(selector, X, None))


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
    def generate(self, X, n_outliers: int = 10, bins: int = 10, *, selector=None):
        """
        Randomly generates outliers from low density regions.
        Low density regions are estimated through several independent histograms (one for each feature).

        For each feature (column), a histogram is computed (it approximates the marginal distribution).
        Values are generated from bins with a low number of data points.

        All values generated for each feature are simply concatenated (independence hypothesis!).

        The selector's keys choose the columns where the synthesis happens
        (its mask is discarded); the remaining columns of the appended rows
        are per-cell bootstrap copies of real X values. The requested count
        must not exceed ``len(X)`` (a larger count raises).

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The number of outliers to generate
            (must not exceed the number of rows of X).
        :param bins: The number of bins to use when creating histograms for each feature.
        :param selector: optional selector from ``badgers.core.selectors``;
            its KEYS (columns) are the synthesized columns.
        :return: The augmented feature matrix with added outliers.
        """
        if bins <= 0:
            raise ValueError(f"bins must be a positive count, got {bins}")
        targets = _resolve_count_and_targets(X, n_outliers, selector)
        if n_outliers == 0:
            self._set_outlier_labels(len(X), 0)
            return X
        rng = self.random_generator

        # Pre-allocate output array for memory efficiency
        n_features = X.shape[1]
        outliers = np.empty((n_outliers, n_features), dtype=X.dtype)

        # Compute independent histograms for each synthesized feature
        for col in targets:
            hist, bin_edges = np.histogram(X[:, col], bins=bins)

            # Compute inverse density and sampling probabilities
            inv_density = 1 - hist / np.max(hist)
            p = inv_density / np.sum(inv_density)

            # Generate bin indices for this feature
            indices = rng.choice(bins, p=p, size=n_outliers, replace=True)

            # Uniform sampling from selected bins
            outliers[:, col] = rng.uniform(
                low=bin_edges[indices],
                high=bin_edges[indices + 1]
            )

        # Fill the columns the selector did not synthesize with real values
        unselected = [c for c in range(n_features) if c not in targets]
        if unselected:
            outliers[:, unselected] = InstanceSampler().sample(
                rng, X, unselected, n_outliers)

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), n_outliers)
        return Xt


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
    def generate(self, X, n_outliers: int = 10,
                 threshold_low_density: float = 0.1, bins: int = 10, *, selector=None):
        """
        Randomly generates outliers from low density regions. Low density regions are estimated through histograms

        1. Standardize the input data (mean = 0, variance = 1)
        2. Compute and normalize histogram for the data
        3. Sample datapoint uniformly at random within bins of low density
        4. Inverse the standardization transformation

        The selector's keys choose the columns where the synthesis happens —
        the joint histogram is computed over the synthesized columns only
        (its mask is discarded); the remaining columns of the appended rows
        are per-cell bootstrap copies of real X values. The requested count
        must not exceed ``len(X)`` (a larger count raises).

        :param X: the input features
        :param n_outliers: The number of outliers to generate
            (must not exceed the number of rows of X).
        :param threshold_low_density: the threshold that defines a low density region (must be between 0 and 1)
        :param bins: number of bins for the histogram
        :param selector: optional selector from ``badgers.core.selectors``;
            its KEYS (columns) are the synthesized columns.
        :return:
        """
        if bins <= 0:
            raise ValueError(f"bins must be a positive count, got {bins}")
        if not 0 < threshold_low_density < 1:
            raise ValueError(
                f"threshold_low_density must be in (0, 1), got {threshold_low_density}")
        targets = _resolve_count_and_targets(X, n_outliers, selector)
        if n_outliers == 0:
            self._set_outlier_labels(len(X), 0)
            return X
        rng = self.random_generator
        n_features = X.shape[1]
        if len(targets) > 5:
            raise NotImplementedError('So far this generator only supports tabular data with at most 5 columns to synthesize')
        # standardize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)

        # compute the histogram of the synthesized columns
        hist, edges = np.histogramdd(Xt[:, targets], density=False, bins=bins)
        # normalize - use max only (min is always 0 for histogram counts)
        norm_hist = hist / np.max(hist)
        # get coordinates of the histogram where the density is low (below a certain threshold)
        hist_coords_low_density = np.where(norm_hist <= threshold_low_density)
        low_density_coordinates = list(zip(*hist_coords_low_density))
        if len(low_density_coordinates) == 0:
            raise ValueError(f"Could not find regions of low density. Maybe threshold {threshold_low_density} is too high!")
        # randomly pick some coordinates in the histogram where the density is low
        hist_coords_random = rng.choice(low_density_coordinates, n_outliers, replace=True)

        # Vectorized outlier generation: one bin per synthesized column
        n_out = hist_coords_random.shape[0]
        outliers = np.empty((n_out, n_features), dtype=Xt.dtype)

        for j, col in enumerate(targets):
            # Get bin indices for this dimension from the random coordinates
            bin_indices = hist_coords_random[:, j]
            # Sample uniformly within each selected bin
            bin_width = edges[j][bin_indices + 1] - edges[j][bin_indices]
            random_offsets = rng.uniform(0, 1, size=n_out)
            outliers[:, col] = edges[j][bin_indices] + random_offsets * bin_width

        # Fill the columns the selector did not synthesize, in STANDARDIZED
        # space so the single inverse transform below restores real values
        unselected = [c for c in range(n_features) if c not in targets]
        if unselected:
            outliers[:, unselected] = InstanceSampler().sample(
                rng, Xt, unselected, n_out)

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        outliers = scaler.inverse_transform(outliers)

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), n_out)
        return Xt


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
    def generate(self, X, n_outliers: int = 10,
                 threshold_low_density: float = 0.1, max_samples: int = 100, *, selector=None):
        """
        Generate data points belonging to low density regions.

        Pseudo code:
        - Standardize the data X
        - Estimate the density based upon the original data X
        - Computes a threshold for determining low density (so far 10th percentile)
        - Sample uniformly at random within the hypercube [min, max]
        - Estimate the density of the new points and filter out the ones with a density that is above the threshold

        The selector's keys choose the columns where the synthesis happens —
        the density is estimated on the synthesized columns only (its mask
        is discarded); the remaining columns of the appended rows are
        per-cell bootstrap copies of real X values. The requested count
        must not exceed ``len(X)`` (a larger count raises).

        :param X: the input features
        :param n_outliers: The number of outliers to generate
            (must not exceed the number of rows of X).
        :param threshold_low_density: the threshold that defines a low density region (must be between 0 and 1)
        :param max_samples:
        :param selector: optional selector from ``badgers.core.selectors``;
            its KEYS (columns) are the synthesized columns.
        :return:
        """
        if not 0 < threshold_low_density < 1:
            raise ValueError(
                f"threshold_low_density must be in (0, 1), got {threshold_low_density}")
        if max_samples <= 0:
            raise ValueError(f"max_samples must be a positive count, got {max_samples}")
        targets = _resolve_count_and_targets(X, n_outliers, selector)
        if n_outliers == 0:
            self._set_outlier_labels(len(X), 0)
            return X
        rng = self.random_generator
        n_features = X.shape[1]
        # standardize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # fit density estimator on the synthesized columns
        self.density_estimator = self.density_estimator.fit(Xt[:, targets])
        low_density_threshold = np.percentile(
            self.density_estimator.score_samples(Xt[:, targets]), threshold_low_density)

        if max_samples is None:
            max_samples = n_outliers * 100

        # Cache min/max for vectorized sampling
        dim_min = np.min(Xt[:, targets], axis=0)
        dim_max = np.max(Xt[:, targets], axis=0)

        # Generate all candidate samples at once (vectorized)
        candidates = rng.uniform(
            low=dim_min,
            high=dim_max,
            size=(max_samples, len(targets))
        )

        # Vectorized density estimation for all candidates (batch operation)
        densities = self.density_estimator.score_samples(candidates)

        # Vectorized filtering - keep only low density samples
        mask = densities <= low_density_threshold
        outliers_synth = candidates[mask]

        if outliers_synth.shape[0] < n_outliers:
            warnings.warn(
                f'LowDensitySamplingGenerator could not generate all {n_outliers} outliers. It only generated {len(outliers_synth)}.')
        else:
            outliers_synth = outliers_synth[:n_outliers]

        n_out = outliers_synth.shape[0]
        outliers = np.empty((n_out, n_features), dtype=Xt.dtype)
        outliers[:, targets] = outliers_synth
        # Fill the columns the selector did not synthesize, in STANDARDIZED
        # space so the single inverse transform below restores real values
        unselected = [c for c in range(n_features) if c not in targets]
        if unselected:
            outliers[:, unselected] = InstanceSampler().sample(
                rng, Xt, unselected, n_out)

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        outliers = scaler.inverse_transform(outliers)

        # in the case no outliers could be generated
        if outliers.shape[0] == 0:
            self._set_outlier_labels(len(X), 0)
            return X

        # Append outliers to original data
        Xt = np.vstack([X, outliers])

        self._set_outlier_labels(len(X), len(outliers))
        return Xt
