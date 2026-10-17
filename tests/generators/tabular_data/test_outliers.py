import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from badgers.generators.tabular_data.outliers import DecompositionAndOutlierGenerator
from badgers.generators.tabular_data.outliers.distribution_sampling import (
    ZScoreSamplingGenerator, HypersphereSamplingGenerator, HyperCubeSampling,
)
from badgers.generators.tabular_data.outliers.low_density_sampling import (
    HistogramSamplingGenerator, LowDensitySamplingGenerator,
    IndependentHistogramsGenerator,
)

COMMON_GENERATORS = [
    ("hyper_cube", HyperCubeSampling, {"expansion": 0.0}),
    ("z_score", ZScoreSamplingGenerator, {"scale": 1.0}),
    ("hypersphere", HypersphereSamplingGenerator, {"scale": 1.0}),
    ("histogram", HistogramSamplingGenerator, {"bins": 3, "threshold_low_density": 0.5}),
    ("low_density", LowDensitySamplingGenerator, {}),
    ("independent_histograms", IndependentHistogramsGenerator, {"bins": 3}),
]


def test_outliers__correct_shape_and_labels(tabular_data):
    """For each generator and input type: Xt has original + outliers, and the
    internal labels via get_outliers_labels() are False/True aligned with rows."""
    n_outliers = 10
    X, y = tabular_data
    X_np = np.asarray(X)
    n_samples = len(X_np)

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        generator = gen_class(random_generator=default_rng(0))

        try:
            Xt, returned_y = generator.generate(
                X_np.copy(), y, n_outliers=n_outliers, **default_kwargs
            )
        except (NotImplementedError, AttributeError, ValueError):
            continue

        if len(Xt) == n_samples:
            continue  # LowDensitySamplingGenerator may generate 0 outliers

        # Xt has original + outlier rows
        assert Xt.shape[0] == n_samples + n_outliers
        assert Xt.shape[1] == (X_np.shape[1] if X_np.ndim > 1 else 1)

        # y is a pure pass-through
        if y is not None:
            np.testing.assert_array_equal(returned_y, y)

        # internal labels: False for original rows, True for outliers
        labels = generator.get_outliers_labels()
        assert labels.dtype == bool
        assert len(labels) == len(Xt)
        assert not labels[:n_samples].any()
        assert labels[n_samples:].all()


def test_outliers__reproducibility_given_same_seed(tabular_data):
    """With the same RNG seed and same inputs, generators produce the same output."""
    X, y = tabular_data
    X_np = np.asarray(X)

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        rng1 = default_rng(42)
        rng2 = default_rng(42)
        gen1 = gen_class(random_generator=rng1)
        gen2 = gen_class(random_generator=rng2)

        try:
            Xt1, _ = gen1.generate(X_np.copy(), y, n_outliers=20, **default_kwargs)
            Xt2, _ = gen2.generate(X_np.copy(), y, n_outliers=20, **default_kwargs)
        except (NotImplementedError, AttributeError, ValueError):
            continue

        if len(Xt1) == len(X_np):
            continue

        np.testing.assert_allclose(Xt1, Xt2)


def test_outliers__scores_worse_than_original():
    """Outlier scores from IsolationForest should be lower than original data scores."""
    rng = default_rng(0)
    X = rng.normal(size=(100, 10))
    y = None

    detector = IsolationForest(random_state=0, n_estimators=100, contamination="auto")
    detector.fit(X)
    original_scores = detector.decision_function(X)
    mean_original_score = original_scores.mean()

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        generator = gen_class(random_generator=default_rng(0))
        try:
            Xt, yt = generator.generate(X.copy(), y, n_outliers=50, **default_kwargs)
        except (NotImplementedError, AttributeError, ValueError):
            continue

        Xt = np.asarray(Xt)
        n_original = len(X)
        if len(Xt) == n_original:
            continue
        # Extract only the outlier rows (appended after original data)
        outliers = Xt[n_original:]
        outlier_scores = detector.decision_function(outliers)
        mean_outlier_score = outlier_scores.mean()

        assert mean_outlier_score < mean_original_score, (
            f"Outlier scores for {gen_name} are not worse than original data on average."
        )


def test_zscore__scale_effect_on_zscore_magnitude():
    """Larger scale should, on average, yield larger z-scores."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    y = None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    generator = ZScoreSamplingGenerator(random_generator=default_rng(0))
    n_outliers = 200
    Xt_small, _ = generator.generate(X_scaled.copy(), y, n_outliers=n_outliers, scale=0.1)
    Xt_large, _ = generator.generate(X_scaled.copy(), y, n_outliers=n_outliers, scale=5.0)

    # Extract only the outlier rows
    n_original = len(X_scaled)
    outliers_small = Xt_small[n_original:]
    outliers_large = Xt_large[n_original:]

    z_small = np.abs(outliers_small)
    z_large = np.abs(outliers_large)

    assert z_large.mean() > z_small.mean()


def test_histogram__2d_raises_not_implemented():
    """HistogramSamplingGenerator raises NotImplementedError for >3 columns."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    y = None

    generator = HistogramSamplingGenerator(random_generator=default_rng(0))
    try:
        generator.generate(X.copy(), y, n_outliers=10, bins=3)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass


def test_histogram__3cols_works():
    """HistogramSamplingGenerator works with 3 columns."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = None

    generator = HistogramSamplingGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, n_outliers=10, bins=3)
    n_original = len(X)
    assert len(Xt) == n_original + 10
    assert Xt.shape[1] == 3


def test_hypersphere__radius_ge_three_in_standardized_space():
    """In standardized space, outliers should have Euclidean norm >= 3."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    y = None

    X_arr = np.asarray(X)
    scaler = StandardScaler().fit(X_arr)

    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, n_outliers=100, scale=1.0)
    assert len(Xt) == 200  # 100 original + 100 outliers

    # Extract only the outlier rows
    labels = generator.get_outliers_labels()
    outliers = Xt[labels]
    assert len(outliers) == 100

    outliers_std = scaler.transform(outliers)
    radii = np.linalg.norm(outliers_std, axis=1)
    assert np.all(radii >= 3.0 - 1e-8)


def test_hypersphere__scale_effect_on_radius():
    """Larger scale should, on average, yield larger radii."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    y = None

    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    Xt_small, _ = generator.generate(X.copy(), y, n_outliers=200, scale=0.1)
    Xt_large, _ = generator.generate(X.copy(), y, n_outliers=200, scale=5.0)

    # Extract only the outlier rows
    outliers_small = Xt_small[200:]  # last 200 rows are outliers
    outliers_large = Xt_large[200:]

    r_small = np.linalg.norm(outliers_small, axis=1).mean()
    r_large = np.linalg.norm(outliers_large, axis=1).mean()

    assert r_large > r_small


def test_hypercube__expansion_zero_within_min_max():
    """For expansion=0, all outliers should lie within [min(X), max(X)] per feature."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    y = None

    X_arr = np.asarray(X)
    generator = HyperCubeSampling(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, n_outliers=100, expansion=0.0)
    assert len(Xt) == 200  # 100 original + 100 outliers

    # Extract only the outlier rows
    labels = generator.get_outliers_labels()
    outliers = Xt[labels]
    assert len(outliers) == 100

    mins = X_arr.min(axis=0)
    maxs = X_arr.max(axis=0)
    assert np.all(outliers >= mins - 1e-8)
    assert np.all(outliers <= maxs + 1e-8)


def test_hypercube__expansion_positive_respects_range():
    """For expansion>0, outliers lie within expanded [0-expansion, 1+expansion] hypercube."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    y = None

    X_arr = np.asarray(X)
    scaler = MinMaxScaler().fit(X_arr)
    expansion = 0.1

    generator = HyperCubeSampling(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y, n_outliers=100, expansion=expansion)
    assert len(Xt) == 200  # 100 original + 100 outliers

    # Extract only the outlier rows
    labels = generator.get_outliers_labels()
    outliers = Xt[labels]
    assert len(outliers) == 100

    outliers_scaled = scaler.transform(outliers)
    assert np.all(outliers_scaled >= 0 - expansion - 1e-8)
    assert np.all(outliers_scaled <= 1 + expansion + 1e-8)


def test_hypercube__expansion_negative_raises():
    """Negative expansion should trigger an assertion error."""
    rng = default_rng(0)
    X = rng.normal(size=(100, 10))
    y = None

    generator = HyperCubeSampling(random_generator=default_rng(0))
    try:
        generator.generate(X.copy(), y, n_outliers=10, expansion=-0.1)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_decomposition_and_outlier__generates_correct_shape():
    """DecompositionAndOutlierGenerator produces correct shape outliers."""
    X, y = make_blobs(centers=5, n_features=10, n_samples=100)
    n_outliers = 10
    outlier_generators = [
        ZScoreSamplingGenerator(),
        HistogramSamplingGenerator(),
        HypersphereSamplingGenerator(),
        LowDensitySamplingGenerator(),
    ]
    for outlier_generator in outlier_generators:
        generator = DecompositionAndOutlierGenerator(
            dimension_reduction_transformer=PCA(n_components=3),
            outlier_generator=outlier_generator,
        )
        Xt, returned_y = generator.generate(X.copy(), y, n_outliers=n_outliers)
        assert Xt.shape[0] == X.shape[0] + n_outliers
        assert Xt.shape[1] == X.shape[1]

        # labels propagated from the inner outlier generator
        labels = generator.get_outliers_labels()
        assert len(labels) == len(Xt)
        assert labels.dtype == bool

        # Extract only the outlier rows
        outliers = Xt[labels]
        assert outliers.shape[0] == n_outliers


def test_outliers__labels_before_generate_raises():
    """get_outliers_labels() raises RuntimeError if generate() was not called."""
    generator = ZScoreSamplingGenerator(random_generator=default_rng(0))
    with pytest.raises(RuntimeError):
        generator.get_outliers_labels()


def test_outliers__labels_reflect_last_generation():
    """Labels are aligned with the LAST generated output, not the first."""
    rng = default_rng(0)
    X = rng.normal(size=(50, 3))

    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    generator.generate(X.copy(), n_outliers=5)
    labels1 = generator.get_outliers_labels()
    assert len(labels1) == 55
    assert labels1[50:].all()

    generator.generate(X.copy(), n_outliers=20)
    labels2 = generator.get_outliers_labels()
    assert len(labels2) == 70
    assert labels2[50:].all()


def test_causal_outlier_propagation__labels_and_y_passthrough():
    """CausalOutlierPropagationGenerator also exposes internal outlier labels."""
    import networkx as nx
    from badgers.core.dgp import OlsDGP
    from badgers.generators.tabular_data.outliers.causal import (
        CausalOutlierPropagationGenerator,
    )

    rng = default_rng(0)
    n_samples = 50
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    column_mapping = {n: i for i, n in enumerate(sorted(graph.nodes))}
    X = rng.normal(size=(n_samples, 3))
    y = rng.integers(0, 2, size=n_samples)

    dgp = OlsDGP(graph, column_mapping).fit(X)
    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=dgp)
    with pytest.raises(RuntimeError):
        generator.get_outliers_labels()

    Xt, returned_y = generator.generate(
        X, y=y, perturbation_nodes=["a"], n_outliers=7
    )

    # y is a pure pass-through
    np.testing.assert_array_equal(returned_y, y)

    labels = generator.get_outliers_labels()
    assert labels.dtype == bool
    assert len(labels) == len(Xt)
    assert not labels[:n_samples].any()
    assert labels[n_samples:].all()
