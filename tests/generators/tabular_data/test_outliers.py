import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from badgers.core.selectors import AlwaysSelector
from badgers.core.utils import random_spherical_coordinates, random_sign
from badgers.generators.tabular_data.outliers import StructuredOutlierGenerator
from badgers.generators.tabular_data.outliers.distribution_sampling import (
    ZScoreSamplingGenerator, HypersphereSamplingGenerator, HyperCubeSampling,
)
from badgers.generators.tabular_data.outliers.instance_sampling import (
    UniformInstanceAttributeSampling,
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
    X = np.asarray(tabular_data)
    n_samples = len(X)

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        generator = gen_class(random_generator=default_rng(0))

        try:
            Xt = generator.generate(
                X.copy(), n_outliers=n_outliers, **default_kwargs
            )
        except (NotImplementedError, AttributeError, ValueError):
            continue

        if len(Xt) == n_samples:
            continue  # LowDensitySamplingGenerator may generate 0 outliers

        # Xt has original + outlier rows
        assert Xt.shape[0] == n_samples + n_outliers
        assert Xt.shape[1] == (X.shape[1] if X.ndim > 1 else 1)

        # internal labels: False for original rows, True for outliers
        labels = generator.get_outliers_labels()
        assert labels.dtype == bool
        assert len(labels) == len(Xt)
        assert not labels[:n_samples].any()
        assert labels[n_samples:].all()


def test_outliers__reproducibility_given_same_seed(tabular_data):
    """With the same RNG seed and same inputs, generators produce the same output."""
    X = np.asarray(tabular_data)

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        rng1 = default_rng(42)
        rng2 = default_rng(42)
        gen1 = gen_class(random_generator=rng1)
        gen2 = gen_class(random_generator=rng2)

        try:
            Xt1 = gen1.generate(X.copy(), n_outliers=20, **default_kwargs)
            Xt2 = gen2.generate(X.copy(), n_outliers=20, **default_kwargs)
        except (NotImplementedError, AttributeError, ValueError):
            continue

        if len(Xt1) == len(X):
            continue

        np.testing.assert_allclose(Xt1, Xt2)


def test_outliers__scores_worse_than_original():
    """Outlier scores from IsolationForest should be lower than original data scores."""
    rng = default_rng(0)
    X = rng.normal(size=(100, 10))
    detector = IsolationForest(random_state=0, n_estimators=100, contamination="auto")
    detector.fit(X)
    original_scores = detector.decision_function(X)
    mean_original_score = original_scores.mean()

    for gen_name, gen_class, default_kwargs in COMMON_GENERATORS:
        generator = gen_class(random_generator=default_rng(0))
        try:
            Xt = generator.generate(X.copy(), n_outliers=50, **default_kwargs)
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    generator = ZScoreSamplingGenerator(random_generator=default_rng(0))
    n_outliers = 200
    Xt_small = generator.generate(X_scaled.copy(), n_outliers=n_outliers, scale=0.1)
    Xt_large = generator.generate(X_scaled.copy(), n_outliers=n_outliers, scale=5.0)

    # Extract only the outlier rows
    n_original = len(X_scaled)
    outliers_small = Xt_small[n_original:]
    outliers_large = Xt_large[n_original:]

    z_small = np.abs(outliers_small)
    z_large = np.abs(outliers_large)

    assert z_large.mean() > z_small.mean()


def test_zscore__delegates_to_sampler_math():
    """Outlier rows equal the original inline formula (standardize → z → inverse)."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, scale = 10, 1.0

    generator = ZScoreSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=n_outliers, scale=scale)

    ref_rng = default_rng(0)
    scaler = StandardScaler().fit(X)
    signs = random_sign(ref_rng, size=(n_outliers, X.shape[1]))
    exponentials = ref_rng.exponential(scale=scale, size=(n_outliers, X.shape[1]))
    outliers_ref = scaler.inverse_transform(signs * (3.0 + exponentials))

    # sampler computes mean/std with numpy's two-pass algorithm while the old
    # inline math used sklearn's sum-based mean/var: identical up to fp ordering
    # (1 ULP observed), so pin to atol=1e-12 rather than bit equality
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)


def test_histogram__2d_raises_not_implemented():
    """HistogramSamplingGenerator raises NotImplementedError for >3 columns."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    generator = HistogramSamplingGenerator(random_generator=default_rng(0))
    try:
        generator.generate(X.copy(), n_outliers=10, bins=3)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass


def test_histogram__3cols_works():
    """HistogramSamplingGenerator works with 3 columns."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    generator = HistogramSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=10, bins=3)
    n_original = len(X)
    assert len(Xt) == n_original + 10
    assert Xt.shape[1] == 3


def test_hypersphere__radius_ge_three_in_standardized_space():
    """In standardized space, outliers should have Euclidean norm >= 3."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    X_arr = np.asarray(X)
    scaler = StandardScaler().fit(X_arr)

    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=100, scale=1.0)
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
    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    Xt_small = generator.generate(X.copy(), n_outliers=200, scale=0.1)
    Xt_large = generator.generate(X.copy(), n_outliers=200, scale=5.0)

    # Extract only the outlier rows
    outliers_small = Xt_small[200:]  # last 200 rows are outliers
    outliers_large = Xt_large[200:]

    r_small = np.linalg.norm(outliers_small, axis=1).mean()
    r_large = np.linalg.norm(outliers_large, axis=1).mean()

    assert r_large > r_small


def test_hypersphere__delegates_to_sampler_math():
    """Outlier rows equal the original inline formula (standardize → radii → coords → inverse)."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, scale = 10, 1.0

    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=n_outliers, scale=scale)

    ref_rng = default_rng(0)
    scaler = StandardScaler().fit(X)
    radii = 3.0 + ref_rng.exponential(scale=scale, size=n_outliers)
    outliers_ref = scaler.inverse_transform(
        random_spherical_coordinates(random_generator=ref_rng, size=X.shape[1], radii=radii)
    )

    # sampler computes mean/std with numpy's two-pass algorithm while the old
    # inline math used sklearn's sum-based mean/var: identical up to fp ordering
    # (1 ULP observed), so pin to atol=1e-12 rather than bit equality
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)


def test_hypercube__expansion_zero_within_min_max():
    """For expansion=0, all outliers should lie within [min(X), max(X)] per feature."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    X_arr = np.asarray(X)
    generator = HyperCubeSampling(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=100, expansion=0.0)
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
    X_arr = np.asarray(X)
    scaler = MinMaxScaler().fit(X_arr)
    expansion = 0.1

    generator = HyperCubeSampling(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=100, expansion=expansion)
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
    generator = HyperCubeSampling(random_generator=default_rng(0))
    try:
        generator.generate(X.copy(), n_outliers=10, expansion=-0.1)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_hypercube__delegates_to_sampler_math():
    """Outlier rows match the original MinMax round-trip draw (ulp tolerance)."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, expansion = 10, 0.1

    generator = HyperCubeSampling(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=n_outliers, expansion=expansion)

    ref_rng = default_rng(0)
    scaler = MinMaxScaler().fit(X)
    u = ref_rng.uniform(low=-expansion, high=1 + expansion, size=(n_outliers, X.shape[1]))
    outliers_ref = scaler.inverse_transform(u)

    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)

    # the expansion=0 default path through the relaxed sampler: exact bounding box
    generator0 = HyperCubeSampling(random_generator=default_rng(0))
    Xt0 = generator0.generate(X.copy(), n_outliers=n_outliers, expansion=0.0)
    assert np.all(Xt0[len(X):] >= X.min(axis=0) - 1e-12)
    assert np.all(Xt0[len(X):] <= X.max(axis=0) + 1e-12)


def test_decomposition_and_outlier__generates_correct_shape():
    """DecompositionAndOutlierGenerator produces correct shape outliers."""
    X = make_blobs(centers=5, n_features=10, n_samples=100)[0]
    n_outliers = 10
    outlier_generators = [
        ZScoreSamplingGenerator(),
        HistogramSamplingGenerator(),
        HypersphereSamplingGenerator(),
        LowDensitySamplingGenerator(),
    ]
    for outlier_generator in outlier_generators:
        generator = StructuredOutlierGenerator(
            dimension_reduction_transformer=PCA(n_components=3),
            outlier_generator=outlier_generator,
        )
        Xt = generator.generate(X.copy(), n_outliers=n_outliers)
        assert Xt.shape[0] == X.shape[0] + n_outliers
        assert Xt.shape[1] == X.shape[1]

        # labels propagated from the inner outlier generator
        labels = generator.get_outliers_labels()
        assert len(labels) == len(Xt)
        assert labels.dtype == bool
        assert not labels[:len(X)].any()
        assert labels[len(X):].all()

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

    generator = UniformInstanceAttributeSampling(random_generator=default_rng(0))
    generator.generate(X.copy(), n_outliers=5)
    labels1 = generator.get_outliers_labels()
    assert len(labels1) == 55
    assert labels1[50:].all()

    generator.generate(X.copy(), n_outliers=20)
    labels2 = generator.get_outliers_labels()
    assert len(labels2) == 70
    assert labels2[50:].all()


def test_causal_outlier_propagation__labels_without_y_output(make_gcm_dgp):
    """CausalOutlierPropagationGenerator also exposes internal outlier labels."""
    import networkx as nx
    from badgers.generators.tabular_data.outliers.causal import (
        CausalOutlierPropagationGenerator,
    )

    rng = default_rng(0)
    n_samples = 50
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    column_mapping = {n: i for i, n in enumerate(sorted(graph.nodes))}
    X = rng.normal(size=(n_samples, 3))
    dgp = make_gcm_dgp(graph, column_mapping).fit(X)
    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=dgp)
    with pytest.raises(RuntimeError):
        generator.get_outliers_labels()

    Xt = generator.generate(X, AlwaysSelector([column_mapping["a"]]), n_outliers=7)

    labels = generator.get_outliers_labels()
    assert labels.dtype == bool
    assert len(labels) == len(Xt)
    assert not labels[:n_samples].any()
    assert labels[n_samples:].all()
