import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from badgers.core.sampling import (
    HypersphereSampler,
    UniformOutOfDistributionSampler,
    ZScoreSampler,
)
from badgers.core.selectors import AlwaysSelector, CountSelector, fired
from badgers.core.utils import random_sign
from badgers.generators.tabular_data.outliers._firing import iter_firing_groups
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
    X = pd.DataFrame(rng.normal(size=(300, 10)), columns=[f"col{i}" for i in range(10)])
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
    """Outlier rows equal the selector-flow replay on the same stream.

    Pins the draw order of the new flow: CountSelector row draw, then
    fired()'s realization draws, then the sampler's signs + exponentials.
    The sampler MATH exactness itself is pinned in tests/core/test_sampling.py.
    """
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, scale = 10, 1.0

    generator = ZScoreSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=n_outliers, scale=scale)

    ref_rng = default_rng(0)
    ref_selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=ref_rng)
    ref_field = fired(X, ref_selector, ref_rng)
    ref_targets = list(range(X.shape[1]))
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, fired_cols in iter_firing_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, fired_cols)] = ZScoreSampler(scale=scale).sample(
            ref_rng, X, fired_cols, group_rows.size)
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)


def test_zscore__selector_partial_fire_keeps_seed_values():
    """A column not fired keeps the seed row's own value (oracle)."""
    rng = default_rng(0)
    X = rng.normal(size=(40, 3))
    mask0 = np.zeros(40, dtype=bool)
    mask0[:5] = True
    generator = ZScoreSamplingGenerator(random_generator=default_rng(1))
    Xt = generator.generate(X.copy(), selector=lambda X_, targets=None: {0: mask0})
    outliers = Xt[len(X):]
    assert outliers.shape == (5, 3)
    labels = generator.get_outliers_labels()
    assert not labels[:len(X)].any()
    assert labels[len(X):].all()
    # unfired columns keep the seed row's values exactly
    np.testing.assert_array_equal(outliers[:, 1:], X[:5, 1:])
    np.testing.assert_array_equal(outliers[:, 2:], X[:5, 2:])
    # fired column is beyond 3 sigma of its column
    stds = X.std(axis=0)
    assert np.all(np.abs(outliers[:, 0] - X[:, 0].mean()) > 3.0 * stds[0])


def test_zscore__explicit_count_selector():
    """An explicit CountSelector owns count and targets."""
    X = default_rng(0).normal(size=(40, 3))
    generator = ZScoreSamplingGenerator(random_generator=default_rng(2))
    selector = CountSelector([0], count=6, rng=default_rng(3))
    ref_selector = CountSelector([0], count=6, rng=default_rng(3))
    Xt = generator.generate(X.copy(), selector=selector)
    assert Xt.shape == (46, 3)
    labels = generator.get_outliers_labels()
    assert labels[:40].sum() == 0 and labels[40:].all()
    outliers = Xt[40:]
    # the fired rows (selector's own draw, replayed on an equal-seeded
    # selector) are the seeds: the unfired columns keep the seed values
    seeds = np.flatnonzero(ref_selector(X, targets=[0])[0])
    assert seeds.size == 6
    np.testing.assert_array_equal(outliers[:, 1:], X[seeds, 1:])
    stds = X.std(axis=0)
    assert np.all(np.abs(outliers[:, 0] - X[:, 0].mean()) > 3.0 * stds[0])


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
    X = pd.DataFrame(rng.normal(size=(300, 10)), columns=[f"col{i}" for i in range(10)])
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
    """Outlier rows equal the selector-flow replay on the same stream."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, scale = 10, 1.0

    generator = HypersphereSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=n_outliers, scale=scale)

    ref_rng = default_rng(0)
    ref_selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=ref_rng)
    ref_field = fired(X, ref_selector, ref_rng)
    ref_targets = list(range(X.shape[1]))
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, fired_cols in iter_firing_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, fired_cols)] = HypersphereSampler(scale=scale).sample(
            ref_rng, X, fired_cols, group_rows.size)
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
    """Outlier rows equal the selector-flow replay on the same stream."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, expansion = 10, 0.1

    generator = HyperCubeSampling(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=n_outliers, expansion=expansion)

    ref_rng = default_rng(0)
    ref_selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=ref_rng)
    ref_field = fired(X, ref_selector, ref_rng)
    ref_targets = list(range(X.shape[1]))
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, fired_cols in iter_firing_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, fired_cols)] = UniformOutOfDistributionSampler(
            expansion=expansion).sample(ref_rng, X, fired_cols, group_rows.size)
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

# --- UniformInstanceAttributeSampling (selector surface) ---

def test_instance_sampling__bit_identical_delegation():
    """Full-width delegation keeps today's inline per-cell bootstrap values."""
    rng = default_rng(42)
    X = rng.normal(size=(30, 4))
    generator = UniformInstanceAttributeSampling(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=5)
    ref = X[default_rng(0).integers(0, 30, size=(5, 4)), np.arange(4)]
    np.testing.assert_array_equal(Xt[len(X):], ref)


def test_instance_sampling__n_outliers_capped_at_row_count():
    X = np.arange(20.0).reshape(10, 2)
    generator = UniformInstanceAttributeSampling(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="must not exceed"):
        generator.generate(X.copy(), n_outliers=11)


def test_instance_sampling__selector_is_ignored_with_warning():
    rng = default_rng(0)
    X = rng.normal(size=(30, 3))
    generator = UniformInstanceAttributeSampling(random_generator=default_rng(1))
    with pytest.warns(UserWarning, match="selector is ignored"):
        Xt = generator.generate(
            X.copy(), n_outliers=5,
            selector=CountSelector([0], count=3, rng=default_rng(2)))
    assert Xt.shape == (35, 3)
    labels = generator.get_outliers_labels()
    assert labels[:30].sum() == 0
    assert labels[30:].all()


# --- shared direct-four selector-flow suite ---

DIRECT_FOUR = [
    (HyperCubeSampling, {"expansion": 0.1}),
    (ZScoreSamplingGenerator, {"scale": 1.0}),
    (HypersphereSamplingGenerator, {"scale": 1.0}),
]


@pytest.mark.parametrize("gen_class,kwargs", DIRECT_FOUR)
def test_direct_four__n_outliers_capped_at_row_count(gen_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="exceeds the number of rows"):
        gen_class(random_generator=default_rng(0)).generate(
            X.copy(), n_outliers=11, **kwargs)


@pytest.mark.parametrize("gen_class,kwargs", DIRECT_FOUR)
def test_direct_four__negative_n_outliers_raises(gen_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="count must be >= 0"):
        gen_class(random_generator=default_rng(0)).generate(
            X.copy(), n_outliers=-1, **kwargs)


@pytest.mark.parametrize("gen_class,kwargs", DIRECT_FOUR)
def test_direct_four__n_outliers_with_selector_raises(gen_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="only the default count"):
        gen_class(random_generator=default_rng(0)).generate(
            X.copy(), n_outliers=5,
            selector=CountSelector([0], count=3, rng=default_rng(1)), **kwargs)


@pytest.mark.parametrize("gen_class,kwargs", DIRECT_FOUR)
def test_direct_four__n_outliers_zero_is_noop(gen_class, kwargs):
    X = np.arange(40.0).reshape(20, 2)
    generator = gen_class(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), n_outliers=0, **kwargs)
    np.testing.assert_array_equal(Xt, X)
    assert not generator.get_outliers_labels().any()


@pytest.mark.parametrize("gen_class,kwargs", DIRECT_FOUR)
def test_direct_four__default_count_is_ten(gen_class, kwargs):
    X = np.arange(40.0).reshape(20, 2)
    generator = gen_class(random_generator=default_rng(0))
    Xt = generator.generate(X.copy(), **kwargs)
    assert Xt.shape == (30, 2)
    labels = generator.get_outliers_labels()
    assert labels[:20].sum() == 0 and labels[20:].all()


@pytest.mark.parametrize("gen_class,kwargs", DIRECT_FOUR)
def test_direct_four__selector_oracle_unfired_cells(gen_class, kwargs):
    """Unfired cells of an appended row equal the seed row's values."""
    rng = default_rng(0)
    X = rng.normal(size=(40, 3))
    mask0 = np.zeros(40, dtype=bool)
    mask0[:5] = True
    generator = gen_class(random_generator=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector=lambda X_, targets=None: {0: mask0}, **kwargs)
    outliers = Xt[len(X):]
    assert outliers.shape == (5, 3)
    np.testing.assert_array_equal(outliers[:, 1:], X[:5, 1:])
    np.testing.assert_array_equal(outliers[:, 2:], X[:5, 2:])
    # fired cells were overwritten (not seed copies)
    assert not np.array_equal(outliers[:, 0], X[:5, 0])
