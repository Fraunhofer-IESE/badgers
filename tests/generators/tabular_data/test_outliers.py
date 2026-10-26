import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity as _KernelDensityEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from badgers.core.sampling import (
    HypersphereSampler,
    IndependentHistogramSampler,
    JointHistogramSampler,
    KernelDensitySampler,
    TreeCellSampler,
    UniformOutOfDistributionSampler,
    ZScoreSampler,
    create_out_of_distribution_sampler,
)
from badgers.core.selectors import CountSelector, get_selected_fields
from badgers.generators.tabular_data.outliers._selection import iter_selection_groups
from badgers.generators.tabular_data.outliers import StructuredOutlierGenerator
from badgers.generators.tabular_data.outliers.instance_sampling import (
    UniformInstanceAttributeSampling,
)
from badgers.generators.tabular_data.outliers.sampling import OutlierSamplingGenerator


COMMON_GENERATORS = [
    ("uniform", "uniform", {"expansion": 0.0}),
    ("zscore", "zscore", {"scale": 1.0}),
    ("hypersphere", "hypersphere", {"scale": 1.0}),
    ("histogram", "histogram", {"bins": 3, "threshold_low_density": 0.5}),
    ("low_density", "low_density", {}),
    ("independent_histogram", "independent_histogram", {"bins": 3}),
    ("tree", "tree", {"leaf_size": 4}),
]


def test_outliers__correct_shape_and_labels(tabular_data):
    """For each sampler and input type: Xt has original + outliers, and the
    internal labels via get_outliers_labels() are False/True aligned with rows."""
    n_outliers = 10
    X = np.asarray(tabular_data)
    n_samples = len(X)

    for gen_name, ood_name, ood_kwargs in COMMON_GENERATORS:
        generator = OutlierSamplingGenerator(random_generator=default_rng(0))
        selector = CountSelector(
            list(range(X.shape[1])), count=n_outliers, rng=default_rng(1))

        try:
            Xt = generator.generate(
                X.copy(), selector,
                out_of_distribution_sampler=create_out_of_distribution_sampler(
                    ood_name, **ood_kwargs))
        except (NotImplementedError, AttributeError, ValueError):
            continue

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
    """With the same RNG seeds and same inputs, the generator reproduces output."""
    X = np.asarray(tabular_data)

    for gen_name, ood_name, ood_kwargs in COMMON_GENERATORS:
        gen1 = OutlierSamplingGenerator(random_generator=default_rng(42))
        gen2 = OutlierSamplingGenerator(random_generator=default_rng(42))
        sel1 = CountSelector(list(range(X.shape[1])), count=20, rng=default_rng(7))
        sel2 = CountSelector(list(range(X.shape[1])), count=20, rng=default_rng(7))

        try:
            Xt1 = gen1.generate(
                X.copy(), sel1,
                out_of_distribution_sampler=create_out_of_distribution_sampler(
                    ood_name, **ood_kwargs))
            Xt2 = gen2.generate(
                X.copy(), sel2,
                out_of_distribution_sampler=create_out_of_distribution_sampler(
                    ood_name, **ood_kwargs))
        except (NotImplementedError, AttributeError, ValueError):
            continue

        np.testing.assert_allclose(Xt1, Xt2)


def test_outliers__scores_worse_than_original():
    """Outlier scores from IsolationForest should be lower than original data scores.

    Uses a low feature count: the hypersphere sampler's anomaly is radial
    (a large joint norm), which spreads to ~1 sigma per coordinate and is only
    separable by an axis-aligned IsolationForest when the dimension is small.
    """
    rng = default_rng(0)
    X = rng.normal(size=(100, 5))
    detector = IsolationForest(random_state=0, n_estimators=100, contamination="auto")
    detector.fit(X)
    original_scores = detector.decision_function(X)
    mean_original_score = original_scores.mean()

    for gen_name, ood_name, ood_kwargs in COMMON_GENERATORS:
        generator = OutlierSamplingGenerator(random_generator=default_rng(0))
        selector = CountSelector(
            list(range(X.shape[1])), count=50, rng=default_rng(1))
        try:
            Xt = generator.generate(
                X.copy(), selector,
                out_of_distribution_sampler=create_out_of_distribution_sampler(
                    ood_name, **ood_kwargs))
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

    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    n_outliers = 200
    Xt_small = generator.generate(
        X_scaled.copy(),
        CountSelector(range(10), count=n_outliers, rng=default_rng(1)),
        out_of_distribution_sampler=ZScoreSampler(scale=0.1))
    Xt_large = generator.generate(
        X_scaled.copy(),
        CountSelector(range(10), count=n_outliers, rng=default_rng(1)),
        out_of_distribution_sampler=ZScoreSampler(scale=5.0))

    # Extract only the outlier rows
    n_original = len(X_scaled)
    outliers_small = Xt_small[n_original:]
    outliers_large = Xt_large[n_original:]

    z_small = np.abs(outliers_small)
    z_large = np.abs(outliers_large)

    assert z_large.mean() > z_small.mean()


def test_zscore__delegates_to_sampler_math():
    """Outlier rows equal the selector-flow replay on the same stream.

    Pins the draw order of the new flow: CountSelector row draw (selector's
    own rng), then get_selected_fields()'s realization draws, then the
    sampler's signs + exponentials (all on the generator's rng).
    The sampler MATH exactness itself is pinned in tests/core/test_sampling.py.
    """
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, scale = 10, 1.0

    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler=ZScoreSampler(scale=scale))

    ref_rng = default_rng(0)  # the generator's stream: realization + sampler
    ref_selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(1))
    ref_field = get_selected_fields(X, ref_selector, ref_rng)
    ref_targets = list(range(X.shape[1]))
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, selected_cols in iter_selection_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, selected_cols)] = ZScoreSampler(scale=scale).sample(
            ref_rng, X, selected_cols, group_rows.size)
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)


def test_zscore__selector_partial_selection_keeps_seed_values():
    """A column not selected keeps the seed row's own value (oracle)."""
    rng = default_rng(0)
    X = rng.normal(size=(40, 3))
    mask0 = np.zeros(40, dtype=bool)
    mask0[:5] = True
    generator = OutlierSamplingGenerator(random_generator=default_rng(1))
    Xt = generator.generate(
        X.copy(),
        lambda X_, targets=None: {0: mask0},
        out_of_distribution_sampler=ZScoreSampler(scale=1.0))
    outliers = Xt[len(X):]
    assert outliers.shape == (5, 3)
    labels = generator.get_outliers_labels()
    assert not labels[:len(X)].any()
    assert labels[len(X):].all()
    # unselected columns keep the seed row's values exactly
    np.testing.assert_array_equal(outliers[:, 1:], X[:5, 1:])
    np.testing.assert_array_equal(outliers[:, 2:], X[:5, 2:])
    # selected column is beyond 3 sigma of its column
    stds = X.std(axis=0)
    assert np.all(np.abs(outliers[:, 0] - X[:, 0].mean()) > 3.0 * stds[0])


def test_zscore__explicit_count_selector():
    """An explicit CountSelector owns count and targets."""
    X = default_rng(0).normal(size=(40, 3))
    generator = OutlierSamplingGenerator(random_generator=default_rng(2))
    selector = CountSelector([0], count=6, rng=default_rng(3))
    ref_selector = CountSelector([0], count=6, rng=default_rng(3))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler=ZScoreSampler(scale=1.0))
    assert Xt.shape == (46, 3)
    labels = generator.get_outliers_labels()
    assert labels[:40].sum() == 0 and labels[40:].all()
    outliers = Xt[40:]
    # the selected rows (selector's own draw, replayed on an equal-seeded
    # selector) are the seeds: the unselected columns keep the seed values
    seeds = np.flatnonzero(ref_selector(X, targets=[0])[0])
    assert seeds.size == 6
    np.testing.assert_array_equal(outliers[:, 1:], X[seeds, 1:])
    stds = X.std(axis=0)
    assert np.all(np.abs(outliers[:, 0] - X[:, 0].mean()) > 3.0 * stds[0])


def test_histogram__2d_raises_not_implemented():
    """JointHistogramSampler raises NotImplementedError for >5 columns."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(range(10), count=10, rng=default_rng(1))
    try:
        generator.generate(
            X.copy(), selector, out_of_distribution_sampler=JointHistogramSampler(bins=3))
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass


def test_histogram__3cols_works():
    """JointHistogramSampler works with 3 columns."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(),
        CountSelector(range(3), count=10, rng=default_rng(1)),
        out_of_distribution_sampler=JointHistogramSampler(bins=3,
                                                          threshold_low_density=0.5))
    n_original = len(X)
    assert len(Xt) == n_original + 10
    assert Xt.shape[1] == 3


def test_hypersphere__radius_ge_three_in_standardized_space():
    """In standardized space, outliers should have Euclidean norm >= 3."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    X_arr = np.asarray(X)
    scaler = StandardScaler().fit(X_arr)

    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(),
        CountSelector(range(10), count=100, rng=default_rng(1)),
        out_of_distribution_sampler=HypersphereSampler(scale=1.0))
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
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    Xt_small = generator.generate(
        X.copy(),
        CountSelector(range(10), count=200, rng=default_rng(1)),
        out_of_distribution_sampler=HypersphereSampler(scale=0.1))
    Xt_large = generator.generate(
        X.copy(),
        CountSelector(range(10), count=200, rng=default_rng(1)),
        out_of_distribution_sampler=HypersphereSampler(scale=5.0))

    # Extract only the outlier rows
    outliers_small = Xt_small[300:]  # last 200 rows are outliers
    outliers_large = Xt_large[300:]

    r_small = np.linalg.norm(outliers_small, axis=1).mean()
    r_large = np.linalg.norm(outliers_large, axis=1).mean()

    assert r_large > r_small


def test_hypersphere__delegates_to_sampler_math():
    """Outlier rows equal the selector-flow replay on the same stream."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, scale = 10, 1.0

    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler=HypersphereSampler(scale=scale))

    ref_rng = default_rng(0)
    ref_selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(1))
    ref_field = get_selected_fields(X, ref_selector, ref_rng)
    ref_targets = list(range(X.shape[1]))
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, selected_cols in iter_selection_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, selected_cols)] = HypersphereSampler(scale=scale).sample(
            ref_rng, X, selected_cols, group_rows.size)
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)


def test_hypercube__expansion_zero_within_min_max():
    """For expansion=0, all outliers should lie within [min(X), max(X)] per feature."""
    rng = default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 10)), columns=[f"col{i}" for i in range(10)])
    X_arr = np.asarray(X)
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(),
        CountSelector(range(10), count=100, rng=default_rng(1)),
        out_of_distribution_sampler=UniformOutOfDistributionSampler(expansion=0.0))
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

    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    Xt = generator.generate(
        X.copy(),
        CountSelector(range(10), count=100, rng=default_rng(1)),
        out_of_distribution_sampler=UniformOutOfDistributionSampler(expansion=expansion))
    assert len(Xt) == 200  # 100 original + 100 outliers

    # Extract only the outlier rows
    labels = generator.get_outliers_labels()
    outliers = Xt[labels]
    assert len(outliers) == 100

    outliers_scaled = scaler.transform(outliers)
    assert np.all(outliers_scaled >= 0 - expansion - 1e-8)
    assert np.all(outliers_scaled <= 1 + expansion + 1e-8)


def test_hypercube__expansion_negative_raises():
    """Negative expansion raises ValueError (no bare asserts: validation
    must survive python -O)."""
    X = default_rng(0).normal(size=(100, 10))
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(range(10), count=10, rng=default_rng(1))
    with pytest.raises(ValueError, match="expansion"):
        generator.generate(
            X.copy(), selector,
            out_of_distribution_sampler=UniformOutOfDistributionSampler(expansion=-0.1))


def test_hypercube__delegates_to_sampler_math():
    """Outlier rows equal the selector-flow replay on the same stream."""
    rng = default_rng(42)
    X = rng.normal(size=(50, 4))
    n_outliers, expansion = 10, 0.1

    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector,
        out_of_distribution_sampler=UniformOutOfDistributionSampler(expansion=expansion))

    ref_rng = default_rng(0)
    ref_selector = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(1))
    ref_field = get_selected_fields(X, ref_selector, ref_rng)
    ref_targets = list(range(X.shape[1]))
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, selected_cols in iter_selection_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, selected_cols)] = UniformOutOfDistributionSampler(
            expansion=expansion).sample(ref_rng, X, selected_cols, group_rows.size)
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)

    # the expansion=0 path through the sampler: exact bounding box
    generator0 = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector0 = CountSelector(range(X.shape[1]), count=n_outliers, rng=default_rng(5))
    Xt0 = generator0.generate(
        X.copy(), selector0,
        out_of_distribution_sampler=UniformOutOfDistributionSampler(expansion=0.0))
    assert np.all(Xt0[len(X):] >= X.min(axis=0) - 1e-12)
    assert np.all(Xt0[len(X):] <= X.max(axis=0) + 1e-12)


def test_decomposition_and_outlier__generates_correct_shape():
    """DecompositionAndOutlierGenerator produces correct shape outliers."""
    X = make_blobs(centers=5, n_features=10, n_samples=100)[0]
    n_outliers = 10
    samplers = [
        ZScoreSampler(scale=1.0),
        JointHistogramSampler(bins=10, threshold_low_density=0.1),
        HypersphereSampler(scale=1.0),
        KernelDensitySampler(),
        TreeCellSampler(),
    ]
    for ood in samplers:
        outlier_generator = OutlierSamplingGenerator(
            random_generator=np.random.default_rng(0))
        generator = StructuredOutlierGenerator(
            dimension_reduction_transformer=PCA(n_components=3),
            outlier_generator=outlier_generator,
        )
        # the reduced space is 3-D; the selector operates in that space
        selector = CountSelector(range(3), count=n_outliers,
                                 rng=np.random.default_rng(0))
        Xt = generator.generate(
            X.copy(), selector=selector, out_of_distribution_sampler=ood)
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
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
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

    Xt = generator.generate(
        X, CountSelector([column_mapping["a"]], count=7, rng=rng))

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


# --- shared direct-four sampler-flow suite ---

DIRECT_FOUR = [
    ("uniform", UniformOutOfDistributionSampler, {"expansion": 0.1}),
    ("zscore", ZScoreSampler, {"scale": 1.0}),
    ("hypersphere", HypersphereSampler, {"scale": 1.0}),
]


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", DIRECT_FOUR)
def test_direct_four__selector_count_capped_at_row_count(gen_name, sampler_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(list(range(X.shape[1])), count=11, rng=default_rng(1))
    with pytest.raises(ValueError, match="exceeds the number of rows"):
        generator.generate(
            X.copy(), selector, out_of_distribution_sampler=sampler_class(**kwargs))


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", DIRECT_FOUR)
def test_direct_four__negative_count_raises(gen_name, sampler_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="count must be >= 0"):
        CountSelector(list(range(X.shape[1])), count=-1, rng=default_rng(1))


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", DIRECT_FOUR)
def test_direct_four__selector_none_raises(gen_name, sampler_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="selector is required"):
        generator.generate(
            X.copy(), None, out_of_distribution_sampler=sampler_class(**kwargs))


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", DIRECT_FOUR)
def test_direct_four__count_zero_is_noop(gen_name, sampler_class, kwargs):
    X = np.arange(40.0).reshape(20, 2)
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(list(range(X.shape[1])), count=0, rng=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler=sampler_class(**kwargs))
    np.testing.assert_array_equal(Xt, X)
    assert not generator.get_outliers_labels().any()


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", DIRECT_FOUR)
def test_direct_four__selector_oracle_unselected_cells(gen_name, sampler_class, kwargs):
    """Unselected cells of an appended row equal the seed row's values."""
    rng = default_rng(0)
    X = rng.normal(size=(40, 3))
    mask0 = np.zeros(40, dtype=bool)
    mask0[:5] = True
    generator = OutlierSamplingGenerator(random_generator=default_rng(1))
    Xt = generator.generate(
        X.copy(), lambda X_, targets=None: {0: mask0},
        out_of_distribution_sampler=sampler_class(**kwargs))
    outliers = Xt[len(X):]
    assert outliers.shape == (5, 3)
    np.testing.assert_array_equal(outliers[:, 1:], X[:5, 1:])
    np.testing.assert_array_equal(outliers[:, 2:], X[:5, 2:])
    # selected cells were overwritten (not seed copies)
    assert not np.array_equal(outliers[:, 0], X[:5, 0])


# --- low-density trio selector surface ---

def test_independent_histograms__selector_keys_choose_perturbed_columns():
    """The selector's columns are the perturbed cells; the selected rows are
    the seeds, so unselected columns keep the SEED row's own value."""
    rng = default_rng(0)
    X = rng.normal(size=(60, 3))
    generator = OutlierSamplingGenerator(random_generator=default_rng(1))
    selector = CountSelector([0], count=3, rng=default_rng(2))
    Xt = generator.generate(
        X.copy(), selector,
        out_of_distribution_sampler=IndependentHistogramSampler(bins=5))
    assert Xt.shape == (63, 3)
    labels = generator.get_outliers_labels()
    assert labels[:60].sum() == 0 and labels[60:].all()
    # unselected columns (1, 2) keep the seed row's values
    ref_selector = CountSelector([0], count=3, rng=default_rng(2))
    seeds = np.flatnonzero(ref_selector(X, targets=[0])[0])
    np.testing.assert_array_equal(Xt[60:, 1], X[seeds, 1])
    np.testing.assert_array_equal(Xt[60:, 2], X[seeds, 2])
    # selected column 0 was overwritten (not the seed values)
    assert not np.array_equal(Xt[60:, 0], X[seeds, 0])


def test_independent_histograms__default_ood_sampler_is_zscore():
    """out_of_distribution_sampler=None defaults to the zscore sampler."""
    X = default_rng(0).normal(size=(60, 3))
    gen1 = OutlierSamplingGenerator(random_generator=default_rng(1))
    sel1 = CountSelector(list(range(3)), count=10, rng=default_rng(2))
    Xt1 = gen1.generate(X.copy(), sel1)  # default -> zscore
    gen2 = OutlierSamplingGenerator(random_generator=default_rng(1))
    sel2 = CountSelector(list(range(3)), count=10, rng=default_rng(2))
    Xt2 = gen2.generate(
        X.copy(), sel2, out_of_distribution_sampler="zscore")
    np.testing.assert_array_equal(Xt1, Xt2)


# --- delegation parity: the generator reproduces the shared samplers ---


def test_independent_histograms_generator__delegates_to_sampler():
    """Appended rows equal a replay of the exact generator rng flow."""
    X = default_rng(0).normal(size=(80, 2))
    generator = OutlierSamplingGenerator(random_generator=default_rng(1))
    selector = CountSelector([0], count=12, rng=default_rng(2))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler=IndependentHistogramSampler(bins=5))

    ref_rng = default_rng(1)
    ref_selector = CountSelector([0], count=12, rng=default_rng(2))
    ref_field = get_selected_fields(X, ref_selector, ref_rng)
    ref_targets = [0]
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, selected_cols in iter_selection_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, selected_cols)] = IndependentHistogramSampler(bins=5).sample(
            ref_rng, X, selected_cols, group_rows.size)

    np.testing.assert_array_equal(Xt[:len(X)], X)
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)
    # unselected column keeps the seed value
    np.testing.assert_array_equal(Xt[len(X):, 1], X[ref_seeds, 1])
    labels = generator.get_outliers_labels()
    assert labels.shape == (92,)
    assert labels[:80].sum() == 0 and labels[80:].all()


def test_histogram_sampling_generator__delegates_to_sampler():
    """Appended rows equal a replay of the exact generator rng flow."""
    X = default_rng(0).normal(size=(80, 2))
    generator = OutlierSamplingGenerator(random_generator=default_rng(1))
    selector = CountSelector([0], count=12, rng=default_rng(2))
    Xt = generator.generate(
        X.copy(), selector,
        out_of_distribution_sampler=JointHistogramSampler(
            bins=6, threshold_low_density=0.25))

    ref_rng = default_rng(1)
    ref_selector = CountSelector([0], count=12, rng=default_rng(2))
    ref_field = get_selected_fields(X, ref_selector, ref_rng)
    ref_targets = [0]
    ref_seeds = np.flatnonzero(ref_field.any(axis=1))
    outliers_ref = X[ref_seeds].astype(np.float64)
    for group_rows, selected_cols in iter_selection_groups(ref_field, ref_targets, ref_seeds):
        outliers_ref[np.ix_(group_rows, selected_cols)] = JointHistogramSampler(
            bins=6, threshold_low_density=0.25).sample(
            ref_rng, X, selected_cols, group_rows.size)

    np.testing.assert_array_equal(Xt[:len(X)], X)
    np.testing.assert_allclose(Xt[len(X):], outliers_ref, rtol=0, atol=1e-12)
    # unselected column keeps the seed value
    np.testing.assert_array_equal(Xt[len(X):, 1], X[ref_seeds, 1])
    labels = generator.get_outliers_labels()
    assert labels.shape == (92,)
    assert labels[:80].sum() == 0 and labels[80:].all()


# --- low-density trio count/noop surface ---

TRIO = [
    ("independent_histogram", IndependentHistogramSampler, {"bins": 3}),
    ("histogram", JointHistogramSampler, {"bins": 3, "threshold_low_density": 0.5}),
    ("low_density", KernelDensitySampler, {}),
    ("tree", TreeCellSampler, {"leaf_size": 4}),
]


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", TRIO)
def test_trio__selector_count_capped_at_row_count(gen_name, sampler_class, kwargs):
    X = np.arange(20.0).reshape(10, 2)
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(list(range(X.shape[1])), count=11, rng=default_rng(1))
    with pytest.raises(ValueError, match="exceeds the number of rows"):
        generator.generate(
            X.copy(), selector, out_of_distribution_sampler=sampler_class(**kwargs))


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", TRIO)
def test_trio__negative_count_raises(gen_name, sampler_class, kwargs):
    with pytest.raises(ValueError, match="count must be >= 0"):
        CountSelector([0], count=-1, rng=default_rng(1))


@pytest.mark.parametrize("gen_name,sampler_class,kwargs", TRIO)
def test_trio__count_zero_is_noop(gen_name, sampler_class, kwargs):
    X = np.arange(40.0).reshape(20, 2)
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(list(range(X.shape[1])), count=0, rng=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler=sampler_class(**kwargs))
    np.testing.assert_array_equal(Xt, X)
    assert not generator.get_outliers_labels().any()


# --- new OutlierSamplingGenerator contract ---

def test_sampling__selector_none_raises():
    """A missing selector is a hard error: it owns count and targets."""
    X = default_rng(0).normal(size=(20, 3))
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    with pytest.raises(ValueError, match="selector is required"):
        generator.generate(X.copy(), None)


def test_sampling__ood_sampler_str_and_instance():
    """out_of_distribution_sampler accepts a registry string or an instance."""
    X = default_rng(0).normal(size=(40, 3))
    for sampler in ("zscore", "histogram",
                    ZScoreSampler(scale=1.0),
                    JointHistogramSampler(bins=3, threshold_low_density=0.5)):
        generator = OutlierSamplingGenerator(random_generator=default_rng(0))
        selector = CountSelector(list(range(3)), count=8, rng=default_rng(1))
        Xt = generator.generate(
            X.copy(), selector, out_of_distribution_sampler=sampler)
        assert Xt.shape == (48, 3)
        labels = generator.get_outliers_labels()
        assert labels[:40].sum() == 0 and labels[40:].all()


def test_sampling__ood_sampler_invalid_type_raises():
    """A non-str / non-sampler out_of_distribution_sampler is rejected."""
    X = default_rng(0).normal(size=(20, 3))
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector([0], count=3, rng=default_rng(1))
    with pytest.raises(ValueError, match="OutOfDistributionSampler instance or str"):
        generator.generate(
            X.copy(), selector, out_of_distribution_sampler=42)


def test_sampling__create_low_density_resolves():
    """create_out_of_distribution_sampler resolves the low_density entry."""
    sampler = create_out_of_distribution_sampler("low_density", grid_budget=50)
    assert isinstance(sampler, KernelDensitySampler)
    assert sampler.grid_budget == 50


def test_sampling__create_tree_resolves():
    """create_out_of_distribution_sampler resolves the tree entry."""
    sampler = create_out_of_distribution_sampler("tree", leaf_size=5)
    assert isinstance(sampler, TreeCellSampler)
    assert sampler.leaf_size == 5


def test_sampling__low_density_string_end_to_end():
    """The 'low_density' registry name works end-to-end in generate()."""
    X = default_rng(0).normal(size=(60, 3))
    generator = OutlierSamplingGenerator(random_generator=default_rng(0))
    selector = CountSelector(list(range(3)), count=12, rng=default_rng(1))
    Xt = generator.generate(
        X.copy(), selector, out_of_distribution_sampler="low_density")
    assert Xt.shape == (72, 3)
    labels = generator.get_outliers_labels()
    assert labels[:60].sum() == 0 and labels[60:].all()


# --- low_density / tree targeted sampler behaviour ---

@pytest.fixture
def two_blob_2d():
    """Two well-separated 2D blobs: the region between them is low density."""
    return make_blobs(
        n_samples=200, centers=[[-6.0, -6.0], [6.0, 6.0]],
        cluster_std=1.0, random_state=0)[0]


def _low_density_frac(X, samples, percentile=10):
    """Fraction of `samples` whose sklearn log-density is at or below the
    `percentile`-th percentile of the data's log-densities (the low-density
    boundary both grid/tree samplers aim for)."""
    estimator = _KernelDensityEstimator(
        bandwidth="scott", algorithm="ball_tree").fit(X)
    threshold = np.percentile(
        estimator.score_samples(X), percentile)
    return (estimator.score_samples(samples) <= threshold).mean()


@pytest.mark.parametrize("sampler", [
    pytest.param(KernelDensitySampler(threshold_low_density=0.1),
                 id="low_density"),
    pytest.param(TreeCellSampler(threshold_low_density=0.1),
                 id="tree"),
])
def test_low_density_samplers__samples_lie_in_low_density_region(
        two_blob_2d, sampler):
    """Sampled rows sit in the low-density region between/around the blobs:
    at least 80% of them score at or below the data's 10th-percentile
    log-density (a few may sit in fallback or cell-interior positions)."""
    X = two_blob_2d
    n_samples = 100
    samples = sampler.sample(default_rng(0), X, [0, 1], n_samples)
    assert samples.shape == (n_samples, 2)
    assert _low_density_frac(X, samples, percentile=10) >= 0.8


def test_low_density__always_returns_exactly_n_samples(two_blob_2d):
    """No rejection budget: every draw count comes back in full."""
    for n in (0, 1, 3, 17):
        samples = KernelDensitySampler(grid_budget=32).sample(
            default_rng(7), two_blob_2d, [0, 1], n)
        assert samples.shape == (n, 2)


def test_low_density__deterministic_for_same_seed(two_blob_2d):
    first = KernelDensitySampler(grid_budget=32).sample(
        default_rng(7), two_blob_2d, [0, 1], 25)
    second = KernelDensitySampler(grid_budget=32).sample(
        default_rng(7), two_blob_2d, [0, 1], 25)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("kwargs, message", [
    ({"grid_budget": 0}, "grid_budget"),
    ({"threshold_low_density": 1.0}, "threshold_low_density"),
])
def test_low_density__rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KernelDensitySampler(**kwargs)


def test_tree__always_returns_exactly_n_samples(two_blob_2d):
    """No rejection budget: every draw count comes back in full."""
    for n in (0, 1, 3, 17):
        samples = TreeCellSampler(leaf_size=4).sample(
            default_rng(7), two_blob_2d, [0, 1], n)
        assert samples.shape == (n, 2)


def test_tree__deterministic_for_same_seed(two_blob_2d):
    first = TreeCellSampler(leaf_size=4).sample(
        default_rng(7), two_blob_2d, [0, 1], 25)
    second = TreeCellSampler(leaf_size=4).sample(
        default_rng(7), two_blob_2d, [0, 1], 25)
    np.testing.assert_array_equal(first, second)


def test_tree__reaches_uncovered_corners(two_blob_2d):
    """Corner regions beyond the data box on both axes (previously dead
    space between the kd-cell extensions) receive samples too."""
    X = two_blob_2d
    samples = TreeCellSampler(threshold_low_density=0.1).sample(
        default_rng(0), X, [0, 1], 400)
    x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    tl = (samples[:, 0] < x0_min) & (samples[:, 1] > x1_max)
    br = (samples[:, 0] > x0_max) & (samples[:, 1] < x1_min)
    assert tl.sum() >= 10
    assert br.sum() >= 10


@pytest.mark.parametrize("kwargs, message", [
    ({"leaf_size": 0}, "leaf_size"),
    ({"threshold_low_density": 0.0}, "threshold_low_density"),
    ({"threshold_low_density": -0.1}, "threshold_low_density"),
])
def test_tree__rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TreeCellSampler(**kwargs)


def test_tree__threshold_one_is_valid(two_blob_2d):
    """threshold_low_density=1.0 accepts every cell; it must not raise."""
    samples = TreeCellSampler(threshold_low_density=1.0).sample(
        default_rng(0), two_blob_2d, [0, 1], 12)
    assert samples.shape == (12, 2)


def test_tree__constant_column_returns_full_shape():
    """A degenerate (zero-variance) column does not break the partition."""
    X = default_rng(0).normal(size=(40, 2))
    X[:, 0] = 3.14
    samples = TreeCellSampler(leaf_size=8).sample(default_rng(0), X, [0, 1], 15)
    assert samples.shape == (15, 2)
