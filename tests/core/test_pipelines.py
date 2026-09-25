import numpy as np
import pytest

from numpy.random import default_rng
from sklearn.datasets import make_blobs

from badgers.core.pipeline import Pipeline
from badgers.generators.tabular_data.imbalance.imbalance import RandomSamplingClassesGenerator
from badgers.generators.tabular_data.noise.noise import GaussianNoiseGenerator


def test_pipeline__preserves_shape():
    """Pipeline output should have same number of rows as input."""
    random_generator = default_rng(0)
    X, y = make_blobs(centers=3, random_state=0)
    generators = {
        'imbalance': RandomSamplingClassesGenerator(random_generator=random_generator),
        'noise': GaussianNoiseGenerator(random_generator=random_generator)
    }
    pipeline = Pipeline(generators=generators)
    params = {
        'imbalance': {'proportion_classes': {0: 0.5, 1: 0.25, 2: 0.25}},
        'noise': {'noise_std': 0.5}
    }
    Xt, yt = pipeline.generate(X=X, y=y, params=params)
    assert len(Xt) == len(X)
    assert len(yt) == len(y)


def test_pipeline__modifies_data():
    """Pipeline with noise should modify the data."""
    random_generator = default_rng(0)
    X, y = make_blobs(centers=3, random_state=0)
    generators = {
        'noise': GaussianNoiseGenerator(random_generator=random_generator)
    }
    pipeline = Pipeline(generators=generators)
    params = {'noise': {'noise_std': 1.0}}
    Xt, yt = pipeline.generate(X=X.copy(), y=y, params=params)
    assert not np.allclose(Xt, X)


def test_pipeline__empty_generators_preserves_data():
    """Pipeline with no generators should return data unchanged."""
    X, y = make_blobs(centers=3, random_state=0)
    pipeline = Pipeline(generators={})
    Xt, yt = pipeline.generate(X=X.copy(), y=y, params={})
    assert np.allclose(Xt, X)
    assert np.array_equal(yt, y)


# --- Requirement-derived unit tests (spec: badgers/core/pipeline.py docstrings, req/REQ-P-00x) ---


class RecordingGenerator:
    """Test double at the Pipeline/generator boundary: records each call and
    applies a deterministic, spec-independent transform (X -> X + offset)."""

    def __init__(self, name, offset=0.0, log=None):
        self.name = name
        self.offset = offset
        self.log = log if log is not None else []
        self.calls = []

    def generate(self, X, y, **params):
        self.calls.append({'X': X, 'y': y, 'params': params})
        self.log.append(self.name)
        return X + self.offset, y


def test_pipeline__chaining_output_feeds_next_input():
    """REQ-P-003: each generator receives the previous generator's output."""
    g1 = RecordingGenerator('g1', offset=1.0)
    g2 = RecordingGenerator('g2', offset=10.0)
    pipeline = Pipeline(generators={'g1': g1, 'g2': g2})
    X = np.array([[1.0], [2.0]])
    Xt, yt = pipeline.generate(X=X, y=None)
    # spec-derived: g2 must have received (X + 1), final output must be (X + 1) + 10
    assert np.array_equal(g2.calls[0]['X'], X + 1)
    assert np.array_equal(Xt, X + 11.0)
    assert yt is None


def test_pipeline__executes_generators_in_insertion_order():
    """REQ-P-004: generators run in the order given by the generators mapping."""
    log = []
    gens = {
        'a': RecordingGenerator('a', offset=1.0, log=log),
        'b': RecordingGenerator('b', offset=2.0, log=log),
        'c': RecordingGenerator('c', offset=3.0, log=log),
    }
    Xt, _ = Pipeline(generators=gens).generate(X=np.zeros((2, 1)), y=None)
    assert log == ['a', 'b', 'c']
    # order is also observable in the values: offsets accumulate in sequence
    assert np.array_equal(Xt, np.full((2, 1), 6.0))


def test_pipeline__params_routed_by_generator_name():
    """REQ-P-002: each generator receives only the kwargs stored under its own name."""
    g1 = RecordingGenerator('g1')
    g2 = RecordingGenerator('g2')
    params = {'g1': {'alpha': 1}, 'g2': {'beta': 2}}
    Pipeline(generators={'g1': g1, 'g2': g2}).generate(X=np.zeros((2, 1)), y=None, params=params)
    assert g1.calls[0]['params'] == {'alpha': 1}
    assert g2.calls[0]['params'] == {'beta': 2}


def test_pipeline__generator_without_params_entry_is_still_called():
    """REQ-P-006: a generator missing from params is called with no extra kwargs."""
    g1 = RecordingGenerator('g1', offset=1.0)
    g2 = RecordingGenerator('g2', offset=2.0)
    Xt, _ = Pipeline(generators={'g1': g1, 'g2': g2}).generate(
        X=np.zeros((2, 1)), y=None, params={'g1': {'unused': 0}}
    )
    assert g2.calls[0]['params'] == {}          # M3 (skip mutant) fails here
    assert np.array_equal(Xt, np.full((2, 1), 3.0))


def test_pipeline__params_none_calls_all_generators_with_no_kwargs():
    """REQ-P-005: params=None behaves like an empty params mapping."""
    g1 = RecordingGenerator('g1', offset=1.0)
    g2 = RecordingGenerator('g2', offset=2.0)
    Xt, yt = Pipeline(generators={'g1': g1, 'g2': g2}).generate(
        X=np.array([[3.0]]), y=np.array([7]), params=None
    )
    assert g1.calls[0]['params'] == {}
    assert g2.calls[0]['params'] == {}
    assert np.array_equal(Xt, np.array([[6.0]]))  # M4 (None not defaulted) fails here
    assert np.array_equal(yt, np.array([7]))      # REQ-P-007: y flows through unchanged


def test_pipeline__returns_last_generators_result():
    """REQ-P-007: the returned (Xt, yt) is the output of the final generator."""
    g1 = RecordingGenerator('g1', offset=1.0)
    g2 = RecordingGenerator('g2', offset=2.0)
    Xt, yt = Pipeline(generators={'g1': g1, 'g2': g2}).generate(
        X=np.array([[5.0]]), y=np.array([1])
    )
    assert np.array_equal(Xt, np.array([[8.0]]))
    assert np.array_equal(yt, np.array([1]))


def test_pipeline__single_generator_boundary():
    """REQ-P-002 boundary: a mapping with one generator acts like a direct call."""
    g1 = RecordingGenerator('g1', offset=1.0)
    Xt, yt = Pipeline(generators={'g1': g1}).generate(X=np.array([[0.0]]), y=np.array([2]))
    assert np.array_equal(Xt, np.array([[1.0]]))
    assert np.array_equal(yt, np.array([2]))


def test_pipeline__unknown_param_keys_are_ignored():
    """REQ-P-008 (records current behavior; ambiguous in the docstring):
    keys in params that match no generator do not raise."""
    g1 = RecordingGenerator('g1', offset=1.0)
    Xt, _ = Pipeline(generators={'g1': g1}).generate(
        X=np.array([[0.0]]), y=None, params={'ghost': {'x': 1}}
    )
    assert g1.calls[0]['params'] == {}
    assert np.array_equal(Xt, np.array([[1.0]]))


def test_pipeline__y_flows_through_real_generators():
    """REQ-P-007 integration: with a real noise generator, y is documented to
    remain unchanged while flowing through the pipeline (GaussianNoiseGenerator)."""
    X, y = make_blobs(centers=3, random_state=0)
    gen = GaussianNoiseGenerator(random_generator=default_rng(0))
    Xt, yt = Pipeline(generators={'noise': gen}).generate(
        X=X.copy(), y=y, params={'noise': {'noise_std': 1.0}}
    )
    assert np.array_equal(yt, y)          # strengthens the weak `len(yt)` check
    assert np.asarray(Xt).shape == X.shape
    assert not np.allclose(Xt, X)
