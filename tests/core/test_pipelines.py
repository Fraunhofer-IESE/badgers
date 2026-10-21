import numpy as np
import scipy.stats
from numpy.random import default_rng

from badgers.core.pipeline import Pipeline
from badgers.generators.tabular_data.drift.drift import RandomShiftGenerator
from badgers.generators.tabular_data.noise.noise import AdditiveNoiseGenerator


class RecordingGenerator:
    """A deterministic X-only generator for testing the Pipeline contract."""

    def __init__(self, name, result=None, offset=0.0, log=None):
        self.name = name
        self.result = result
        self.offset = offset
        self.log = log if log is not None else []
        self.calls = []

    def generate(self, X, **params):
        self.calls.append({'X': X, 'params': params})
        self.log.append(self.name)
        if self.result is not None:
            return self.result
        return X + self.offset


def test_pipeline__passes_and_returns_single_X_result():
    """Each stage receives the initial X or the previous stage's Xt."""
    initial_X = np.array([[1.0]])
    first_Xt = np.array([[2.0]])
    last_Xt = np.array([[3.0]])
    first = RecordingGenerator('first', result=first_Xt)
    second = RecordingGenerator('second', result=last_Xt)

    result = Pipeline({'first': first, 'second': second}).generate(X=initial_X)

    assert first.calls[0]['X'] is initial_X
    assert second.calls[0]['X'] is first_Xt
    assert result is last_Xt


def test_pipeline__executes_generators_in_insertion_order():
    log = []
    generators = {
        'a': RecordingGenerator('a', offset=1.0, log=log),
        'b': RecordingGenerator('b', offset=2.0, log=log),
        'c': RecordingGenerator('c', offset=3.0, log=log),
    }

    result = Pipeline(generators).generate(X=np.zeros((2, 1)))

    assert log == ['a', 'b', 'c']
    assert np.array_equal(result, np.full((2, 1), 6.0))


def test_pipeline__params_routed_by_generator_name():
    first = RecordingGenerator('first')
    second = RecordingGenerator('second')
    params = {'first': {'alpha': 1}, 'second': {'beta': 2}}

    Pipeline({'first': first, 'second': second}).generate(
        X=np.zeros((2, 1)), params=params
    )

    assert first.calls[0]['params'] == {'alpha': 1}
    assert second.calls[0]['params'] == {'beta': 2}


def test_pipeline__generator_without_params_entry_is_still_called():
    first = RecordingGenerator('first', offset=1.0)
    second = RecordingGenerator('second', offset=2.0)

    result = Pipeline({'first': first, 'second': second}).generate(
        X=np.zeros((2, 1)), params={'first': {'unused': 0}}
    )

    assert second.calls[0]['params'] == {}
    assert np.array_equal(result, np.full((2, 1), 3.0))


def test_pipeline__params_none_calls_all_generators_with_no_kwargs():
    first = RecordingGenerator('first', offset=1.0)
    second = RecordingGenerator('second', offset=2.0)

    result = Pipeline({'first': first, 'second': second}).generate(
        X=np.array([[3.0]]), params=None
    )

    assert first.calls[0]['params'] == {}
    assert second.calls[0]['params'] == {}
    assert np.array_equal(result, np.array([[6.0]]))


def test_pipeline__empty_generators_preserves_data():
    X = np.array([[1.0], [2.0]])

    result = Pipeline({}).generate(X=X)

    assert result is X


def test_pipeline__unknown_param_keys_are_ignored():
    generator = RecordingGenerator('generator', offset=1.0)

    result = Pipeline({'generator': generator}).generate(
        X=np.zeros((1, 1)), params={'ghost': {'x': 1}}
    )

    assert generator.calls[0]['params'] == {}
    assert np.array_equal(result, np.ones((1, 1)))


def test_pipeline__real_generators_chain_to_single_array():
    """Two real migrated tabular generators chain through Pipeline and the
    pipeline returns a single ndarray of the same shape as X."""
    rng = default_rng(0)
    X = rng.normal(size=(100, 3))
    X0 = X.copy()
    pipeline = Pipeline({
        'shift': RandomShiftGenerator(random_generator=default_rng(1)),
        'noise': AdditiveNoiseGenerator(random_generator=default_rng(2)),
    })
    params = {
        'shift': {'shift_std': 0.5},
        'noise': {'distribution': scipy.stats.norm(scale=0.1), 'noise_std': 1.0},
    }

    Xt = pipeline.generate(X, params=params)

    assert isinstance(Xt, np.ndarray)
    assert not isinstance(Xt, tuple)
    assert Xt.shape == X.shape
    assert not np.allclose(Xt, X0)
    # both stages applied: the pipeline output differs from a shift-only run
    shift_only = Pipeline({
        'shift': RandomShiftGenerator(random_generator=default_rng(1)),
    }).generate(X0.copy(), params={'shift': {'shift_std': 0.5}})
    assert not np.allclose(Xt, shift_only)
