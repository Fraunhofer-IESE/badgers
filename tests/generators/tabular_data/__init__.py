import numpy as np
import numpy.random
import pandas as pd
from numpy.random import default_rng


def generate_test_data_only_features(rng: numpy.random.Generator = default_rng(0)):
    """

    :param rng:
    :return:
    """
    return {
        'list_1D': (rng.normal(size=100).tolist(), None),
        'list_2D': (rng.normal(size=(100, 10)).tolist(), None),
        'numpy_1D': (rng.normal(size=100).reshape(-1, 1), None),
        'numpy_2D': (rng.normal(size=(100, 10)), None),
        'pandas_1D': (
            pd.DataFrame(
                data=rng.normal(size=100).reshape(-1, 1),
                columns=['col']
            ), None),
        'pandas_2D': (
            pd.DataFrame(
                data=rng.normal(size=(100, 10)),
                columns=[f'col{i}' for i in range(10)]
            ), None),
    }


def generate_test_data_with_classification_labels(rng: numpy.random.Generator = default_rng(0)):
    """

    :param rng:
    :return:
    """
    return {
        'list_1D': (rng.normal(size=100).reshape(-1, 1).tolist(), [i % 5 for i in range(100)]),
        'list_2D': (rng.normal(size=(100, 10)).tolist(), [0, 1, 2, 3, 4] * 20),
        'numpy_1D': (rng.normal(size=100).reshape(-1, 1), np.array([i % 5 for i in range(100)])),
        'numpy_2D': (rng.normal(size=(100, 10)), np.array([0, 1, 2, 3, 4] * 20)),
        'pandas_1D': (
            pd.DataFrame(data=rng.normal(size=100).reshape(-1, 1),
                         columns=['col']),
            pd.Series(data=[i % 5 for i in range(100)])
        ),
        'pandas_2D': (
            pd.DataFrame(data=rng.normal(size=(100, 10)),
                         columns=[f'col{i}' for i in range(10)]),
            pd.Series(data=[0, 1, 2, 3, 4] * 20)
        )
    }


def generate_test_data_with_regression_targets(rng: numpy.random.Generator = default_rng(0)):
    """

    :param rng:
    :return:
    """
    return {
        'list_1D': (rng.normal(size=100).reshape(-1, 1).tolist(), rng.normal(size=100).tolist()),
        'list_2D': (rng.normal(size=(100, 10)).tolist(), rng.normal(size=100).tolist()),
        'numpy_1D': (rng.normal(size=100).reshape(-1, 1), rng.normal(size=100)),
        'numpy_2D': (rng.normal(size=(100, 10)), rng.normal(size=100)),
        'pandas_1D': (
            pd.DataFrame(data=rng.normal(size=100).reshape(-1, 1),
                         columns=['col']),
            pd.Series(data=rng.normal(size=100))
        ),
        'pandas_2D': (
            pd.DataFrame(data=rng.normal(size=(100, 10)),
                         columns=[f'col{i}' for i in range(10)]),
            pd.Series(rng.normal(size=100))
        )
    }
