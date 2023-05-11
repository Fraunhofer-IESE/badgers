import numpy as np
import numpy.random
import pandas as pd
from numpy.random import default_rng


def generate_test_data_without_labels(rng: numpy.random.Generator = default_rng(0)):
    """

    :param rng:
    :return:
    """
    return {
        'numpy_1D': rng.normal(size=10).reshape(-1, 1),
        'numpy_2D': rng.normal(size=(100, 10)),
        'pandas_1D': pd.DataFrame(
            data=rng.normal(size=10).reshape(-1, 1),
            columns=['col']
        ),
        'pandas_2D': pd.DataFrame(
            data=rng.normal(size=(100, 10)),
            columns=[f'col{i}' for i in range(10)]
        )
    }


def generate_test_data_with_labels(rng: numpy.random.Generator = default_rng(0)):
    """

    :param rng:
    :return:
    """
    return {
        'numpy_1D': (rng.normal(size=10).reshape(-1, 1), np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])),
        'numpy_2D': (rng.normal(size=(100, 10)), np.array([1, 2, 3, 4, 5] * 20)),
        'pandas_1D': (
            pd.DataFrame(data=rng.normal(size=10).reshape(-1, 1),
                         columns=['col']),
            pd.Series(data=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ),
        'pandas_2D': (
            pd.DataFrame(data=rng.normal(size=(100, 10)),
                         columns=[f'col{i}' for i in range(10)]),
            pd.Series(data=[1, 2, 3, 4, 5] * 20)
        )
    }
