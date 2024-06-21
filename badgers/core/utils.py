from typing import Tuple

import numpy as np
import numpy.random
from numpy.random import default_rng


def normalize_proba(p: np.array) -> np.array:
    """
    Make sure the probability array respects the following constraints:
     - the sum of each column must be equal to 1 (or very close to 1)

    :param p: The probability array to normalize
    :return: The normalized array
    """
    # make the sum of each column = 1
    sum = p.sum(axis=0)
    # assure that no division by 0
    if isinstance(sum, np.ndarray) and sum.ndim > 1:
        sum[sum == 0] = 1
    # normalize
    p = p / sum
    return p


def random_sign(random_generator: numpy.random.Generator = default_rng(0), size: Tuple[int] = (1,)) -> np.array:
    """
    Generates an array full of ones, with randomly assigned signs.

    :param random_generator: a random number generator
    :param size: the shape of the array to generate
    :return: an array full of ones, with randomly assigned signs
    """
    signs = np.ones(shape=size)
    mask = random_generator.random(size=size) < 0.5
    signs[mask] *= -1
    return signs


def random_spherical_coordinate(random_generator: numpy.random.Generator = default_rng(0),
                                size: int = None,
                                radius: float = None) -> np.array:
    """
    Randomly generates points on a hypersphere of dimension `size`
    :param random_generator: a random number generator
    :param size: the dimension of the hypersphere
    :param radius: the radius of the hypersphere
    :return: an array of shape (`size`,) containing the values of the point generated
    """
    assert size > 0
    if size == 1:
        x = random_sign(random_generator, size=(1,)) * radius
    elif size == 2:
        phi = random_generator.uniform(0, 2. * np.pi)
        x = np.array([radius * np.cos(phi), radius * np.sin(phi)])
    else:
        phis = np.concatenate([
            random_generator.uniform(0, np.pi, size=size - 2),
            [random_generator.uniform(0, 2. * np.pi)]
        ])

        cos_phis = np.cos(phis)
        sin_phis = np.sin(phis)

        x = np.array(
            [radius * cos_phis[i] * np.prod(sin_phis[:i]) for i in range(size - 1)] + [radius * np.prod(sin_phis)]
        )
    return x
