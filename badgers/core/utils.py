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
    sum[sum == 0] = 1
    # normalize
    p = p / sum
    return p


def random_sign(random_generator: numpy.random.Generator = default_rng(0), shape: Tuple[int] | int = 1) -> np.array:
    """
    Generates an array full of ones, with randomly assigned signs.

    :param random_generator: a random number generator
    :param shape: the shape of the array to generate
    :return: an array full of ones, with randomly assigned signs
    """
    signs = np.ones(shape=shape)
    mask = random_generator.random(size=shape) < 0.5
    signs[mask] *= -1
    return signs
