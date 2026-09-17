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

        # Vectorized: use cumprod to avoid O(d²) Python loop
        cumprod_sin = np.cumprod(sin_phis)
        # x_i = radius * cos(phi_i) * prod(sin(phi_0)...sin(phi_{i-1})) for i=0..d-2
        # x_{d-1} = radius * prod(sin(phi_0)...sin(phi_{d-1}))
        x_head = radius * cos_phis[:-1] * cumprod_sin[:-1]
        x_tail = np.array([radius * cumprod_sin[-1]])
        x = np.concatenate([x_head, x_tail])
    return x


def random_spherical_coordinates(
    random_generator: np.random.Generator = default_rng(0),
    size: int = None,
    radii: np.ndarray = None,
) -> np.ndarray:
    """Generate multiple points on a hypersphere in batch (vectorized).

    This is the batch equivalent of :func:`random_spherical_coordinate`,
    producing ``n_samples`` points at once using vectorized NumPy operations.
    It is significantly faster than calling the scalar version in a loop
    when ``n_samples`` is large.

    Args:
        random_generator: A numpy random number generator.
        size: The dimension of the hypersphere (number of coordinates per point).
        radii: Array of shape ``(n_samples,)`` with the radius for each point.

    Returns:
        An array of shape ``(n_samples, size)`` where each row is a point
        on the hypersphere with the corresponding radius from ``radii``.
    """
    assert size > 0
    n_samples = len(radii)

    if size == 1:
        signs = random_sign(random_generator, size=(n_samples, 1))
        return signs * radii.reshape(-1, 1)
    elif size == 2:
        phi = random_generator.uniform(0, 2. * np.pi, size=n_samples)
        x = np.column_stack([
            radii * np.cos(phi),
            radii * np.sin(phi),
        ])
        return x
    else:
        # Generate all angles at once: (n_samples, size-1)
        phis = np.empty((n_samples, size - 1))
        phis[:, :-1] = random_generator.uniform(0, np.pi, size=(n_samples, size - 2))
        phis[:, -1] = random_generator.uniform(0, 2. * np.pi, size=n_samples)

        cos_phis = np.cos(phis)
        sin_phis = np.sin(phis)

        # Vectorized cumprod along axis=1
        cumprod_sin = np.cumprod(sin_phis, axis=1)

        # x[:, i] = radii * cos_phis[:, i] * cumprod_sin[:, i-1]  for i=0..d-2
        # x[:, d-1] = radii * cumprod_sin[:, -1]
        x_head = radii.reshape(-1, 1) * cos_phis[:, 1:] * cumprod_sin[:, :-1]
        x_tail = radii.reshape(-1, 1) * cumprod_sin[:, -1:]

        # First coordinate: radii * cos_phis[:, 0]
        x_first = radii.reshape(-1, 1) * cos_phis[:, :1]

        return np.concatenate([x_first, x_head, x_tail], axis=1)
