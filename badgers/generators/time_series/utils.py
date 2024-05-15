from typing import List, Tuple

import numpy as np


def generate_random_patterns_indices(random_generator: np.random.Generator, signal_size: int, n_patterns: int,
                                     min_width_pattern: int, max_width_patterns: int) -> List[Tuple[int, int]]:
    """
    Generates a list of patterns indices (start index, stop index) randomly such that:
    - the patterns have a size in [min_width_pattern, max_width_patterns[
    - the patterns do not overlap
    - the patterns indices stays within the total signal size

    The algorithm is the following:

    1.  Compute patterns size (uniformly between min_width_pattern and max_width_patterns).

    2.  Split the total signal size into equal length segments.
        Each segment will contain exactly one pattern.
        So there are `n_patterns` segments.

    3.  For each segment, choose uniformly the starting and end points of the pattern such that the patterns fits into the segment.



    :param random_generator: a random generator
    :param signal_size: total signal size
    :param n_patterns: total number of patterns
    :param min_width_pattern: the minimal width of the patterns
    :param max_width_patterns: the maximal width of the patterns
    :return: a list of patterns indices (start index, stop index)
    """
    assert max_width_patterns * n_patterns < signal_size, 'the number of patterns * their maximum width should be smaller thant the total signal size'
    # randomly generates sizes for all patterns
    patterns_sizes = random_generator.integers(low=min_width_pattern, high=max_width_patterns, size=n_patterns)
    # size of segments (for splitting the total signal length)
    segment_size = signal_size // n_patterns
    # compute patterns indices
    patterns_indices = []
    for i in range(n_patterns - 1):
        start = random_generator.integers(low=i * segment_size, high=(i + 1) * segment_size - patterns_sizes[i])
        end = start + patterns_sizes[i]
        patterns_indices += [(start, end)]
    else:
        # take care of the last remaining segment
        start = random_generator.integers(low=(i + 1) * segment_size, high=signal_size - patterns_sizes[i])
        end = start + patterns_sizes[i + 1]
        patterns_indices += [(start, end)]
    return patterns_indices
