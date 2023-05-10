def normalize_proba(p):
    """
    Make sure the probability array respects the following constraints:
     - the sum of each column must be equal to 1 (or very close to 1)

    :param p: np.array,
    """
    # make the sum of each column = 1
    sum = p.sum(axis=0)
    # assure that no division by 0
    sum[sum == 0] = 1
    # normalize
    p = p / sum
    return p


def random_sign(random_generator):
    """
    Randomly return 1 or -1
    :param random_generator:
    :return:
    """
    return 1 if random_generator.random() < 0.5 else -1