def normalize_proba(p):
    """
    Make sure the probability array respects the following constraints:
     - the sum of each column must be equal to 1 (or very close to 1)
    Parameters
    ----------
    p

    Returns
    -------

    """
    # make the sum of each column = 1
    sum = p.sum(axis=0)
    # assure that no division by 0
    sum[sum == 0] = 1
    # normalize
    p = p / sum
    return p