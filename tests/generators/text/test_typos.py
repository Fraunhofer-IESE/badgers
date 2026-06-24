from copy import deepcopy

from numpy.random import default_rng

from badgers.generators.text.typos import SwapLettersGenerator, LeetSpeakGenerator, SwapCaseGenerator


def test_swap_letters__transforms_long_words():
    """SwapLettersGenerator swaps inner letters of words longer than 3 chars."""
    trf = SwapLettersGenerator(random_generator=default_rng(0))
    X = ['abcdef', 'abcde', 'abcd', 'abc', 'ab', 'a']
    Xt, _ = trf.generate(X=deepcopy(X), y=None, swap_proba=1)

    for i in range(len(X)):
        if len(X[i]) > 3:
            assert Xt[i] != X[i]
            assert Xt[i][0] == X[i][0]
            assert Xt[i][-1] == X[i][-1]
            assert len(Xt[i]) == len(X[i])
            assert set(Xt[i]) == set(X[i])
        else:
            assert Xt[i] == X[i]


def test_leet_speak__generates_same_length():
    """LeetSpeakGenerator returns same number of words."""
    X = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'fox', ' <> ']
    trf = LeetSpeakGenerator()
    Xt, _ = trf.generate(X, None)
    assert len(X) == len(Xt)


def test_swap_case__uppercases_all():
    """SwapCaseGenerator uppercases all words when proba=1."""
    X = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'fox', ' <> ']
    trf = SwapCaseGenerator()
    Xt, _ = trf.generate(X, None, swapcase_proba=1.)
    assert len(X) == len(Xt)
    for w1, w2 in zip(X, Xt):
        assert w1.upper() == w2
