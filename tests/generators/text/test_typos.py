from numpy.random import default_rng

from badgers.generators.text.typos import SwapLettersGenerator, LeetSpeakGenerator, SwapCaseGenerator


def test_swap_letters__transforms_long_words():
    """SwapLettersGenerator swaps inner letters of words longer than 3 chars
    and returns a NEW list without mutating the input."""
    trf = SwapLettersGenerator(random_generator=default_rng(0))
    X = ['abcdef', 'abcde', 'abcd', 'abc', 'ab', 'a']
    original = list(X)
    Xt = trf.generate(X=X, swap_proba=1)

    # the input list is left unchanged (no in-place mutation)
    assert Xt is not X
    assert X == original

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
    trf = LeetSpeakGenerator(random_generator=default_rng(0))
    Xt = trf.generate(X)
    assert len(X) == len(Xt)


def test_leet_speak__modifies_some_words():
    """LeetSpeakGenerator should modify at least some words when proba=1."""
    X = ['leet', 'speak', 'hacker', 'elite', 'password']
    trf = LeetSpeakGenerator(random_generator=default_rng(42))
    Xt = trf.generate(X, replacement_proba=1.0)
    # At least one word should be modified
    assert any(xt != x for xt, x in zip(Xt, X))
    # All words should be non-empty
    assert all(len(xt) > 0 for xt in Xt)


def test_swap_case__uppercases_all():
    """SwapCaseGenerator uppercases all words when proba=1."""
    X = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'fox', ' <> ']
    trf = SwapCaseGenerator(random_generator=default_rng(0))
    Xt = trf.generate(X, swapcase_proba=1.)
    assert len(X) == len(Xt)
    for w1, w2 in zip(X, Xt):
        assert w1.upper() == w2
