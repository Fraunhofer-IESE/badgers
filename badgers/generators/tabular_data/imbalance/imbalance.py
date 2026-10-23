import abc

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import fired_mask
from badgers.core.utils import normalize_proba


class ImbalanceGenerator(GeneratorMixin):
    """
    Base class for transformers that makes tabular data imbalanced
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the ImbalanceGenerator with a specified random number generator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        self.random_generator = random_generator
        self.imbalance_labels_: np.ndarray | None = None

    def get_imbalance_labels(self) -> np.ndarray:
        """
        Per-row imbalance labels of the LAST generated output.

        :return: A boolean array aligned with the rows of the LAST input
            passed to ``generate()``; ``True`` marks a row on which the
            deficit fired (the row was dropped from the output), ``False``
            a row that is part of the generated output.
        :raises RuntimeError: if ``generate()`` has not been called yet.
        """
        if self.imbalance_labels_ is None:
            raise RuntimeError(
                "no labels available: call generate() first "
                "(the labels describe the LAST input to generate())"
            )
        return self.imbalance_labels_

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        Abstract method to generate imbalanced data from the input data.
        This should be overridden

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param params: Additional keyword arguments that might be required for specific implementations.
        :type params: dict
        """
        pass


class RandomSamplingClassesGenerator(ImbalanceGenerator):
    """
    Randomly samples data points within predefined classes
    """

    def __init__(self, random_generator=default_rng(seed=0), ):
        """
        Initialize the RandomSamplingClassesGenerator with a specified random number generator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, selector=None):
        """
        Drops the rows fired by the selector (subsampling deficit),
        per-class: exactly ``int(prop * n_rows)`` rows of each listed class
        are removed, classes absent from the mapping are kept.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param selector: A selector (e.g.
                         :class:`badgers.core.selectors.ClassSelector`)
                         deciding which rows are dropped. ``None`` =
                         no-op (all rows kept). The class variable is a
                         column of ``X`` (read from the selector's
                         ``class_column``).
        :type selector: callable, optional
        :return: The sampled features (Xt).
        """
        if selector is None:
            self.imbalance_labels_ = np.zeros(X.shape[0], dtype=bool)
            return X
        # a row is dropped if fired in any class column
        fired = fired_mask(X, selector, self.random_generator)
        Xt = X[~fired, :]
        self.imbalance_labels_ = fired
        return Xt


class RandomSamplingTargetsGenerator(ImbalanceGenerator):
    """
    Resamples rows with per-row probabilities derived from a selector.

    The selector decides, for each row, the probability that the deficit
    is applied. The deficit HERE is "this row is not kept": the generator
    realizes the fractional scores as KEEP weights
    (``1 - deficit_probability``, noisy-OR combined over the selector's
    variables) and resamples rows from those weights.

    This is the single score-driven resampler of the imbalance family,
    covering the two former classes: a FRACTIONAL selector over a
    continuous column (e.g. ``ProportionalSelector``) gives the
    value-weighted resample of the removed ``RandomSamplingFeaturesGenerator``,
    while a BOOLEAN selector over a discrete class column (e.g.
    ``ClassSelector``) gives an exact per-class subsample (the fired rows
    are dropped).

    With ``replace=True`` (default) rows may be DUPLICATED -- the classic
    importance-resampling deficit. With ``replace=False`` the output is a
    weighted thinning (each row kept at most once).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomSamplingTargetsGenerator with a specified random number generator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, *, selector=None, replace=True, size=None):
        """
        Resamples instances from a selector's deficit probabilities.

        The selector returns per-row deficit probabilities (float scores in
        ``[0, 1]``, boolean masks being the degenerate case); rows are
        resampled with probability proportional to ``1 - p`` (i.e. rows
        with a high deficit probability are largely dropped, rows with a
        low deficit probability are kept/replicated).

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :param selector: A selector deciding which rows get the deficit.
            E.g. ``ProportionalSelector([0], 0)`` (deficit
            probability proportional to a column), a ``BernoulliSelector``,
            or any callable accepting X and returning ``dict[str, array]``
            of per-row scores. Required (``None`` raises).
        :param replace: Whether to sample with replacement (default True:
            rows may be duplicated). ``False`` gives a weighted thinning.
        :param size: Number of rows to draw (default: the input row count).
        :return: The resampled features (Xt).
        """
        if selector is None:
            raise ValueError(
                "RandomSamplingTargetsGenerator needs a selector deciding "
                "which rows get the deficit (e.g. ProportionalSelector "
                "over a column of X)."
            )
        scores = [np.asarray(s, dtype=float) for s in selector(X).values()]
        if len(scores) == 1:
            deficit_proba = scores[0]
        else:
            # noisy-OR: the probabilistic generalization of the boolean OR
            keep = np.prod([1.0 - s for s in scores], axis=0)
            deficit_proba = 1.0 - keep
        if (deficit_proba < 0).any() or (deficit_proba > 1).any():
            raise ValueError(
                "selector scores must be in [0, 1] "
                f"(got [{deficit_proba.min()}, {deficit_proba.max()}])"
            )
        keep_weights = 1.0 - deficit_proba
        if not keep_weights.sum() > 0:
            raise ValueError(
                "every row has deficit probability 1: no row can be kept "
                "(lower the selector's deficit probabilities)"
            )
        # Boolean scores (all 0/1) are DETERMINISTIC: realize them as an
        # exact subsample of the rows that never fire (deficit proba 0) --
        # a weighted draw without replacement of size n would draw every
        # index once, zero-weight rows included.
        if np.isin(deficit_proba, (0.0, 1.0)).all():
            kept = np.flatnonzero(deficit_proba == 0.0)
            if size is not None and size < len(kept):
                kept = self.random_generator.choice(kept, size=size, replace=False)
            Xt = X[kept, :]
            fired = np.ones(X.shape[0], dtype=bool)
            fired[kept] = False
            self.imbalance_labels_ = fired
            return Xt
        # Fractional scores: normalize the keep weights into a probability
        # vector and resample from them.
        p = normalize_proba(keep_weights.reshape(-1, 1)).ravel()
        n = self.random_generator.choice(
            X.shape[0], p=p, size=X.shape[0] if size is None else size,
            replace=replace,
        )
        Xt = X[n, :]
        fired = np.ones(X.shape[0], dtype=bool)
        fired[n] = False
        self.imbalance_labels_ = fired
        return Xt
