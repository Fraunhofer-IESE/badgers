"""Imbalance generators.

Two complementary, unambiguous classes — no ``size`` / ``replace`` flag:

- :class:`UndersamplingGenerator` — rows SELECTED by a deficit
  :class:`~badgers.core.selectors.Selector` are REMOVED. The selector decides
  which rows to drop; the amount lives entirely in the selector.
- :class:`OversamplingGenerator` — rows are DUPLICATED according to per-row
  integer multiplicities supplied by a multiplicity selector. The selector
  decides which rows to copy and how many times; the output always contains
  every input row at least once (nothing is dropped).

Both expose :meth:`ImbalanceGenerator.get_imbalance_labels` (a boolean vector
aligned with the last generated output: removed rows for undersampling,
duplicated rows for oversampling).
"""

import numpy as np

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors.base import get_selected_fields

from .multiplicity import resolve_multiplicities


class ImbalanceGenerator(GeneratorMixin):
    """Base for imbalance generators (undersampling / oversampling).

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.
    """

    deficit_name = "imbalance"

    def __init__(self, random_generator):
        self.random_generator = random_generator

    def get_imbalance_labels(self) -> np.ndarray | None:
        """Return per-row imbalance labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the imbalance deficit (rows removed
        by undersampling, or rows that were duplicated by oversampling),
        False otherwise. ``None`` if nothing has been generated yet.
        """
        return self._imbalance_labels_

    def _set_imbalance_labels(self, labels: np.ndarray | None):
        self._imbalance_labels_ = (
            None if labels is None else labels.astype(bool)
        )


class UndersamplingGenerator(ImbalanceGenerator):
    """Remove rows selected by a deficit selector.

    The selector encodes the amount of imbalance entirely (a ``[0, 1]``
    deficit spec per column). A fractional spec is realized stochastically
    (each selected row is dropped with probability ``spec[i]``); a
    boolean/integer spec is deterministic. With multiple target columns the
    union of selected rows is dropped.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.

    generate() parameters
    ----------
    selector : Selector or None, default=None
        Deficit row-selection rule from ``badgers.core.selectors``. ``None``
        returns the input unchanged.
    """

    @preprocess_inputs
    def generate(self, X, selector=None):
        """Return the input with selector-selected rows removed."""
        if selector is None:
            self._set_imbalance_labels(np.zeros(X.shape[0], dtype=bool))
            return X
        selected = get_selected_fields(X, selector, self.random_generator).any(axis=1)
        Xt = X[~selected, :]
        # True = removed; aligned with the input (the pre-subset frame).
        self._set_imbalance_labels(selected)
        return Xt


class OversamplingGenerator(ImbalanceGenerator):
    """Duplicate rows according to per-row integer multiplicities.

    The multiplicity selector encodes the amount of imbalance entirely: each
    row ``i`` appears ``m[i]`` times in the output (``m[i] >= 1``), in the
    original row order. The output always contains every input row at least
    once — oversampling never drops a row.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.

    generate() parameters
    ----------
    selector : MultiplicitiesSelector or int or array-like or None, \
default=None
        Per-row multiplicities: a :class:`MultiplicitiesSelector` (``__call__``
        returns an integer array ``(n_rows,)``), a bare integer (every row),
        or an integer array of shape ``(n_rows,)``. All values must be
        ``>= 1``. ``None`` returns the input unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from badgers.generators.tabular_data.imbalance import (
    ...     OversamplingGenerator, ConstantMultiplicity,
    ... )
    >>> gen = OversamplingGenerator(np.random.default_rng(0))
    >>> X = np.arange(6).reshape(3, 2)
    >>> gen.generate(X, ConstantMultiplicity(2)).shape
    (6, 2)
    """

    @preprocess_inputs
    def generate(self, X, selector=None):
        """Return the input with rows duplicated per their multiplicity."""
        m = resolve_multiplicities(X, selector)
        idx = np.repeat(np.arange(X.shape[0]), m)
        Xt = X[idx, :]
        # True = duplicated (multiplicity > 1); aligned with the input.
        self._set_imbalance_labels(m > 1)
        return Xt
