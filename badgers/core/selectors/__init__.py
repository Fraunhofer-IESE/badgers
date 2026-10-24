"""Selectors: the observation-side contract for selecting deficit rows.

A SELECTOR decides which rows of the input get a data quality deficit:

    Selector = Callable[[X, targets], dict[int, spec]]

- keys   = column positions (targets: the columns the mask applies to)
- values = per-row boolean masks or float SCORES -- a ``(n,)`` array
  or, for Always-like constant levels, a scalar applied to the whole
  column
- polarity: ``True``/``1.0`` = ``R_X = 1`` = deficit applied;
  ``False``/``0.0`` = record as-is

A spec is a scalar or ``(n,)`` array of values in ``[0, 1]``: values
restricted to ``{0, 1}`` are DECISIONS, anything else is a PROBABILITY
(the chance the deficit is applied to that row). Two rng streams
cooperate: the PRESET's own rng drives the spec draw (the decision or
probability computation, the stochastic row draws, the row weights),
while the CONSUMER's rng drives the realization -- every spec is
realized with the single rule ``rng.random(n) < spec``. ``selector=None``
means no selector draw at all (the consumer decides). Being in ``core``,
the package
is modality-agnostic: time-series / graph / text generators can reuse
the same contract with masks over their own axes.

The presets are self-contained: their column scope (``columns``) and the
column(s) they READ (bare int positions) are bound at construction, so
an intensity sweep means constructing a new selector (nothing is swept
on the call). There is deliberately no base class and no registry --
conformance to the :class:`Selector` protocol is defined by the call
signature alone, and each preset's rule is inlined in ``__call__``.

MODIFYING consumers preserve the input's shape and only ALTER the fired
cells (missingness, noise, outliers, ...); SUBSAMPLING consumers
(imbalance) DROP the fired rows, so their output is smaller than the
input. The contract itself is only "which rows fire, on which columns".
All data used to determine selection, including class and driver
columns, is supplied as ``X``; there is no separate target argument.

Layout: :mod:`.base` holds the contract (protocol, shared validation,
consumer helpers); :mod:`.value_free` and :mod:`.value_driven` hold the
presets, split along the one constructor-contract difference they have
(value-free presets never read ``X``, value-driven presets read column
positions).
"""

from .base import Selector, fired, resolve_class_column
from .value_driven import ClassSelector, ProportionalSelector, TopQuantileSelector
from .value_free import AlwaysSelector, BernoulliSelector, ExactCountSelector

__all__ = [
    "Selector",
    "fired",
    "resolve_class_column",
    "AlwaysSelector",
    "BernoulliSelector",
    "ExactCountSelector",
    "TopQuantileSelector",
    "ClassSelector",
    "ProportionalSelector",
]
