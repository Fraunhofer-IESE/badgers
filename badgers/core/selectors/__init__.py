"""Selectors: the observation-side contract for selecting deficit rows.

A SELECTOR decides which rows of the input get a data quality deficit

    Selector = Callable[[X], dict[str, bool-mask | score]]

- keys   = variable names (the subscript in ``R_X``: which column the mask applies to)
- values = per-row boolean masks or float SCORES, ``len(mask) == X.shape[0]``
- polarity: ``True``/``1.0`` = ``R_X = 1`` = deficit applied; ``False``/``0.0`` = record as-is

Boolean masks are the common case, but a selector may return FLOAT scores in
``[0, 1]``: the PROBABILITY that the deficit is applied to that row (the
deterministic ``1.0``/``0.0``/``bool`` values being the degenerate cases). A
selector itself never draws randomness for these rates -- realization belongs
to the consuming generator, which binds a single rng stream: a MODIFYING consumer
(noise, missingness) realizes ``p = 0.3`` as a Bernoulli firing; a RESAMPLING
consumer (imbalance) uses it as a selection weight. Being in ``core``, the
package is modality-agnostic: time-series / graph / text generators can reuse
the same contract with masks over their own axes.

The presets are small callable objects so their configuration (which
variables, which rng, which column mapping, which intensity) stays
introspectable (``repr`` and stored attributes). The FIRE INTENSITY is part
of the preset configuration where it exists: ``rate`` for the stochastic
presets (``BernoulliSelector``/``ExactCountSelector``), ``quantile`` for
the deterministic ``TopQuantileSelector``, and ``proportions`` for the
per-class ``ClassSelector`` are constructor arguments, so an
intensity sweep means constructing a new selector (nothing is swept on the
call). The trivial ``AlwaysSelector`` has no intensity knob at all. There
is deliberately no
base class and no
registry -- conformance to the :class:`Selector` protocol is defined by the
call signature alone. Each
preset is self-contained: its mask logic is inlined in ``__call__`` (no free
helper functions), so a preset depends on nothing beyond numpy and the
consuming generators simply construct one and call it.

MODIFYING consumers preserve the input's shape and only ALTER the fired cells
(missingness, noise, outliers, ...); SUBSAMPLING consumers (imbalance) DROP
the fired rows, so their output is smaller than the input. The contract
itself is only "which rows fire".
All data used to determine selection, including class and driver columns,
is supplied as ``X``; there is no separate target argument.

Layout: :mod:`.base` holds the contract (protocol, shared validation,
consumer helpers); :mod:`.value_free` and :mod:`.value_driven` hold the
presets, split along the one constructor-contract difference they have
(value-free presets never read ``X``, value-driven presets read columns via
a ``column_mapping``).
"""

from .base import Selector, fired_mask, fired_rows, resolve_class_column
from .value_driven import ClassSelector, ProportionalSelector, TopQuantileSelector
from .value_free import AlwaysSelector, BernoulliSelector, ExactCountSelector

__all__ = [
    "Selector",
    "fired_mask",
    "fired_rows",
    "resolve_class_column",
    "AlwaysSelector",
    "BernoulliSelector",
    "ExactCountSelector",
    "TopQuantileSelector",
    "ClassSelector",
    "ProportionalSelector",
]
