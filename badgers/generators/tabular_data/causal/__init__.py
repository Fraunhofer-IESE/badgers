"""Causal data quality defect generators.

The causal defect family generates bad data by perturbing the *structure*
(causal graph + data generation process) of the data, rather than the raw
values. Each generator fits a per-node DGP to the data (the default OLS
backend is pure NumPy and needs no extra dependencies), applies a structural
perturbation, then propagates the effect through the descendants.

Custom estimation backends (e.g. PyMC- or DoWhy-GCM-based) can be provided
via the ``dgp`` argument of ``fit``; the optional ``pymc`` / ``dowhy`` extras
are only needed for using those backends::

    pip install "badgers[pymc]"    # PyMC-based backend
    pip install "badgers[dowhy]"   # DoWhy-GCM-based backend
"""

# Generators are re-exported lazily (PEP 562). This avoids a circular import:
# each generator module imports ``causal._base``, which (when imported via its
# full path) would otherwise execute this package ``__init__`` mid-way through
# the import of a generator that is only half initialised. Lazy resolution
# keeps both import styles working:
#   from badgers.generators.tabular_data.causal import CausalNoiseGenerator
#   from badgers.generators.tabular_data.noise.causal import CausalNoiseGenerator
_LAZY_ATTRS = {
    "CausalGenerator": "badgers.generators.tabular_data.causal._base",
    "CausalDriftGenerator": "badgers.generators.tabular_data.drift.causal",
    "CausalMissingnessGenerator": (
        "badgers.generators.tabular_data.missingness.causal"
    ),
    "CausalNoiseGenerator": "badgers.generators.tabular_data.noise.causal",
    "CausalAdditiveNoiseGenerator": (
        "badgers.generators.tabular_data.noise.causal"
    ),
    "CausalOutlierPropagationGenerator": (
        "badgers.generators.tabular_data.outliers.causal"
    ),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    """Resolve generator names on first access (lazy import)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))
