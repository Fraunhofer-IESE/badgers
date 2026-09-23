"""Causal data quality defect generators.

The causal defect family generates bad data by perturbing the *structure*
(causal graph + data generation process) of the data, rather than the raw
values. Each generator fits a ``PyMCDGP`` to the data, applies a structural
perturbation, then propagates the effect through the descendants.

This subpackage requires the optional ``causal`` extra (PyMC + PyTensor)::

    pip install "badgers[causal]"
"""

from badgers.generators.tabular_data.causal._base import CausalGenerator
from badgers.generators.tabular_data.drift.causal import CausalDriftGenerator
from badgers.generators.tabular_data.imbalance.causal import (
    CausalImbalanceGenerator,
)
from badgers.generators.tabular_data.missingness.causal import (
    CausalMissingnessGenerator,
)
from badgers.generators.tabular_data.noise.causal import CausalNoiseGenerator
from badgers.generators.tabular_data.relationship.causal import (
    CausalEdgePerturbationGenerator,
)

__all__ = [
    "CausalGenerator",
    "CausalDriftGenerator",
    "CausalImbalanceGenerator",
    "CausalMissingnessGenerator",
    "CausalNoiseGenerator",
    "CausalEdgePerturbationGenerator",
]
