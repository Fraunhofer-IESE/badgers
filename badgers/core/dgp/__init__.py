"""Per-node, backend-agnostic data generation process (DGP).

Public surface:

- ``DGP`` — the backend-agnostic protocol: per-node mechanisms
  (``fit_node``/``response``/``draw_noise``), joint ``sample``, and the
  value-level ``intervene`` (Pearl surgery on given rows). The DGP is
  an immutable, fitted model — it has NO structure-edit methods;
- ``DGPMixin`` — the generic reference implementations of everything
  above except ``fit_node`` (backends may override any method natively,
  preserving the protocol semantics);
- ``edits`` — the structure-modification vocabulary as free functions
  (``scale_edge``/``remove_edge``/``add_edge_contribution``/
  ``exogenize``/``set_form``), returning
  NEW DGPs;
- ``backends`` — the concrete backend modules
  (``badgers.core.dgp.backends``);
- ``ordered_descendants`` — topological descendant walk (in
  ``badgers.core.causal_graph``).

Heavy estimation libraries (PyMC, DoWhy) live in their own backend modules
under ``badgers.core.dgp.backends`` and are imported lazily **inside those
modules**; they are never imported by this package.
"""

from badgers.core.causal_graph import ordered_descendants
from badgers.core.dgp.base import DGPMixin
from badgers.core.dgp.edits import (
    add_edge_contribution,
    exogenize,
    remove_edge,
    scale_edge,
    set_form,
)
from badgers.core.dgp.protocol import DGP

__all__ = [
    "DGP",
    "DGPMixin",
    "ordered_descendants",
    "scale_edge",
    "remove_edge",
    "add_edge_contribution",
    "exogenize",
    "set_form",
]
