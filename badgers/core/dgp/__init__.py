"""Per-node, backend-agnostic data generation process (DGP).

Public surface:

- ``DGP`` — the backend-agnostic protocol (``DGPMixin`` provides the
  default ``fit``/``sample`` topological walk);
- ``do`` / ``ordered_descendants`` — the generic deficit algebra;
- ``OlsDGP`` — the default deterministic backend (pure NumPy);
- ``ScaledEdgeDGP`` — a read-only view of a fitted DGP with one edge's
  contribution scaled (variance-preserving noise rescaling).

Heavy estimation libraries (PyMC, DoWhy) live in their own backend modules
under ``badgers.core.dgp.backends`` and are imported lazily **inside those
modules**; they are never imported by this package.
"""

from badgers.core.dgp.algebra import (
    ScaledEdgeDGP,
    do,
    ordered_descendants,
)
from badgers.core.dgp.backends.ols import OlsDGP
from badgers.core.dgp.base import DGPMixin
from badgers.core.dgp.protocol import DGP

__all__ = [
    "DGP",
    "DGPMixin",
    "OlsDGP",
    "ScaledEdgeDGP",
    "do",
    "ordered_descendants",
]
