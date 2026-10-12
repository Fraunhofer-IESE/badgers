"""Per-node, backend-agnostic data generation process (DGP).

Public surface:

- ``DGP`` — the backend-agnostic protocol;
- ``do`` / ``ordered_descendants`` — the generic deficit algebra;
- ``OlsDGP`` — the default deterministic backend (pure NumPy).

Heavy estimation libraries (PyMC, DoWhy) live in their own backend modules
under ``badgers.core.dgp.backends`` and are imported lazily **inside those
modules**; they are never imported by this package.
"""

from badgers.core.dgp.algebra import do, ordered_descendants

__all__ = [
    "do",
    "ordered_descendants",
]
