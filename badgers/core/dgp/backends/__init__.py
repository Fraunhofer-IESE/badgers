"""Backend modules for the DGP protocol.

The default ``OlsDGP`` backend is pure NumPy and is imported eagerly by the
``badgers.core.dgp`` package; heavier estimation libraries (PyMC, DoWhy)
live in their own backend modules under this package and are imported
lazily inside those modules.
"""
