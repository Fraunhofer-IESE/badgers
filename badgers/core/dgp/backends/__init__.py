"""Backend modules for the DGP protocol.

The backend adapters for externally built causal models
(``GcmSCMDGP`` for ``dowhy.gcm.StructuralCausalModel`` objects,
``PyMCSCMDGP`` for user-built PyMC models) are imported LAZILY
(PEP 562 ``__getattr__`` below) so that importing the core never pulls
their dependencies — the import-hygiene test
(``tests/core/test_import_hygiene.py``) pins that guarantee.

- ``GcmSCMDGP`` (lazy, in ``gcm_scm.py``): adapter for externally built
  ``dowhy.gcm.StructuralCausalModel`` objects (custom mechanisms
  included); needs the ``dowhy`` extra.
- ``PyMCSCMDGP`` (lazy, in ``pymc_scm.py``): adapter for user-built
  PyMC causal models (one ``pm.Model``, full posterior); needs the
  ``pymc`` extra.
"""

_LAZY = {
    "DowhyGcmDGP": "badgers.core.dgp.backends.dowhy_gcm",
    "GcmSCMDGP": "badgers.core.dgp.backends.gcm_scm",
    "PyMCDGP": "badgers.core.dgp.backends.pymc",
    "PyMCSCMDGP": "badgers.core.dgp.backends.pymc_scm",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
