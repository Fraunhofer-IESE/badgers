import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# -- Shared SCM-adapter DGP factory ------------------------------------------

def make_gcm_scm(graph):
    """Build a ``dowhy.gcm.StructuralCausalModel`` for ``graph``.

    Root nodes get ``EmpiricalDistribution`` mechanisms; non-root nodes
    get linear ``AdditiveNoiseModel``s (the same estimation family the
    removed ``OlsDGP`` used, modulo its zero intercept). Requires the
    ``dowhy`` extra.
    """
    import dowhy.gcm as gcm

    cm = gcm.StructuralCausalModel(graph)
    for node in graph.nodes:
        if not list(graph.predecessors(node)):
            cm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        else:
            cm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    prediction_model=gcm.ml.create_linear_regressor()
                ),
            )
    return cm


@pytest.fixture
def make_gcm_dgp():
    """Factory fixture: ``make_gcm_dgp(graph, mapping=None)`` returns an
    UNFITTED ``GcmSCMDGP`` wrapping an SCM built by :func:`make_gcm_scm`
    (EmpiricalDistribution roots, linear ANM non-roots). Skips the test
    when the ``dowhy`` extra is not installed."""
    pytest.importorskip("dowhy.gcm")
    from badgers.core.dgp.backends.gcm_scm import GcmSCMDGP

    def _factory(graph, mapping=None):
        return GcmSCMDGP(make_gcm_scm(graph), mapping)

    return _factory
