# Getting started:

- the link to tutorial is dead

# Future work

- Re-introduce the `CausalEdgePerturbationGenerator` (relationship-level
  defect: scale one edge coefficient by `weight_factor`, recompute the
  child marginal-preserving, propagate the descendants through the DGP).
  It was removed before the 0.0.15 release; the underlying DGP edit
  (`edits.scale_edge` in `badgers.core.dgp.edits`) is still shipped and
  tested, so the generator can be re-added on top of it — see the
  git history (`docs/tutorials/tabular-data/Edge-Perturbation.ipynb`,
  `badgers/generators/tabular_data/relationship/causal.py`).