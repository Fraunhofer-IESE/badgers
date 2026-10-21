# Getting started:

- section 6 shall introduce "SCM" and make it more explicit what hte code is doing (intervenene and sample)

- all links at the end are broken (example: https://fraunhofer-iese.github.io/badgers/tutorials/concepts/tabular-data/Outliers.ipynb shall be https://fraunhofer-iese.github.io/badgers/tutorials/tabular-data/Outliers.ipynb). Maybe move all tutorials (also the ones in docs/dev) to the same folder docs/tutorials and update the links accordingly.

# Future work

- Re-introduce the `CausalEdgePerturbationGenerator` (relationship-level
  defect: scale one edge coefficient by `weight_factor`, recompute the
  child marginal-preserving, propagate the descendants through the DGP).
  It was removed before the 0.0.15 release; the underlying DGP edit
  (`edits.scale_edge` in `badgers.core.dgp.edits`) is still shipped and
  tested, so the generator can be re-added on top of it — see the
  git history (`docs/tutorials/tabular-data/Edge-Perturbation.ipynb`,
  `badgers/generators/tabular_data/relationship/causal.py`).