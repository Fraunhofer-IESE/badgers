# Notebooks
## Tabular

✅ ALL DONE in 0.0.15 (commit 9d7e1b8): the causal/non-causal pairs were
merged into one notebook per deficit (`Outliers.ipynb`, `Noise.ipynb`,
`Missingness.ipynb`, `Imbalance.ipynb`, `Drift.ipynb`,
`Edge-Perturbation.ipynb`), each with **Part A** (observation process, no
DGP) and **Part B** (acting on the DGP), explicit `cls`/`labels` vectors in
visualizations, and the content fixes listed per notebook below (Noise
ClassSelector vs noise_std, Missingness explicit selectors +
MCAR/MAR/MNAR, Imbalance `y` renamed to `score` + repaired last section,
Drift `drift_std` semantics, Causal Outlier shortened). The old pair
notebooks were removed; mkdocs nav points at the merged notebooks.

# API
## Tabular

✅ DONE in 0.0.15:
- `DecompositionAndOutlierGenerator` renamed to `StructuredOutlierGenerator`
  (backwards-compatible alias kept) with an optional `preprocessor=`
  constructor argument (preprocessing is not badgers' responsibility; no
  new object). Pipelining stays possible via `badgers.core.pipeline`.
- `CausalImbalanceGenerator` accepts an explicit `selector=` (fired rows
  are dropped; `quantile_range` remains the `selector=None` default).