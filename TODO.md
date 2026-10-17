# Notebooks
## Tabular

### ALL: 

- Reflect upon the need for separate notebooks causal vs non causal, or if in a single notebook fro each defect (I would rather have one notebook)

- We need more information (more explanations about what the goal of the deficit is and how it can be achieved in badgers)

- Old "tabular" shall emphasize that they are not using any DGP and only act on the obervation process

- Visualization uses y and yt as label placeholders: change to explicit class / labels vectors

### Outliers tabular:

(uses y and yt, see above)

### Noise tabular: 

- More information about the ClassSelector used. Especially the difference between the  {"cls": {0: 1., 1: 1., 2: 1., 3: 1.}} and the noise_std_per_class={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}. Explain clearly the different purposes (selector: which rows will be selected within the column "cls" vs the strenght of the noise applied to each separate class)

### Missingness Tabular: 

- it shall explicitely make use of Selectors!
- it shall also showcase the difference between MCAR (we shall see that missingness is random), MAR (missingness depends on the X-Axis value, so more missing values on the right side thant the left side), MNAR, we shall see that missingness applies more to a certain type of points

### Imbalance Tabular:

- uses `y = -2 * X[:, 0] + 3 * X[:, 1] + 5` rename it.
- make the use of selector explicit (explain what it's purpose is)
- the last section is not clear. the selector is only built for visualzation purpose and the `sampling_proba_func(X)` is not explicitly used!

### Drift Tabular and Causal

- drift_std is missleading. I understand the drift as either positive of negative value (normalized to sigma for each X columns), if drift = 2 for a given X_i column, then the underlying idea was to do X_i <- X_i + 2 Sigma (because we normalize X in the first place, sigma = 1 for each column), so it is basically: normalize X, add 2 to the selected column, "un-normalize", propagate 


### Causal Outlier:

- Way too long (comes from a previous version when the contracts and interfaces and causal objects were not defined), repeats stuff that is already in getting started!!!

# API
## Tabular
- Decomposition and Outlier (first naming: "decomposition" not really good): do we need a preprocessor object? or do we leave it out (not badgers responsibility), what about pipelining?

- Causal Imbalance shall make use of an explicit selector! (not only quantile, it could also be proportion, etc.)