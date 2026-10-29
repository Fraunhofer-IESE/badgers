# Predictor Axes Renderers Design

## Goal

Make the notebook's predictor visualizations composable. Each predictor panel
renderer draws into an Axes supplied by its caller and returns that same Axes.
Notebook cells own figure creation and layout, so readers can arrange or omit
panels without changing rendering logic.

## Scope

This design covers the predictor visualizations only:

- Random Forest classifier map
- Isolation Forest score map
- Random Forest regression curves

The initial clean-data overview and direct/causal generator 2x2 displays are
out of scope.

## API and responsibilities

Replace the complete-figure helpers `boundary_figure` and `regression_figure`
with three panel renderers:

- `plot_classifier(ax, predictions, grid, rgba)` draws the frozen classifier
  map and the stored case observations, including darker misclassified rows.
- `plot_detector(ax, predictions, grid)` draws the frozen detector score map,
  its flag contour, and the stored observations.
- `plot_regression(ax, predictions, regressor, x_grid)` draws the four
  class-conditioned Random Forest step curves on one Axes and overlays the
  case's observed points, including existing error shading and appended-outlier
  marks.

Each function uses the provided Axes, draws only its panel, and returns the
identical Axes object. None creates a Figure, subplot, legend, or colorbar.
Shared module/notebook helpers such as class colors, marker definitions,
`wrong_clf`, `wrong_reg`, and `darken` remain reusable by these renderers.
The regression renderer uses `(x_grid, cls)` as model input for each class,
with `cls` fixed to the corresponding class value.

## Composition

Notebook display cells create figures and axes using `plt.subplots` or another
layout mechanism, then call the renderers in the desired arrangement. The
caller sets titles and axis labels and adds shared legends and colorbars.
Existing default examples remain: classifier and detector panels composed
side-by-side for clean, direct, and causal data; regression curves composed as
one Axes per scenario, with all four classes together in each Axes.

Prediction and grid computation remain separate from rendering. The classifier
and detector continue to use their existing `(x, y)` grids; regression plots
continue to display the continuous `y` target against `x`, conditioning the
prediction curve on class.

## Testing and validation

Add focused tests proving that each renderer returns the exact Axes passed to
it and draws its intended data on that Axes. The regression test also verifies
that four class-specific prediction inputs use the correct fixed class value.
Execute the notebook to confirm its composition cells render without errors,
then run the focused tests and the repository's selected flake8 checks.

## Error behavior

Renderers do not create fallback figures or silently replace missing
prediction columns, grids, or models. Missing required inputs should raise
their normal Python errors at the point of use.
