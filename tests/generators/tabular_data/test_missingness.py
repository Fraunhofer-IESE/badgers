import numpy as np

from badgers.generators.tabular_data.missingness.missingness import MissingValueGenerator


def test_missingness__preserves_shape_and_inserts_nans(tabular_data):
    """All MissingValueGenerator subclasses preserve shape and insert correct NaN count."""
    X = np.asarray(tabular_data)
    n_features = X.shape[1] if X.ndim > 1 else 1
    total_cells = X.shape[0] * n_features
    percentage_missing = 0.1

    for cls in MissingValueGenerator.__subclasses__():
        transformer = cls(random_generator=np.random.default_rng(0))
        Xt = transformer.generate(X.copy(), percentage_missing=percentage_missing)
        assert Xt.shape[1] == n_features
        assert np.sum(np.isnan(Xt).sum()) == int(percentage_missing * total_cells)
