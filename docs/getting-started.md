# Getting started

Install `badgers` with pip:

```
pip install badgers
```

Import badgers as any other library and start using it:

```python
from sklearn.datasets import make_blobs
from badgers.transforms.tabular_data.noise import GaussianNoiseTransformer

X, y = make_blobs()
trf = GaussianNoiseTransformer(noise_std=0.5)
Xt = trf.transform(X)
```

More examples are available in the [tutorials](tutorials/Imbalance-Tabular-Data/) section.

The API documentation is also available in the [API](reference/badgers/) section.