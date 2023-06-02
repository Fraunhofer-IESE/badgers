# Getting started

Install `badgers` with pip:

```
pip install badgers
```

Import badgers as any other library and start using it:

```python
from sklearn.datasets import make_blobs
from badgers.transforms.tabular_data.noise import GaussianNoiseGenerator

X, y = make_blobs()
trf = GaussianNoiseGenerator(noise_std=0.5)
Xt, yt = trf.generate(X, y)
```

More examples are available in the [tutorials](../tutorials/Imbalance-Tabular-Data/) section.

The API documentation is also available in the [API](../reference/badgers/) section.
