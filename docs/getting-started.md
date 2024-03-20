# Getting started

Install `badgers` with pip:

```
pip install badgers
```

Import badgers as any other library and start using it:

```python
from sklearn.datasets import make_blobs
from badgers.generators.tabular_data.noise import GlobalGaussianNoiseGenerator

X, y = make_blobs()
trf = GlobalGaussianNoiseGenerator(noise_std=0.5)
Xt, yt = trf.generate(X, y)
```

More examples are available in the [tutorials](../tutorials/Imbalance-Tabular-Data/) section.

The API documentation is also available in the [API](../reference/badgers/) section.
